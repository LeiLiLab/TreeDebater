"""Replay frozen helper schema checks and account for unchanged recovery behavior."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from statistics import mean, median

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from scripts.benchmark_incremental_planning import code_digest
from utils.llm_schemas import BattlefieldResponse, StatementsResponse, LinkedStatementsResponse
from utils.tool import parse_llm_json

SCHEMAS={'BattlefieldResponse':(BattlefieldResponse,'response'),
         'StatementsResponse':(StatementsResponse,'statements'),
         'LinkedStatementsResponse':(LinkedStatementsResponse,'statements')}


def diagnose(run_id='flat-linear-legacy-v1'):
    base=ROOT/'experiments/incremental_planning';run=base/'run'/run_id
    metadata=json.loads((run/'metadata_worker0.json').read_text())
    if code_digest()!=metadata['source_digest']:
        raise ValueError('Must use frozen source validator')
    results=[json.loads(p.read_text()) for p in run.glob('*__*.json')]
    db=sqlite3.connect(f'file:{base/"run/cost.sqlite"}?mode=ro',uri=True)
    records=[];groups=defaultdict(list);by_label=defaultdict(list)
    for request_id,label,state in db.execute('select id,label,state from calls order by id'):
        if not label.startswith(run_id+'/'):continue
        if state=='pending':raise ValueError('Wait for all requests to finish')
        artifact=json.loads((base/'run'/f'call_{request_id:06}.json').read_text())
        request=artifact['request'];title=None
        for msg in request['messages']:
            content=msg.get('content')
            if isinstance(content,str) and '\nReturn JSON matching this schema:\n' in content:
                title=json.loads(content.split('\nReturn JSON matching this schema:\n')[-1]).get('title')
        if title not in SCHEMAS:continue
        schema,key=SCHEMAS[title];error='';valid=False
        try:
            raw=artifact['response']['choices'][0]['message']['content']
            parse_llm_json(raw,response_model=schema,required_key=key);valid=True
        except (ValueError,KeyError,TypeError) as exc:
            error=str(exc)[:1200]
        record=dict(id=request_id,label=label,mode=label.split('/')[2],schema=title,
                    schema_valid=valid,error=artifact.get('error') or error,truncated=artifact.get('truncated',False))
        records.append(record);by_label[label].append(record)
        signature=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        groups[label,signature].append(record)
    recoveries=[]
    for (label,signature),entries in groups.items():
        if len(entries)>1 and any(not e['schema_valid'] for e in entries[:-1]):
            recoveries.append(dict(label=label,schema=entries[0]['schema'],request_sha256=signature,
                request_ids=[e['id'] for e in entries],attempts=len(entries),success=entries[-1]['schema_valid'],
                invalid_attempts=sum(not e['schema_valid'] for e in entries)))
    metrics=[]
    for mode in metadata['modes']:
        rs=[r for r in results if r['mode']==mode];calls=[r for r in records if r['mode']==mode]
        waits=[]
        for r in rs:
            label=f'{run_id}/{r["case"]}/{mode}/{r["repeat"]}'
            # Battlefield preparation is after the final input and before text-ready.
            count=sum(c['schema']=='BattlefieldResponse' and not c['schema_valid'] for c in by_label[label])
            waits.append(dict(case=r['case'],repeat=r['repeat'],battlefield_schema_failures=count,
                known_battlefield_wait_seconds=30*count,observed_text_seconds=r['estimated_residual_text_seconds'],
                text_seconds_minus_known_battlefield_wait=r['estimated_residual_text_seconds']-30*count))
        latencies=sorted(r['estimated_residual_text_seconds'] for r in rs)
        metrics.append(dict(mode=mode,answers=len(rs),helper_requests=len(calls),schema_failures=sum(not c['schema_valid'] for c in calls),
            battlefield_requests=sum(c['schema']=='BattlefieldResponse' for c in calls),
            battlefield_schema_failures=sum(c['schema']=='BattlefieldResponse' and not c['schema_valid'] for c in calls),
            identical_request_recovery_groups=sum(g['label'].split('/')[2]==mode for g in recoveries),
            exhausted_recovery_groups=sum(g['label'].split('/')[2]==mode and not g['success'] for g in recoveries),
            mean_text_seconds=mean(latencies),median_text_seconds=median(latencies),
            p90_text_seconds_nearest_rank=latencies[math.ceil(.9*len(latencies))-1],max_text_seconds=max(latencies),
            mean_text_seconds_minus_known_battlefield_wait=mean(w['text_seconds_minus_known_battlefield_wait'] for w in waits),
            per_answer_wait_accounting=waits))
    report=dict(run_id=run_id,source_digest=metadata['source_digest'],metrics=metrics,helper_checks=records,recovery_groups=recoveries,
        paid_requests=0,limitations=[
            'Original pipeline behavior is unchanged: helper error recovery allows three attempts and sleeps 30 seconds after each failure, including the final exhausted attempt.',
            'Primary latency is observed simulated text-ready latency with all waits included. Subtracted-wait figures are arithmetic sensitivity checks, not a rerun or evidence of a fixed pipeline.',
            'Only battlefield sleeps are subtracted because their endpoint placement is known; extraction work and all model calls remain included.',
            'Schema validation does not establish semantic correctness. Existing exact-string legacy matching can also fail without invalidating JSON.'])
    (base/f'{run_id}_helper_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in m.items() if k!='per_answer_wait_accounting'} for m in metrics],indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run_id',nargs='?',default='flat-linear-legacy-v1');diagnose(p.parse_args().run_id)
