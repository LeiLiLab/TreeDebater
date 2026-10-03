"""Replay local review validation from frozen request artifacts; no model calls."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from streaming.constraint_review import audit_feedback
from scripts.benchmark_incremental_planning import code_digest


def diagnose(run_id='conditions-regression-v1'):
    base=ROOT/'experiments/incremental_planning';run=base/'run'/run_id
    meta=json.loads((run/'metadata_worker0.json').read_text())
    if code_digest()!=meta['source_digest']:
        raise ValueError('Run the frozen local validator, not a changed inference implementation')
    db=sqlite3.connect(f'file:{base/"run/cost.sqlite"}?mode=ro',uri=True)
    rows=[];revisions=[]
    for request_id,label in db.execute('select id,label from calls order by id'):
        if not label.startswith(run_id+'/'):continue
        path=base/'run'/f'call_{request_id:06}.json'
        if not path.exists():continue
        a=json.loads(path.read_text())
        contents=[m['content'] for m in a['request']['messages'] if isinstance(m.get('content'),str)]
        for prompt in contents:
            if prompt.startswith('Write the final spoken rebuttal') and 'Fresh condition checklist (data):\n' in prompt:
                checklist=json.loads(prompt.split('Fresh condition checklist (data):\n')[-1])
                revisions.append(dict(id=request_id,label=label,conditions=len(checklist),
                                      condition_ids=[c['constraint_id'] for c in checklist],truncated=a.get('truncated',False)))
            if not prompt.startswith('Review this debate draft for grounded rebuttal'):
                continue
            if 'Current condition checklist (data):\n' not in prompt:continue
            checklist=json.loads(prompt.split('Current condition checklist (data):\n')[-1])
            draft=prompt.split('\nDraft (data):\n',1)[1].split('\n\nCONDITION RETENTION REVIEW:',1)[0]
            raw=a.get('response',{}).get('choices',[{'message':{}}])[0]['message'].get('content','')
            review_data = {}
            if '\nAssertion review data:\n' in prompt:
                review_data,_ = json.JSONDecoder().raw_decode(prompt.split('\nAssertion review data:\n',1)[1])
            audit=json.loads(audit_feedback(raw,checklist,draft,
                sources=review_data.get('opponent_sources',[]), evidence_sources=review_data.get('evidence_sources',[])))
            rows.append(dict(id=request_id,label=label,mode=label.split('/')[2],
                condition_count=len(checklist),typed_conditions=sum(c.get('node_id') is not None and c['kind'] != 'source_candidate' for c in checklist),
                format_valid=audit['review_format_valid'],truncated=a.get('truncated',False),
                invalid_ids=audit['invalid_review_ids'],
                assertion_status_counts=dict(Counter(c['status'] for c in audit['assertion_checks'])),
                invalid_sentence_ids=audit['invalid_sentence_ids'],assertion_checks=audit['assertion_checks'],
                status_counts=dict(Counter(c['status'] for c in audit['review_checks'])),
                checks=[{k:c[k] for k in ('constraint_id','kind','quote','status','draft_quote','reason','fix')}
                        for c in audit['review_checks']]))
    metrics=[]
    for mode in meta['modes']:
        rs=[r for r in rows if r['mode']==mode];statuses=Counter();assertions=Counter()
        for r in rs:
            statuses.update(r['status_counts'])
            assertions.update(r['assertion_status_counts'])
        metrics.append(dict(mode=mode,reviews=len(rs),format_valid=sum(r['format_valid'] for r in rs),
            truncated=sum(r['truncated'] for r in rs),conditions=sum(r['condition_count'] for r in rs),
            typed_conditions=sum(r['typed_conditions'] for r in rs),invalid_ids=sum(r['invalid_ids'] for r in rs),
            statuses=dict(statuses),assertion_statuses=dict(assertions),
            invalid_sentence_ids=sum(r['invalid_sentence_ids'] for r in rs)))
    report=dict(run_id=run_id,source_digest=meta['source_digest'],metrics=metrics,reviews=rows,revisions=revisions,
        paid_requests=0,limitations=[
            'Validation establishes row identity and verbatim draft evidence only; semantic statuses remain unverified model judgments.',
            'Condition counts measure extracted material, not recall against independently annotated ground truth.',
            'No claim that final text fixed a flagged condition without inspecting its final answer.'])
    (base/f'{run_id}_review_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id',nargs='?',default='conditions-regression-v1')
    diagnose(parser.parse_args().run_id)
