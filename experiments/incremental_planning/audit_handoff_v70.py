"""Read-only published-prefix/version, overlap and transferred-body audit."""
import hashlib
import json
from pathlib import Path
import sqlite3
D=Path('experiments/incremental_planning')
RUN='listening-motion-live-v52-refactor-streamed-motion03-v70'
O=D/'run'/RUN

def read(p,default=None):
    return json.loads(p.read_text()) if p.exists() else default

def main():
    db=sqlite3.connect(f'file:{D}/run/cost.sqlite?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    requests=[]
    for r in db.execute('select id,label,state,seconds from calls where instr(label,?)=1',(RUN,)):
        item=read(D/f'run/call_{r["id"]:06d}.json',{})
        messages=item.get('request',{}).get('messages',[])
        for m in messages:
            prompt=m.get('content','')
            if isinstance(prompt,str) and prompt.startswith('LISTENING PREFIX REVIEW:'):
                payload=json.loads(prompt.splitlines()[-1])
                requests.append(dict(r)|dict(side=payload['context']['our_side'],stage=payload['context']['stage'],text=payload['draft']))
    rows=[];handoffs=[];previous=None;fixes=[]
    for folder in sorted(O.glob('[0-9]*')):
        if not folder.is_dir():continue
        r=read(folder/'result.json',{})
        if not r:continue
        events=read(folder/'events.json',[])
        paths=list(folder.glob('*_chunks/listening_prefix.json'))
        trace=read(paths[0],{}) if paths else {}
        state=trace.get('initial_prefix_review',{})
        versions=state.get('versions',[])
        published=events[0]['text'] if events else None
        matches=[v for v in versions if v['checked_text']==published and (v.get('verdict') or {}).get('accepted')]
        rows.append(dict(turn=folder.name,status=r.get('status'),published_text=published,
            published_matches_accepted_version=bool(matches),accepted_review_stamps=[v['review_stamp'] for v in matches],
            versions=versions,semantic_rewrites=trace.get('semantic_rewrites'),
            review_requests=[q for q in requests if q['side']==r['side'] and q['stage']==r['stage']]))
        endpoint=r.get('previous_opponent_endpoint_monotonic')
        full_asr=previous.get('full_asr_ready_monotonic') if previous else None
        if endpoint is not None and events:
            first=events[0]['ready_monotonic']
            handoffs.append(dict(turn=folder.name,endpoint_to_first_audio_seconds=first-endpoint,
                endpoint_to_full_asr_seconds=full_asr-endpoint if full_asr is not None else None,
                first_audio_before_full_asr=first<full_asr if full_asr is not None else None))
        transfer=trace.get('prefix_audio_transfer',{})
        fb=trace.get('parallel_body_feedback',{})
        body=trace.get('body_transfer',{})
        fs=fb.get('start_seconds');ar=transfer.get('ready_seconds')
        fixes.append(dict(turn=folder.name,publication_mode=trace.get('body_publication_mode'),
            prefix_audio_transfer=transfer,feedback_start_seconds=fs,
            feedback_started_before_transferred_audio_ready=fs<ar if fs is not None and ar is not None else None,
            body_transfer=body,body_context_source=trace.get('body_context_snapshot',{}).get('source_stamp'),
            body_draft_source=trace.get('body_context_snapshot',{}).get('draft_source_stamp'),
            limitation='No pending audio/body transfer means that branch was not exercised on this turn. Adoption alone does not prove late adoption unless timing also establishes it.'))
        previous=r
    hashes=read(O/'source_snapshot/files.json',{})
    changed=[p for p,h in hashes.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    review=dict(turns=rows,review_request_count=len(requests),source_changed=changed,
        all_published_prefixes_reviewed=bool(rows) and all(r['published_matches_accepted_version'] for r in rows))
    (O/'initial_review_audit.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
    (O/'handoff_latency_audit.json').write_text(json.dumps(dict(handoffs=handoffs),indent=2)+'\n')
    (O/'handoff_fix_audit.json').write_text(json.dumps(dict(turns=fixes),indent=2)+'\n')
    print(json.dumps(dict(turns=len(rows),review_request_count=len(requests),all_published_prefixes_reviewed=review['all_published_prefixes_reviewed'],changed_sources=changed,handoffs=handoffs,fixes=fixes),indent=2))

if __name__=='__main__':main()
