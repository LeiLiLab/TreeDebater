"""Reparse frozen requests to isolate parser compatibility; zero model calls.

This is a counterfactual offline parser replay, not new scores or regenerated plans.
Original answers, judgments, diagnostics and request artifacts are never changed.
"""
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from streaming.grounding import parse_state


def replay():
    base=ROOT/'experiments/incremental_planning';run='conditions-regression-v1'
    old=json.loads((base/f'{run}_diagnostics.json').read_text())
    rejected={r['id']:r['reason'] for r in old['invalid_snapshots']}
    db=sqlite3.connect(f'file:{base/"run/cost.sqlite"}?mode=ro',uri=True)
    results=[]
    for request_id,label in db.execute('select id,label from calls order by id'):
        if not label.startswith(run+'/') or label.split('/')[2]!='grounded_tree':continue
        artifact=json.loads((base/'run'/f'call_{request_id:06}.json').read_text())
        prompts=[m['content'] for m in artifact['request']['messages']
                 if isinstance(m.get('content'),str) and m['content'].startswith('Prepare a compact JSON snapshot')]
        if not prompts:continue
        prompt=prompts[0];payload,_=json.JSONDecoder().raw_decode(prompt[prompt.index('{"context":'):])
        error=None
        try:
            state=parse_state(artifact['response']['choices'][0]['message']['content'],
                              ' '.join(payload['heard_prefix']),tree_targets=payload['context']['tree_targets'])
        except (ValueError,KeyError,TypeError) as exc:error=str(exc)
        results.append(dict(id=request_id,label=label,before_error=rejected.get(request_id),after_error=error))
    assert len(results)==64
    assert not any(r['before_error'] is None and r['after_error'] is not None for r in results)
    report=dict(original_run=run,paid_requests=0,snapshots=len(results),
                accepted_before=sum(r['before_error'] is None for r in results),
                accepted_after=sum(r['after_error'] is None for r in results),
                recovered=sum(r['before_error'] is not None and r['after_error'] is None for r in results),
                results=results,limitations=['Frozen completions only; changes in prompting and generation need fresh evaluation.',
                                            'Accepted source attribution/shape does not validate semantic entailment.'])
    (base/'condition-repair-parser-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':replay()
