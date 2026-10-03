"""Offline source/graph/state audit for repaired tree experiments; no model calls."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from debate_tree import DebateTree
from streaming.tree_grounding import tree_targets
from streaming.grounding import parse_state, normalize
from streaming.branch_planning import parse_branch_state


def diagnose(run_id, cases_file):
    base=ROOT/'experiments/incremental_planning';run=base/'run'/run_id
    cases={c['id']:c for c in json.loads(Path(cases_file).read_text())}
    results=[json.loads(p.read_text()) for p in sorted(run.glob('*__*.json'))]
    bindings=[];integrity=[];events=defaultdict(Counter);states=Counter();invalid=[];latest={}
    for result in results:
        if result['mode'] not in ('grounded_tree','light_tree','flat_tree','branch_tree'):continue
        case=cases[result['case']];before=result['before_generation'];opponent='for' if case['side']=='against' else 'against'
        trees=[DebateTree.from_json(before[k]) for k in ('our_tree','opponent_tree')]
        active={n['node_id']:n for n in tree_targets(trees,opponent)}
        selected=before['state'].get('claims',[])
        bindings.append(dict(case=result['case'],repeat=result['repeat'],mode=result['mode'],
            active_targets=len(active),selected_claims=len(selected),
            final_raw_prefix_fallback=not bool(before['state']),
            selected_versions_valid=all(c['node_id'] in active and c['target_version']==active[c['node_id']]['version'] for c in selected),
            selected_reply_paths=sum(bool(active[c['node_id']]['ancestors']) for c in selected if c['node_id'] in active),
            retained_position_limits=len(before['state'].get('position_limits',[]))))
        heard={side:[] for side in ('for','against')}
        heard[case['side']].append(case['own_opening']);heard[opponent].extend(case['chunks'])
        for speech in case.get('prior_history',[]):heard[speech['side']].append(speech['content'])
        for tree in trees:
            for node in tree.get_all_nodes():
                if node.parent is None:continue
                if node.parent.parent is not None and node.side==node.parent.side:
                    integrity.append(dict(case=result['case'],mode=result['mode'],issue='same-speaker response edge',node_id=node.node_id))
                for quote in node.source_spans:
                    if not any(normalize(quote) in normalize(text) for text in heard[node.side]):
                        # A buffered extraction quote can cross two current chunks.
                        if normalize(quote) not in normalize(' '.join(heard[node.side])):
                            integrity.append(dict(case=result['case'],mode=result['mode'],issue='source not in speaker history',node_id=node.node_id))
            events[result['mode']].update(e['action'] for e in getattr(tree,'update_events',[]))
    db=sqlite3.connect(f'file:{base/"run/cost.sqlite"}?mode=ro',uri=True)
    for request_id,label in db.execute('select id,label from calls order by id'):
        if not label.startswith(run_id+'/'):continue
        artifact=json.loads((base/'run'/f'call_{request_id:06}.json').read_text())
        prompts=[m['content'] for m in artifact['request']['messages'] if isinstance(m.get('content'),str)
                 and m['content'].startswith(('Prepare a compact JSON snapshot','Prepare compact JSON rebuttal choices'))]
        if not prompts:continue
        prompt=prompts[0];payload,_=json.JSONDecoder().raw_decode(prompt[prompt.index('{"context":'):]);context=payload['context']
        mode=label.split('/')[2];states[mode]+=1;latest[tuple(label.split('/')[1:])]=context.get('tree_targets',[])
        try:
            raw=artifact['response']['choices'][0]['message']['content']
            if mode in ('flat_tree','branch_tree'):parse_branch_state(raw,' '.join(payload['heard_prefix']),context)
            else:parse_state(raw,' '.join(payload['heard_prefix']),tree_targets=context.get('tree_targets'))
        except (ValueError,KeyError,TypeError) as error:
            invalid.append(dict(id=request_id,label=label,reason=str(error),supplied_targets=len(context.get('tree_targets',[]))))
    for binding in bindings:
        targets={n['node_id']:n for n in latest[binding['case'],binding['mode'],str(binding['repeat'])]}
        result=next(r for r in results if (r['case'],r['mode'],r['repeat'])==(binding['case'],binding['mode'],binding['repeat']))
        binding['request_versions_valid']=all(c['node_id'] in targets and c['target_version']==targets[c['node_id']]['version']
            for c in result['before_generation']['state'].get('claims',[]))
    by_mode=[]
    for mode in sorted({b['mode'] for b in bindings}):
        selected=[b for b in bindings if b['mode']==mode]
        by_mode.append(dict(mode=mode,answers=len(selected),with_bound_targets=sum(b['selected_claims']>0 for b in selected),
            final_raw_fallbacks=sum(b['final_raw_prefix_fallback'] for b in selected),
            graph_update_events=dict(events[mode]),structured_snapshots=states[mode],
            rejected_snapshots=sum(i['label'].split('/')[2]==mode for i in invalid),
            selected_reply_paths=sum(b['selected_reply_paths'] for b in selected)))
    report=dict(run_id=run_id,metrics=by_mode,final_binding_checks=bindings,integrity_issues=integrity,
        structured_snapshot_counts=dict(states),invalid_snapshots=invalid,paid_requests=0,
        limitations=['Speaker/source ownership and version consistency do not validate semantic entailment of extracted claims.',
                     'An edge records a response, not whether the issue was resolved.',
                     'Counts include historical setup and all observed chunks; repeated cases are not independent samples.'])
    (base/f'{run_id}_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'metrics':by_mode,'integrity_issues':integrity},indent=2))
    assert not integrity
    assert all(b['selected_versions_valid'] and b['request_versions_valid'] for b in bindings)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('--cases-file',default=str(ROOT/'experiments/incremental_planning/cases_v4.json'))
    args=parser.parse_args();diagnose(args.run_id,args.cases_file)
