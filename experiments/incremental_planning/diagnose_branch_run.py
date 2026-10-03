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
from streaming.branch_planning import parse_branch_state, planning_material, material_version
from streaming.tree_selection import select_nodes, is_current


def diagnose(run_id, cases_file):
    base=ROOT/'experiments/incremental_planning';run=base/'run'/run_id
    cases={c['id']:c for c in json.loads(Path(cases_file).read_text())}
    metadata=json.loads(next(run.glob('metadata_worker*.json')).read_text())
    limits=metadata.get('tree_limits', {'max_tree_targets':8,'max_tree_context_nodes':16})
    selection_limits=dict(max_targets=limits['max_tree_targets'],max_context_nodes=limits['max_tree_context_nodes'])
    results=[json.loads(p.read_text()) for p in sorted(run.glob('*__*.json'))]
    bindings=[];integrity=[];events=defaultdict(Counter);states=Counter();invalid=[];latest={}
    concessions=Counter();phase_usage=defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0,usage_estimate_usd=0.))
    for result in results:
        if result['mode'] not in ('grounded_tree','light_tree','flat_tree','branch_tree'):continue
        case=cases[result['case']];before=result['before_generation'];opponent='for' if case['side']=='against' else 'against'
        trees=[DebateTree.from_json(before[k]) for k in ('our_tree','opponent_tree')]
        active={n['node_id']:n for n in tree_targets(trees,opponent,**selection_limits)}
        stored=[n for t in trees for n in t.get_all_nodes() if n.parent is not None]
        eligible=[n for n in stored if n.side==opponent and is_current(n) and n.source_spans]
        target_nodes, context_nodes=select_nodes(trees,opponent,**selection_limits)
        all_targets, all_context=select_nodes(trees,opponent,max_targets=max(1,len(stored)),
                                             max_context_nodes=max(1,len(stored)))
        actual_ids={n.node_id for n in target_nodes+context_nodes}
        all_view_ids={n.node_id for n in all_targets+all_context}
        assert len(active)<=selection_limits['max_targets']
        assert len(context_nodes)<=selection_limits['max_context_nodes']
        assert all(is_current(n) for n in target_nodes+context_nodes)
        assert len(actual_ids)==len(target_nodes)+len(context_nodes)
        coverage_valid=True
        if result['mode'] in ('flat_tree','branch_tree') and before['state']:
            material=planning_material(list(active.values()),trees,opponent,topology=result['mode']=='branch_tree')
            coverage_valid=before['state'].get('material_version')==material_version(material)
        selected=before['state'].get('claims',[])
        bindings.append(dict(case=result['case'],repeat=result['repeat'],mode=result['mode'],
            active_targets=len(active),selected_claims=len(selected),
            stored_nodes=len(stored),historical_or_dependent_nodes=sum(not is_current(n) for n in stored),
            eligible_opponent_nodes=len(eligible),omitted_eligible_targets=len(eligible)-len(active),
            additional_context_nodes=len(context_nodes),
            omitted_nodes_from_full_current_view=len(all_view_ids-actual_ids),
            context_cap_saturated=len(context_nodes)==limits['max_tree_context_nodes'],
            coverage_material_version_valid=coverage_valid,
            target_cap_binding=len(eligible)>limits['max_tree_targets'],
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
            concessions[result['mode']]+=sum(e.get('relation')=='concede' for e in getattr(tree,'update_events',[]))
    db=sqlite3.connect(f'file:{base/"run/cost.sqlite"}?mode=ro',uri=True)
    for request_id,label,input_tokens,output_tokens,cost in db.execute('select id,label,input_tokens,output_tokens,estimated_usd from calls order by id'):
        if not label.startswith(run_id+'/'):continue
        artifact=json.loads((base/'run'/f'call_{request_id:06}.json').read_text())
        contents=[m['content'] for m in artifact['request']['messages'] if isinstance(m.get('content'),str)]
        mode=label.split('/')[2]
        if artifact['request']['model']=='gpt-5.6-sol':phase='judging'
        elif any(s.startswith(('Prepare a compact JSON snapshot','Prepare compact JSON rebuttal choices','Prepare concise private rebuttal notes')) for s in contents):phase='planning'
        elif any('Return JSON matching this schema:' in s and ('"title": "LinkedStatementsResponse"' in s or '"title": "StatementsResponse"' in s) for s in contents):phase='tree_extraction'
        else:phase='draft_feedback_revision_or_other'
        usage=phase_usage[mode,phase];usage['calls']+=1;usage['input_tokens']+=input_tokens or 0;usage['output_tokens']+=output_tokens or 0;usage['usage_estimate_usd']+=cost or 0
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
            concession_edges=concessions[mode],
            rejected_snapshots=sum(i['label'].split('/')[2]==mode for i in invalid),
            selected_reply_paths=sum(b['selected_reply_paths'] for b in selected)))
    report=dict(run_id=run_id,metrics=by_mode,final_binding_checks=bindings,integrity_issues=integrity,
        structured_snapshot_counts=dict(states),invalid_snapshots=invalid,paid_requests=0,
        usage_by_phase=[dict(mode=m,phase=p,**u) for (m,p),u in sorted(phase_usage.items())],
        limitations=['Speaker/source ownership and version consistency do not validate semantic entailment of extracted claims.',
                     'An edge records a response, not whether the issue was resolved.',
                     'Counts include historical setup and all observed chunks; repeated cases are not independent samples.'])
    (base/f'{run_id}_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'metrics':by_mode,'integrity_issues':integrity},indent=2))
    assert not integrity
    assert all(b['selected_versions_valid'] and b['request_versions_valid']
               and b['coverage_material_version_valid'] for b in bindings)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');parser.add_argument('--cases-file',default=str(ROOT/'experiments/incremental_planning/cases_v4.json'))
    args=parser.parse_args();diagnose(args.run_id,args.cases_file)
