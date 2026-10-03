"""Offline diagnostics of frozen tree runs; never invokes a model or edits results."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from debate_tree import DebateTree
from streaming.grounding import parse_state
from streaming.tree_grounding import tree_targets


def diagnose(run_id):
    base = ROOT / 'experiments/incremental_planning'
    run = base / 'run' / run_id
    metadata = json.loads((run / 'metadata_worker0.json').read_text())
    results = [json.loads(p.read_text()) for p in sorted(run.glob('*__*.json'))]
    groups = defaultdict(list)
    binding = []
    for result in results:
        group = 'cross_turn' if result['kind'].startswith('cross_turn') else 'long_chunks'
        groups[group, result['mode']].append(result)
        if result['mode'] not in ('grounded_tree', 'light_tree'):
            continue
        before = result['before_generation']
        ours, theirs = [DebateTree.from_json(before[k]) for k in ('our_tree', 'opponent_tree')]
        active = {n['node_id']: n for n in tree_targets((ours, theirs), theirs.side)}
        claims = before['state'].get('claims', [])
        verified = all(c['node_id'] in active and c['target_version'] == active[c['node_id']]['version'] for c in claims)
        binding.append(dict(case=result['case'], mode=result['mode'], active_targets=len(active),
                            selected_claims=len(claims), serialized_tree_versions_match=verified,
                            final_raw_prefix_fallback=not bool(before['state'])))
    group_metrics = []
    for (group, mode), rs in sorted(groups.items()):
        group_metrics.append(dict(group=group, mode=mode, n=len(rs),
            checklist_rate=statistics.mean(sum(c['passed'] for c in r['judge']['checks'])/len(r['judge']['checks']) for r in rs),
            residual_text_seconds=statistics.mean(r['estimated_residual_text_seconds'] for r in rs),
            generation_calls=statistics.mean(r['model_usage']['calls'] for r in rs)))
    db = sqlite3.connect(f'file:{base / "run/cost.sqlite"}?mode=ro', uri=True)
    snapshots, failures, empty_targets, failure_modes = Counter(), [], [], Counter()
    hashes, last_requests = {}, {}
    for request_id, label in db.execute('select id,label from calls order by id'):
        if not label.startswith(run_id + '/'):
            continue
        path = base / 'run' / f'call_{request_id:06}.json'
        raw = path.read_bytes(); hashes[str(request_id)] = hashlib.sha256(raw).hexdigest()
        artifact = json.loads(raw)
        prompt = '\n'.join(m['content'] for m in artifact['request']['messages'] if isinstance(m.get('content'), str))
        if not prompt.startswith('Prepare a compact JSON snapshot'):
            continue
        payload = json.loads(prompt[prompt.index('{"context":'):])
        mode = label.split('/')[2]; snapshots[mode] += 1
        targets = payload['context'].get('tree_targets')
        last_requests[tuple(label.split('/')[1:3])] = {n['node_id']: n for n in targets or []}
        if targets == []:
            empty_targets.append(dict(id=request_id, label=label))
        response = artifact['response']['choices'][0]['message']['content']
        try:
            parse_state(response, ' '.join(payload['heard_prefix']), tree_targets=targets)
        except ValueError as error:
            failures.append(dict(id=request_id, label=label, reason=str(error),
                                 supplied_tree_targets=None if targets is None else len(targets)))
            failure_modes[mode, str(error)] += 1
    for item in binding:
        targets = last_requests[item['case'], item['mode']]
        result = next(r for r in results if (r['case'], r['mode']) == (item['case'], item['mode']))
        item['versions_match_original_planning_request'] = all(
            c['node_id'] in targets and c['target_version'] == targets[c['node_id']]['version']
            for c in result['before_generation']['state'].get('claims', []))
    report = dict(run_id=run_id, metadata=metadata, group_metrics=group_metrics,
                  final_binding_checks=binding, structured_snapshot_counts=dict(snapshots),
                  structured_failures=failures,
                  failure_counts=[dict(mode=m, reason=r, count=n) for (m,r),n in sorted(failure_modes.items())],
                  empty_tree_target_requests=empty_targets,
                  request_artifact_hashes_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                  paid_requests=0,
                  limitations=['Subgroups contain only four authored cases and are descriptive, not independent validation.',
                               'Source binding checks attribution/version consistency, not entailment or argument quality.',
                               'No answer, judgment or inference configuration is changed by this analysis.'])
    mismatches = sum(not b['serialized_tree_versions_match'] for b in binding)
    if mismatches:
        report['limitations'].append(
            f'{mismatches} serialized pre-generation tree versions differ from the original planning requests. '
            'This run used shallow snapshots whose argument lists could change during later own-speech analysis; '
            'original request-time bindings are checked separately and originals are preserved.')
    (base / f'{run_id}_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('group_metrics','structured_snapshot_counts','failure_counts')},indent=2))
    print('Empty target requests:',len(empty_targets),'Final bound states:',sum(b['selected_claims']>0 for b in binding), '/',len(binding))
    assert all(b['versions_match_original_planning_request'] for b in binding), 'Binding differs from original request'
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_id');diagnose(parser.parse_args().run_id)
