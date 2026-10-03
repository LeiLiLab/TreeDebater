"""Join the frozen tree run and supplemental Linear control without rewriting results."""
from collections import defaultdict
import json
from pathlib import Path

from summarize import paired_comparison

ROOT = Path(__file__).resolve().parent
TREE_RUN = 'tree-grounded-heldout-v1'
LINEAR_RUN = 'tree-linear-control-v1'


def compare():
    summaries = [json.loads((ROOT / f'{run}_summary.json').read_text()) for run in (TREE_RUN, LINEAR_RUN)]
    tree, linear = [s['metadata'] for s in summaries]
    # The documented code difference only deep-copies diagnostic snapshots.
    allowed_differences = {'source_digest', 'modes'}
    differences = {k for k in set(tree) | set(linear) if tree.get(k) != linear.get(k)}
    if differences - allowed_differences:
        raise ValueError('Incompatible evaluation settings: ' + str(differences))
    manifest = json.loads((ROOT / 'manifest_linear_control.json').read_text())
    tree_manifest = json.loads((ROOT / 'manifest_v3.json').read_text())
    if linear['source_digest'] != manifest['source_digest'] or tree['cases_digest'] != manifest['cases_digest']:
        raise ValueError('Frozen source/cases do not match recorded control plan')
    if tree['source_digest'] != tree_manifest['heldout']['source_digest']:
        raise ValueError('Tree reference differs from its frozen manifest')
    if any(s['missing'] or s['completed_answers'] != s['expected_answers'] for s in summaries):
        raise ValueError('Comparison requires complete results')
    by_mode = defaultdict(list)
    for run in (TREE_RUN, LINEAR_RUN):
        for path in sorted((ROOT / 'run' / run).glob('*__*.json')):
            result = json.loads(path.read_text())
            by_mode[result['mode']].append(result)
    if set(by_mode) != {'linear', *tree['modes']}:
        raise ValueError('Unexpected comparison arms')
    for mode, rs in by_mode.items():
        if len(rs) != len(tree['case_ids']) * tree['repeats']:
            raise ValueError('Unbalanced comparison: ' + mode)
        identities = {(r['case'], r['repeat']) for r in rs}
        if identities != {(case, rep) for case in tree['case_ids'] for rep in range(tree['repeats'])}:
            raise ValueError('Mismatched case identities')
    order = ['linear', 'tree_plan', 'grounded_linear', 'grounded_tree', 'light_linear', 'light_tree']
    metrics = {m['mode']: m for s in summaries for m in s['metrics']}
    comparisons = {mode: paired_comparison(by_mode, tree, 'linear', mode) for mode in tree['modes']}
    matched = {mode + '_vs_' + base: paired_comparison(by_mode, tree, base, mode)
               for base, mode in [('grounded_linear', 'grounded_tree'), ('light_linear', 'light_tree')]}
    report = dict(reference_mode='linear', source_runs=[TREE_RUN, LINEAR_RUN], total_answers=sum(len(rs) for rs in by_mode.values()),
        cases=tree['case_ids'], repeats=tree['repeats'], metrics=[metrics[m] for m in order],
        paired_vs_original_linear=comparisons, matched_grounding_controls=matched,
        source_metadata={s['run_id']:s['metadata'] for s in summaries},
        costs_including_judges={s['run_id']:s['usage_including_judging'] for s in summaries},
        limitations=manifest['comparison_limitations'] + [
            'One repetition per authored case; intervals bootstrap cases and are unadjusted for multiple comparisons.',
            'Checklist measures explicit coverage of supplied conditions, not overall debate ability; omissions can fail composite checks.',
            'Strawman and unsupported-fact fields are fallible automatic flags, not independently verified error rates.',
            'Residual latency is text-only, simulated from a fixed input schedule and measured call durations; no audio playback.',
            'Comparisons to original Linear bundle grounding/feedback changes with tree changes; matched grounded/light arms better control these components.',
            'Some tree answers used raw-prefix fallback; score differences do not isolate successful node binding.'])
    (ROOT / 'tree-vs-linear-v1_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print('mode\tchecklist\tstrength\tstrawman\tunsupported\ttext_seconds\tcalls\tgen_cost')
    for m in report['metrics']:
        print(f"{m['mode']}\t{m['checklist_rate']:.3f}\t{m['strength_mean']:.3f}\t{m['strawman_rate']:.3f}\t{m['unsupported_fact_rate']:.3f}\t{m['residual_text_mean_s']:.3f}\t{m['generation_calls_mean']:.3f}\t{m['generation_cost_mean_usd']:.6f}")
    print(json.dumps({k:{a:b for a,b in v.items() if a!='case_differences'} for k,v in comparisons.items()},indent=2))
    return report


if __name__ == '__main__':
    compare()
