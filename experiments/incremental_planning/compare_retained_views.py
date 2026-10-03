"""Compare frozen retained-tree runs, including a matched node-cap ablation."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from summarize import paired_comparison, checklist, interval

ROOT = Path(__file__).resolve().parent
RUNS = ('retained-heldout-v1', 'retained-wide-v1')
ORDER = ('grounded_linear', 'grounded_tree', 'flat_tree', 'branch_tree', 'branch_wide')


def compare():
    summaries = [json.loads((ROOT / f'{run}_summary.json').read_text()) for run in RUNS]
    main, wide = [s['metadata'] for s in summaries]
    different = {k for k in set(main) | set(wide) if main.get(k) != wide.get(k)}
    if different - {'modes', 'workers', 'tree_limits'}:
        raise ValueError('Incompatible run settings: ' + str(different))
    manifest = json.loads((ROOT / 'manifest_v5.json').read_text())
    if main['source_digest'] != manifest['source_digest'] or main['cases_digest'] != manifest['cases_digest']:
        raise ValueError('Source or data differs from the frozen plan')
    if main['tree_limits'] != {'max_tree_targets': 8, 'max_tree_context_nodes': 16}:
        raise ValueError('Unexpected default view')
    if wide['tree_limits'] != {'max_tree_targets': 128, 'max_tree_context_nodes': 256}:
        raise ValueError('Unexpected wide view')
    permitted_missing = manifest.get('missing_judgments', [])
    if summaries[0]['missing'] != permitted_missing or summaries[1]['missing']:
        raise ValueError('Unexpected missing judgments')
    by_mode = defaultdict(list)
    all_results = defaultdict(list)
    metrics = []
    for run, summary in zip(RUNS, summaries):
        for row in summary['metrics']:
            metrics.append(dict(row, mode='branch_wide' if run == RUNS[1] else row['mode']))
        for path in sorted((ROOT / 'run' / run).glob('*__*.json')):
            result = json.loads(path.read_text())
            mode = 'branch_wide' if run == RUNS[1] else result['mode']
            all_results[mode].append(result)
            if 'judge' in result:
                by_mode[mode].append(result)
    expected = {(case, rep) for case in main['case_ids'] for rep in range(main['repeats'])}
    assert set(by_mode) == set(ORDER)
    for mode, results in all_results.items():
        if len(results) != len(expected) or {(r['case'], r['repeat']) for r in results} != expected:
            raise ValueError('Unbalanced or duplicated results in ' + mode)
    # Timing/cost use every generated answer, even if its judge is unavailable.
    for row in metrics:
        rs = all_results[row['mode']]
        row['generated_answers'] = len(rs)
        row['residual_text_mean_s'] = mean(r['estimated_residual_text_seconds'] for r in rs)
        row['residual_worker_return_mean_s'] = mean(r['remaining_preparation_seconds'] +
            r['full_generation_including_own_analysis_seconds'] for r in rs)
        row['generation_cost_mean_usd'] = mean(r['model_usage']['reported_usage_estimate_usd'] for r in rs)
        row['generation_calls_mean'] = mean(r['model_usage']['calls'] for r in rs)
        row['prior_context_setup_calls_mean'] = mean(r['prior_context_setup_calls'] for r in rs)
        row['live_generation_calls_mean'] = mean(r['live_generation_calls'] for r in rs)
        row['answer_words_mean'] = mean(r['answer_words'] for r in rs)
    comparisons = {f'{candidate}_vs_{base}': paired_comparison(by_mode, main, base, candidate)
                   for base, candidate in [('grounded_linear', 'grounded_tree'),
                                           ('grounded_linear', 'branch_tree'),
                                           ('flat_tree', 'branch_tree'),
                                           ('branch_wide', 'branch_tree')]}
    for comparison in comparisons.values():
        ids = {d['case'] for d in comparison['case_differences']}
        comparison['paired_baseline_quality'] = mean(checklist(r) for r in by_mode[comparison['baseline']] if r['case'] in ids)
        comparison['paired_candidate_quality'] = mean(checklist(r) for r in by_mode[comparison['candidate']] if r['case'] in ids)
    sensitivity = []
    # These are bounds, not substitute judgments; never write imputed scores to results.
    for assumed in (0.0, 1.0):
        per_case = {}
        for case in main['case_ids']:
            scores = [checklist(r) for r in by_mode['branch_tree'] if r['case']==case]
            if len(scores) < main['repeats']:
                if [case, 'branch_tree', 1] not in permitted_missing or len(scores) != 1:
                    raise ValueError('Unexpected missing-data pattern')
                scores.append(assumed)
            per_case[case] = mean(scores)
        comparisons_at_bound = {}
        for base in ('grounded_linear', 'flat_tree', 'branch_wide'):
            differences = [per_case[case] - mean(checklist(r) for r in by_mode[base] if r['case']==case)
                           for case in main['case_ids']]
            comparisons_at_bound[base] = dict(quality_diff_mean=mean(differences),
                                               quality_diff_95ci=interval(differences))
        sensitivity.append(dict(assumed_missing_checklist_rate=assumed, case_count=len(per_case),
                                branch_quality_mean=mean(per_case.values()), comparisons=comparisons_at_bound))
    groups = []
    for group in ('dense_current_branches', 'position_changes'):
        for mode in ORDER:
            rs = [r for r in by_mode[mode]
                  if (r['kind'] == 'dense_current_branches') == (group == 'dense_current_branches')]
            groups.append(dict(group=group, mode=mode, cases=len({r['case'] for r in rs}), answers=len(rs),
                checklist_rate=mean(checklist(r) for r in rs),
                residual_text_mean_s=mean(r['estimated_residual_text_seconds'] for r in rs),
                generation_cost_mean_usd=mean(r['model_usage']['reported_usage_estimate_usd'] for r in rs)))
    diagnostics = [json.loads((ROOT / f'{run}_diagnostics.json').read_text()) for run in RUNS]
    selection = []
    for run, diagnostic in zip(RUNS, diagnostics):
        for mode in sorted({b['mode'] for b in diagnostic['final_binding_checks']}):
            bs = [b for b in diagnostic['final_binding_checks'] if b['mode'] == mode]
            selection.append(dict(mode='branch_wide' if run == RUNS[1] else mode, answers=len(bs),
                target_cap_binding_answers=sum(b['target_cap_binding'] for b in bs),
                stored_nodes_mean=mean(b['stored_nodes'] for b in bs),
                historical_or_dependent_nodes_mean=mean(b['historical_or_dependent_nodes'] for b in bs),
                eligible_opponent_nodes_mean=mean(b['eligible_opponent_nodes'] for b in bs),
                selected_target_nodes_mean=mean(b['active_targets'] for b in bs),
                omitted_eligible_targets_mean=mean(b['omitted_eligible_targets'] for b in bs),
                additional_context_nodes_mean=mean(b['additional_context_nodes'] for b in bs),
                omitted_nodes_from_full_current_view_mean=mean(b['omitted_nodes_from_full_current_view'] for b in bs),
                context_cap_saturated_answers=sum(b['context_cap_saturated'] for b in bs),
                final_raw_fallbacks=sum(b['final_raw_prefix_fallback'] for b in bs),
                with_bound_claims=sum(b['selected_claims'] > 0 for b in bs)))
    routes = []
    for mode in ORDER[1:]:
        for fallback in (False, True):
            generated = [r for r in all_results[mode] if (not bool(r['before_generation']['state'])) == fallback]
            judged = [r for r in generated if 'judge' in r]
            if generated:
                routes.append(dict(mode=mode, raw_prefix_fallback=fallback, generated=len(generated),
                                   judged=len(judged), checklist_rate=mean(checklist(r) for r in judged) if judged else None))
    case_scores = []
    for case in main['case_ids']:
        case_scores.append(dict(case=case, rates={m: mean(checklist(r) for r in by_mode[m] if r['case']==case)
                                                for m in ORDER},
                                judged_counts={m:sum(r['case']==case for r in by_mode[m]) for m in ORDER}))
    report = dict(source_runs=list(RUNS), source_metadata={s['run_id']: s['metadata'] for s in summaries},
        total_answers=sum(len(v) for v in all_results.values()),
        judged_answers=sum(len(v) for v in by_mode.values()), missing_judgments=permitted_missing,
        independent_cases=len(main['case_ids']),
        metrics=sorted(metrics, key=lambda r: ORDER.index(r['mode'])), paired_comparisons=comparisons,
        descriptive_subgroups=groups, selection_audit=selection, case_scores=case_scores,
        missing_judgment_sensitivity=sensitivity, delivery_route_descriptives=routes,
        costs_including_judges={s['run_id']: s['usage_including_judging'] for s in summaries},
        limitations=manifest['limitations']+[
            'One Branch courtyard judgment is unavailable after two HTTP 503 responses. Paired comparisons involving Branch exclude the whole incomplete case; descriptive quality has 15 versus 16 judgments. Timing/cost use all generated answers.',
            'Intervals resample cases after averaging repeats and are unadjusted for multiple comparisons.',
            'Composite checklist items assess explicit condition coverage; they are not an overall debate skill score.',
            'Automated strawman and unsupported-fact flags are not human-adjudicated factual error rates.',
            'All-current wide view still selects at most three claims in the planner output, with the same 700-token preparation limit.',
            'Cap groups use separately extracted graphs and generation samples, so this is a pipeline comparison rather than a shared-graph intervention.',
            'Fallback answers use the heard prefix; quality differences do not isolate successful graph binding. Delivery-route subgroups are descriptive and contain different cases.',
            'The target cap binds in only 4/16 default Branch endpoints, with just three total omitted current-view nodes across two courtyard answers; this does not establish a clipping mechanism behind score differences.'
        ])
    (ROOT / 'retained-views-v1_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    for row in report['metrics']:
        print(row['mode'], {k: row[k] for k in ('checklist_rate','strength_mean','strawman_rate',
              'unsupported_fact_rate','residual_text_mean_s','generation_cost_mean_usd')})
    print(json.dumps({k: {a:b for a,b in v.items() if a!='case_differences'}
                      for k,v in comparisons.items()}, indent=2))
    return report


if __name__ == '__main__':
    compare()
