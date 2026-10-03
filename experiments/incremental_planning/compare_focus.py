"""Compare three frozen pipelines on fresh cases, preserving every raw verdict."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from compare_conditions import METRICS, paired
from summarize import checklist

ROOT = Path(__file__).resolve().parent
PAIRS = [('linear', 'flat_tree'), ('legacy', 'flat_tree'), ('legacy', 'linear')]


def compare(run_id='flat-linear-legacy-v1', manifest_name='manifest_focus_v1.json'):
    manifest = json.loads((ROOT/manifest_name).read_text())
    summary = json.loads((ROOT/f'{run_id}_summary.json').read_text())
    metadata = summary['metadata']
    if manifest['run_id'] != run_id or metadata['source_digest'] != manifest['source_digest']:
        raise ValueError('Run/source differs from freeze')
    case_path = ROOT.parents[1]/manifest['cases_file']
    if hashlib.sha256(case_path.read_bytes()).hexdigest() != metadata['cases_digest'] or metadata['cases_digest'] != manifest['cases_digest']:
        raise ValueError('Cases differ from freeze')
    cases = {c['id']: c for c in json.loads(case_path.read_text())}
    rows = [json.loads(p.read_text()) for p in sorted((ROOT/'run'/run_id).glob('*__*.json'))]
    expected = {(c, m, i) for c in metadata['case_ids'] for m in metadata['modes'] for i in range(metadata['repeats'])}
    identities = [(r['case'],r['mode'],r['repeat']) for r in rows]
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError('Duplicate/missing/unexpected generated answer')
    if set(metadata['modes']) != {'flat_tree','linear','legacy'}:
        raise ValueError('Unexpected comparison arms')
    by_mode = {m:[r for r in rows if r['mode']==m] for m in metadata['modes']}
    pairs = {f'{candidate}_vs_{baseline}': {
        metric: paired(by_mode[baseline],by_mode[candidate],metadata,fn) for metric,fn in METRICS.items()
    } for baseline,candidate in PAIRS}
    bounds={}
    for mode,rs in by_mode.items():
        observed=sum(checklist(r) for r in rs if 'judge' in r)
        missing=sum('judge' not in r for r in rs)
        bounds[mode]=dict(missing=missing,mean_bounds=[observed/len(rs),(observed+missing)/len(rs)])
    missing_sensitivity=dict(mode_bounds=bounds,pair_difference_bounds={
        f'{candidate}_vs_{baseline}': [bounds[candidate]['mean_bounds'][0]-bounds[baseline]['mean_bounds'][1],
                                      bounds[candidate]['mean_bounds'][1]-bounds[baseline]['mean_bounds'][0]]
        for baseline,candidate in PAIRS},
        interpretation='Unknown scores range from zero to one; logical all-generated bounds, not imputed verdicts or confidence intervals.')
    metrics=[];groups=[]
    for mode,rs in by_mode.items():
        judged=[r for r in rs if 'judge' in r]
        metrics.append(dict(mode=mode,generated=len(rs),judged=len(judged),
            **{k:mean(fn(r) for r in (rs if k in ('residual_text_seconds','generation_cost_usd') else judged))
               for k,fn in METRICS.items()},
            generation_calls_mean=mean(r['model_usage']['calls'] for r in rs),
            prior_setup_calls_mean=mean(r['prior_context_setup_calls'] for r in rs),
            live_calls_mean=mean(r['live_generation_calls'] for r in rs),
            input_tokens_mean=mean(r['model_usage']['input_tokens'] for r in rs),
            output_tokens_mean=mean(r['model_usage']['output_tokens'] for r in rs),
            answer_words_mean=mean(r['answer_words'] for r in rs),
            worker_return_seconds_mean=mean(r['remaining_preparation_seconds']+r['full_generation_including_own_analysis_seconds'] for r in rs),
            invalid_plan_events=sum(e['action']=='INVALID_STATE' for r in rs for e in r['after_generation']['events']),
            final_raw_fallbacks=sum(not bool(r['before_generation']['state']) for r in rs) if mode=='flat_tree' else None))
        for grouping in ('kind','side'):
            for value in sorted({cases[r['case']][grouping] for r in rs}):
                subset=[r for r in judged if cases[r['case']][grouping]==value]
                groups.append(dict(mode=mode,grouping=grouping,value=value,n=len(subset),
                    independent_cases=len({r['case'] for r in subset}),
                    checklist_rate=mean(map(checklist,subset)) if subset else None,
                    strength_mean=mean(r['judge']['rebuttal_strength'] for r in subset) if subset else None,
                    unsupported_fact_rate=mean(int(r['judge']['unsupported_facts']) for r in subset) if subset else None))
    common_cases=[case for case in metadata['case_ids'] if all(
        sum(r['case']==case and 'judge' in r for r in rs)==metadata['repeats'] for rs in by_mode.values())]
    common_metrics=[]
    for mode,rs in by_mode.items():
        subset=[r for r in rs if r['case'] in common_cases]
        common_metrics.append(dict(mode=mode,case_count=len(common_cases),answers=len(subset),
            **{k:mean(fn(r) for r in subset) if subset else None for k,fn in METRICS.items()}))
    flat=by_mode['flat_tree'];binding_groups=[]
    for valid in (True,False):
        subset=[r for r in flat if bool(r['before_generation']['state'])==valid and 'judge' in r]
        binding_groups.append(dict(valid_final_plan=valid,n=len(subset),
            checklist_rate=mean(map(checklist,subset)) if subset else None,
            interpretation='Post-generation descriptive subset; selection prevents a causal comparison.'))
    report=dict(run_id=run_id,metadata=metadata,primary_comparisons=['flat_tree_vs_linear','flat_tree_vs_legacy'],
        metrics=metrics,paired_comparisons=pairs,descriptive_subgroups=groups,flat_binding_subgroups=binding_groups,
        common_complete_cases=common_cases,common_case_metrics=common_metrics,
        missing_judgments=summary['missing'],missing_sensitivity=missing_sensitivity,usage_including_judging=summary['usage_including_judging'],
        completion_audit={**summary['completion_audit'],
            'finished_with_missing_judgment_markers':len(list((ROOT/'run'/run_id).glob('finished_worker*.json'))),
            'generated_answers':len(rows),'judged_answers':sum('judge' in r for r in rows)},paid_requests=0,limitations=manifest['limitations']+[
            'Pairs average repeats within a case and exclude an entire case with any missing judgment; descriptive timing/cost uses all generated answers.',
            'Error flags are automated judgments with known false positives/negatives; zero flags does not establish zero errors.',
            'Subgroups and Flat valid-plan/fallback splits are descriptive, small and selected; no causal isolation.',
            'Positive error/latency/cost differences are worse; bootstrap intervals are unadjusted for multiple comparisons.'])
    (ROOT/f'{run_id}_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(metrics,indent=2))
    for name,result in pairs.items():
        print(name,json.dumps({metric:{k:v for k,v in value.items() if k!='cases'} for metric,value in result.items()}))
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run_id',nargs='?',default='flat-linear-legacy-v1')
    ap.add_argument('--manifest-name',default='manifest_focus_v1.json')
    compare(**vars(ap.parse_args()))
