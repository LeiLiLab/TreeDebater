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
    flat=by_mode['flat_tree'];binding_groups=[]
    for valid in (True,False):
        subset=[r for r in flat if bool(r['before_generation']['state'])==valid and 'judge' in r]
        binding_groups.append(dict(valid_final_plan=valid,n=len(subset),
            checklist_rate=mean(map(checklist,subset)) if subset else None,
            interpretation='Post-generation descriptive subset; selection prevents a causal comparison.'))
    report=dict(run_id=run_id,metadata=metadata,primary_comparisons=['flat_tree_vs_linear','flat_tree_vs_legacy'],
        metrics=metrics,paired_comparisons=pairs,descriptive_subgroups=groups,flat_binding_subgroups=binding_groups,
        missing_judgments=summary['missing'],usage_including_judging=summary['usage_including_judging'],
        completion_audit=summary['completion_audit'],paid_requests=0,limitations=manifest['limitations']+[
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
