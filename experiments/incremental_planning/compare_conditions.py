"""Compare frozen before/after pipelines on the same cases; no model calls."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from summarize import checklist, interval

ROOT = Path(__file__).resolve().parent
BASELINE, CANDIDATE = 'retained-heldout-v1', 'conditions-regression-v1'
METRICS = {
    'checklist_rate': checklist,
    'rebuttal_strength': lambda r: r['judge']['rebuttal_strength'],
    'strawman_rate': lambda r: int(r['judge']['strawman']),
    'unsupported_fact_rate': lambda r: int(r['judge']['unsupported_facts']),
    'residual_text_seconds': lambda r: r['estimated_residual_text_seconds'],
    'generation_cost_usd': lambda r: r['model_usage']['reported_usage_estimate_usd'],
}


def load(run):
    summaries = json.loads((ROOT / f'{run}_summary.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((ROOT/'run'/run).glob('*__*.json'))]
    keys = [(r['case'], r['mode'], r['repeat']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate result identities')
    meta = summaries['metadata']
    expected = {(c,m,r) for c in meta['case_ids'] for m in meta['modes'] for r in range(meta['repeats'])}
    if set(keys) != expected:
        raise ValueError('Missing/unexpected generated answers')
    return summaries, rows


def paired(old, new, metadata, metric):
    cases = []
    for case in metadata['case_ids']:
        a = [r for r in old if r['case'] == case and 'judge' in r]
        b = [r for r in new if r['case'] == case and 'judge' in r]
        if len(a) != metadata['repeats'] or len(b) != metadata['repeats']:
            continue
        av, bv = mean(map(metric,a)), mean(map(metric,b))
        cases.append({'case':case, 'before':av, 'after':bv, 'difference':bv-av})
    return {'case_count':len(cases), 'cases':cases,
            'before_mean':mean(c['before'] for c in cases) if cases else None,
            'after_mean':mean(c['after'] for c in cases) if cases else None,
            'difference':mean(c['difference'] for c in cases) if cases else None,
            'difference_95ci':interval([c['difference'] for c in cases]) if cases else None}


def compare(baseline=BASELINE, candidate=CANDIDATE, baseline_manifest="manifest_v5.json", candidate_manifest="manifest_v6.json"):
    before, old = load(baseline)
    after, new = load(candidate)
    bm, am = before['metadata'], after['metadata']
    diffs = {k for k in set(bm)|set(am) if bm.get(k) != am.get(k)}
    if diffs != {'source_digest'}:
        raise ValueError('Unexpected settings differences: '+str(diffs))
    frozen = json.loads((ROOT/candidate_manifest).read_text())
    original = json.loads((ROOT/baseline_manifest).read_text())
    if am['source_digest'] != frozen['source_digest'] or bm['source_digest'] != original['source_digest']:
        raise ValueError('Source digest mismatch')
    if am['cases_digest'] != frozen['cases_digest']:
        raise ValueError('Cases changed')
    if before['missing'] != original['missing_judgments']:
        raise ValueError('Baseline missing judgments changed')
    comparisons = {}
    descriptive = []
    subgroups = []
    missing_sensitivity = []
    for mode in am['modes']:
        a, b = [r for r in old if r['mode']==mode], [r for r in new if r['mode']==mode]
        comparisons[mode] = {name:paired(a,b,am,metric) for name,metric in METRICS.items()}
        missing_a = sum('judge' not in r for r in a)
        missing_b = sum('judge' not in r for r in b)
        if missing_a or missing_b:
            bounds = []
            for group, missing in ((a,missing_a),(b,missing_b)):
                observed = sum(checklist(r) for r in group if 'judge' in r)
                bounds.append([observed/len(group),(observed+missing)/len(group)])
            missing_sensitivity.append(dict(mode=mode,missing_before=missing_a,missing_after=missing_b,
                before_mean_bounds=bounds[0],after_mean_bounds=bounds[1],
                difference_bounds=[bounds[1][0]-bounds[0][1],bounds[1][1]-bounds[0][0]],
                interpretation='Unobserved checklist score ranges from 0 to 1; logical bounds, not confidence intervals or imputed verdicts.'))
        for name, rows in (('before',a), ('after',b)):
            judged = [r for r in rows if 'judge' in r]
            descriptive.append(dict(mode=mode, version=name, generated=len(rows), judged=len(judged),
                **{k:mean(f(r) for r in (rows if k in ('residual_text_seconds','generation_cost_usd') else judged))
                   for k,f in METRICS.items()},
                generation_calls=mean(r['model_usage']['calls'] for r in rows),
                answer_words=mean(r['answer_words'] for r in rows)))
        for kind in sorted({r['kind'] for r in a+b}):
            subgroups.append(dict(mode=mode,kind=kind,comparison=paired(
                [r for r in a if r['kind']==kind], [r for r in b if r['kind']==kind], am,checklist)))
    report = dict(baseline_run=baseline,candidate_run=candidate,metadata_before=bm,metadata_after=am,
        primary_comparison='branch_tree/checklist_rate',paired_before_after=comparisons,
        descriptive=descriptive,subgroups=subgroups,missing_sensitivity=missing_sensitivity,missing_before=before['missing'],missing_after=after['missing'],
        within_new=after['component_comparisons'],cost_new=after['usage_including_judging'],
        limitations=frozen['limitations']+[
            'All differences are after minus before; positive error-rate/latency/cost differences are worse.',
            'Paired estimates drop a whole case if either version lacks a repeat judgment; descriptive rows use all judged answers.',
            'Case-bootstrap intervals average repeats first and are not corrected for multiple comparisons.',
            'Generation and judging are fresh samples; no direct causal isolation of extraction versus review.'])
    output=ROOT/f'{candidate}_comparison.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    for mode,result in comparisons.items():
        print(mode,{k:{x:y for x,y in v.items() if x!='cases'} for k,v in result.items()})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline', default=BASELINE)
    parser.add_argument('--candidate', default=CANDIDATE)
    parser.add_argument('--baseline-manifest', default='manifest_v5.json')
    parser.add_argument('--candidate-manifest', default='manifest_v6.json')
    compare(**vars(parser.parse_args()))
