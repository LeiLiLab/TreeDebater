"""V2 replay: reused vectors/old outputs, calibrated fresh blind grading, guarded costs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = HERE.parent/'retrieval_eval_v1'
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(HERE))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_materials(matches):
    """Unpack V1's sibling bundles without altering the substantive material text."""
    result = []
    for match in matches:
        for piece in match[5].split('\n\t'):
            text = re.sub(r'\s*\(Strength: [-+\d.]+\)\s*$', '', piece).strip()
            if text and text not in result:
                result.append(text)
    return result


def recover_context(case):
    main, line = case['source'].rsplit(':', 1)
    previous = (ROOT/main).read_text().splitlines()[:int(line)-1]
    ref = next((m[1] for text in reversed(previous)
                if (m := re.search(r'title=Debate-Flow-Tree-Action call_id=(\d+)', text))), None)
    io = main.replace('.log', '_io.log')
    body = (ROOT/io).read_text()
    header = re.search(r'call_id='+str(ref)+r' phase=helper title=Debate-Flow-Tree-Action\n-+\n', body)
    actions = json.JSONDecoder().raw_decode(body[header.end():])[0] if header else []
    matches = [a for a in actions if a['action'] == case['action'] and a['target_claim'] == case['target']]
    argument = matches[0].get('target_argument', '') if len(matches) == 1 else ''
    return dict(case, target_argument=argument,
                argument_source=f'{io}#call_id={ref}' if argument else None)


def cached_vectors(texts):
    vectors = {}
    for offset in range(0, len(texts), 128):
        batch = texts[offset:offset+128]
        body = {'model': 'text-embedding-3-small', 'input': batch}
        key = hashlib.sha256(json.dumps(body, ensure_ascii=False).encode()).hexdigest()
        response = json.loads((V1/'responses'/f'{key}.json').read_text())
        entries = sorted(response['data'], key=lambda d: d['index'])
        if [d['index'] for d in entries] != list(range(len(batch))):
            raise ValueError('Incomplete cached embedding batch; no paid fallback')
        vectors.update(zip(batch, [d['embedding'] for d in entries]))
    return vectors


def summarize(rows):
    result = {}
    for arm in rows[0]['arms']:
        hit = valid_queries = count = valid_count = judged_count = complete_queries = 0
        for row in rows:
            grades = {g['id']: g for g in row['grades']}
            ids = {m: i for i, m in enumerate(row['blind_materials'])}
            texts = row['arms'][arm]
            flags = [grades[ids[text]]['valid'] for text in texts if ids[text] in grades]
            hit += bool(texts)
            valid_queries += any(flags)
            count += len(texts)
            judged_count += len(flags)
            valid_count += sum(flags)
            complete_queries += len(flags) == len(texts)
        result[arm] = dict(queries=len(rows), hit_queries=hit, valid_queries=valid_queries,
                           material_count=count, judged_material_count=judged_count,
                           ungraded_material_count=count-judged_count, quality_complete_queries=complete_queries,
                           valid_material_count=valid_count,
                           valid_material_rate=valid_count/judged_count if judged_count else None)
    times = [r['v2_seconds'] for r in rows if not r.get('generation_failure')]
    result['v2_latency'] = {'mean_seconds': sum(times)/len(times) if times else None,
                           'successful_queries': len(times),
                           'generation_failed_queries': sum(bool(r.get('generation_failure')) for r in rows),
                           'excludes_shared_embedding_generation': True}
    return result


def model_response_failure(exc):
    from pydantic import ValidationError
    from judging import InvalidJudgement
    return (isinstance(exc, (json.JSONDecodeError, ValidationError, InvalidJudgement))
            or isinstance(exc, RuntimeError) and str(exc).startswith('Truncated completion;'))


def main():
    manifest = json.loads((HERE/'manifest.json').read_text())
    if not manifest.get('approved') or not manifest.get('approval'):
        raise RuntimeError('V2 experiment and cumulative USD5 cap require explicit approval')
    for path, digest in manifest['hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Source/input changed: '+path)
    # Reuse the previously tested serial guard, with a separate ledger and remaining cap.
    client = load_module('v2_budget_client', V1/'run.py')
    client.HERE = HERE
    guard = client.Guard(manifest)
    def costs():
        current = guard.summary()
        return dict(current, cumulative_cap_usd=manifest['cumulative_cap_usd'],
                    prior_estimated_usd=manifest['prior_estimated_usd'],
                    cumulative_estimated_usd=manifest['prior_estimated_usd']+current['estimated_peak_usd'],
                    cumulative_exposure_usd=manifest['prior_exposure_usd']+current['exposure_usd'])
    try:
        from judging import CalibratedJudge
        judge = CalibratedJudge(lambda p: guard.chat(manifest['judge_model'], p))
        try:
            checks = judge.calibrate()
        finally:
            (HERE/'calibration.json').write_text(json.dumps(judge.calibration, indent=2))
        print('Calibration passed:', len(checks), flush=True)

        from debate_tree import PrepareTree
        from utils.rehearsal_retrieval import retrieve, relation_prompt
        from utils.llm_schemas import RehearsalRelationResponse
        cases = json.loads((HERE/'cases.json').read_text())
        previous = json.loads((V1/'results.json').read_text())
        pools = {}
        for slug in {c['slug'] for c in cases}:
            for side in ['for','against']:
                data = json.loads((ROOT/f'results/deepseek-chat/{slug}_pool_{side}.json').read_text())
                pools[slug, side] = [PrepareTree.from_json(item[0]['tree_structure']) for item in data]
        texts = sorted({n.claim for trees in pools.values() for tree in trees for n in tree.get_all_nodes()}
                       | {c['target'] for c in cases})
        vectors = cached_vectors(texts)
        results = []
        for index, case in enumerate(cases):
            saved = previous[index]
            assert all(case[k] == v for k, v in saved['case'].items())
            side = case['side']; opposite = 'against' if side == 'for' else 'for'
            own, other = pools[case['slug'], side], pools[case['slug'], opposite]
            motion = own[0].motion
            depth = {'opening_for':3,'opening_against':2,'rebuttal_for':1,'rebuttal_against':0,
                     'closing_for':0,'closing_against':0}[case['stage']+'_'+side]
            arms = {a: atomic_materials(saved['arms'][a]) for a in manifest['archived_arms']}
            decisions = []
            def validate(candidates):
                result = guard.chat(manifest['relation_model'], relation_prompt(
                    motion, case['action'], case['target'], case['target_argument'], candidates,
                    context={'argument':case['target_argument'], 'ancestors':[], 'constraints':[]}))
                parsed = RehearsalRelationResponse.model_validate(result).model_dump()['decisions']
                decisions.extend(parsed)
                return parsed
            start = time.perf_counter()
            generation_failure = None
            try:
                _, matches = retrieve(case['action'], case['target'], side, opposite, own, other, depth,
                                      vectors[case['target']], embed=lambda cs:[vectors[c] for c in cs],
                                      validate=validate, candidate_k=12, max_results=3,
                                      target_argument=case['target_argument'])
            except Exception as exc:
                if not model_response_failure(exc):
                    raise
                generation_failure = type(exc).__name__ + ': ' + str(exc)
                matches = []
            duration = time.perf_counter()-start
            arms['two_stage_v2'] = atomic_materials(matches)
            materials = list(dict.fromkeys(text for texts in arms.values() for text in texts))
            random.Random(1729+index).shuffle(materials)
            grades, judge_failures = [], []
            for offset in range(0,len(materials),6):
                chunk = materials[offset:offset+6]
                try:
                    local = judge.grade(motion,case['action'],case['target'],case['target_argument'],chunk)
                except Exception as exc:
                    if not model_response_failure(exc):
                        raise
                    judge_failures.append({'ids': list(range(offset, offset+len(chunk))),
                                           'error': type(exc).__name__ + ': ' + str(exc)})
                    continue
                grades.extend(dict(g,id=g['id']+offset) for g in local)
            results.append(dict(case=case,arms=arms,blind_materials=materials,grades=grades,
                                v2_seconds=duration,relation_decisions=decisions,
                                generation_failure=generation_failure,judge_failures=judge_failures))
            (HERE/'results.json').write_text(json.dumps(results,indent=2))
            (HERE/'cost_summary.json').write_text(json.dumps(costs(),indent=2))
            print(index+1,'/',len(cases),costs(),flush=True)
        (HERE/'summary.json').write_text(json.dumps(summarize(results),indent=2))
        manifest['status'] = 'completed'
    except BaseException as exc:
        manifest['status'] = 'stopped'
        manifest['stop_reason'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        (HERE/'cost_summary.json').write_text(json.dumps(costs(),indent=2))
        (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
