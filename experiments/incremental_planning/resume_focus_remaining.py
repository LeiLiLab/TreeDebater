"""Finish a frozen focus worker while preserving exhausted missing judgments.

No new settings, retries or invented verdicts. Uses the original frozen benchmark
functions, source/data hashes, job partition and shared budget. The excluded answer
remains present without a judge field and gets a separate unavailable marker.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from scripts import benchmark_incremental_planning as benchmark
from streaming.experiment_client import BudgetedClient

RUN_ID = 'flat-linear-legacy-v1'


def main(worker_index):
    base = ROOT / 'experiments/incremental_planning'
    run = base / 'run' / RUN_ID
    metadata = json.loads((run / f'metadata_worker{worker_index}.json').read_text())
    manifest = json.loads((base / 'manifest_focus_v1.json').read_text())
    case_path = ROOT / manifest['cases_file']
    if benchmark.code_digest() != metadata['source_digest']:
        raise ValueError('Frozen source changed')
    if hashlib.sha256(case_path.read_bytes()).hexdigest() != metadata['cases_digest']:
        raise ValueError('Frozen cases changed')
    if not 0 <= worker_index < metadata['workers']:
        raise ValueError('Invalid worker index')
    excluded = {r['label'] for r in manifest['judge_retries']
                if r['status'] == 'retry exhausted; judgment unavailable'}
    if not excluded:
        raise ValueError('No exhausted judgment to skip')
    client = BudgetedClient(base / 'run', label=RUN_ID, cap=metadata['cap_usd'])
    os.environ['DEBATE_LLM_API_BASE'] = client.base_url
    os.environ['DEBATE_LOG_PROMPTS'] = '0'
    from debate_tree import Tree
    from utils.tool import logger
    import litellm
    logger.setLevel(logging.WARNING)
    def unmetered_call(*args, **kwargs):
        raise RuntimeError('Unmetered LiteLLM call blocked by replay harness')
    litellm.completion = unmetered_call
    Tree.get_most_similar_node = lambda *args, **kwargs: (None, 0.0)
    cases = [c for c in json.loads(case_path.read_text()) if c['id'] in metadata['case_ids']]
    jobs = [(case, mode, rep) for rep in range(metadata['repeats'])
            for case in cases for mode in metadata['modes']]
    random.Random(20261002).shuffle(jobs)
    for index, (case, mode, rep) in enumerate(jobs):
        if index % metadata['workers'] != worker_index:
            continue
        client.label = f"{RUN_ID}/{case['id']}/{mode}/{rep}"
        path = run / f"{case['id']}__{mode}__{rep}.json"
        if client.label in excluded:
            if not path.exists() or 'judge' in json.loads(path.read_text()):
                raise ValueError('Excluded answer state changed')
            print('UNAVAILABLE_JUDGMENT', client.label, flush=True)
            continue
        if path.exists():
            result = json.loads(path.read_text())
        else:
            print('START', client.label, flush=True)
            result = benchmark.run_case(case, mode, rep, client, metadata['tree_limits'])
            benchmark.atomic_json(path, result)
        if 'judge' not in result:
            result['judge'] = benchmark.judge(case, result['answer'], client,
                metadata['judge_model'], max_tokens=metadata['judge_max_tokens'])
            benchmark.atomic_json(path, result)
        print('DONE', client.label, flush=True)
    benchmark.atomic_json(run / f'finished_worker{worker_index}.json', dict(
        status='remaining jobs finished; exhausted judgments retained', excluded_judgments=sorted(excluded),
        source_digest=benchmark.code_digest(), cost=client.summary()))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument("--worker-index",type=int,required=True)
    main(parser.parse_args().worker_index)
