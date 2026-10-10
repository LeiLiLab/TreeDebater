import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('retrieval_eval_v2', ROOT/'experiments/retrieval_eval_v2/run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_bundles_split_and_strength_annotations_removed():
    matches = [['source', 'attack', 'query', 'parent', 0.7,
                'First response. (Strength: 1.2)\n\tSecond response. (Strength: -0.1)\n\t']]
    assert module.atomic_materials(matches) == ['First response.', 'Second response.']
    assert module.atomic_materials(matches*2) == ['First response.', 'Second response.']


def test_new_material_keeps_explanation():
    match = ['source', 'attack', 'query', 'parent', 0.7,
             'Response claim.\nIts explanation. (Strength: 1.0)\n\t']
    assert module.atomic_materials([match]) == ['Response claim.\nIts explanation.']


def test_approval_precedes_any_client_or_model_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'HERE', tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps({'approved': False}))
    monkeypatch.setattr(module, 'load_module', lambda *a: pytest.fail('Client loaded before approval'))
    with pytest.raises(RuntimeError, match='explicit approval'):
        module.main()


def test_summary_scores_atomic_materials_and_query_coverage():
    rows = [dict(arms={'old':['bad'], 'new':['good','bad']},blind_materials=['bad','good'],
                 grades=[{'id':0,'valid':False},{'id':1,'valid':True}],v2_seconds=1.5)]
    result=module.summarize(rows)
    assert result['old']['valid_queries']==0
    assert result['new']['valid_queries']==1
    assert result['new']['valid_material_rate']==0.5


def test_context_recovery_never_reads_a_future_action(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    (tmp_path/'example.log').write_text(
        'title=Debate-Flow-Tree-Action call_id=1\nquery\n'
        'title=Debate-Flow-Tree-Action call_id=2\n')
    blocks=[]
    for call,argument in [(1,'available at query time'),(2,'future argument')]:
        actions=[dict(action='attack',target_claim='target',target_argument=argument)]
        blocks.append(f'call_id={call} phase=helper title=Debate-Flow-Tree-Action\n---\n'+json.dumps(actions))
    (tmp_path/'example_io.log').write_text('\n'.join(blocks))
    case=dict(source='example.log:2',action='attack',target='target')
    assert module.recover_context(case)['target_argument']=='available at query time'


def test_failure_handling_does_not_swallow_budget_stops():
    assert module.model_response_failure(RuntimeError('Truncated completion; no automatic retry'))
    assert not module.model_response_failure(RuntimeError('Budget stop: no dispatch'))
    assert not module.model_response_failure(TimeoutError('Unknown remote status'))


def test_failed_grading_is_unscored_not_incorrect():
    rows = [dict(arms={'new':['ungraded']},blind_materials=['ungraded'],grades=[],
                 v2_seconds=1.0,generation_failure=None)]
    summary = module.summarize(rows)
    assert summary['new']['ungraded_material_count'] == 1
    assert summary['new']['valid_material_rate'] is None
    assert summary['new']['quality_complete_queries'] == 0


def test_generation_failure_is_counted_separately_from_latency():
    rows = [dict(arms={'new':[]},blind_materials=[],grades=[],v2_seconds=0.0,
                 generation_failure='Truncated completion')]
    summary = module.summarize(rows)
    assert summary['v2_latency']['generation_failed_queries'] == 1
    assert summary['v2_latency']['mean_seconds'] is None
