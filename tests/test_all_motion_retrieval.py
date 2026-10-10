"""Source alignment and parsing checks for all-motion replay preparation."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/retrieval_all_motions_v1'
FIXTURES=Path(__file__).parent/'fixtures/experiments'
spec=importlib.util.spec_from_file_location('allmotion_data',HERE/'data.py')
data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)


def test_complete_motion_inventory_and_query_origins():
    inventory=json.loads((FIXTURES/'retrieval_all_motions_v1/inventory.json').read_text())['motions']
    assert len(inventory)==len({m['slug'] for m in inventory})==8
    cases=json.loads((FIXTURES/'retrieval_all_motions_v1/historical_cases.json').read_text())
    assert len(cases)==75
    assert all(sum(c['slug']==m['slug'] for c in cases)==m['selected_historical_queries'] for m in inventory)
    assert sum(m['generated_queries'] for m in inventory)==45
    fat=next(m for m in inventory if m['slug']=='developed_countries_should_impose_a_fat_tax.')
    assert 'emnlp_res/124.json' in fat['json_sources']
    assert any(x['path']=='emnlp_res/126.log' for x in fat['log_sources'])


def test_tree_reconstruction_preserves_side_parent_and_text():
    body='''Level-0 Root Claim: {"claim":"Root","argument":"A"}, Scores: Support Score: 1.2
    Level-1 Opponent's Attack: {"claim":"Counter","argument":["B"]}, Scores: Attack Score: 1.1
        Level-2 Your Rebuttal: {"claim":"Reply","argument":["C"]}, Scores: Support Score: 0.9'''
    trees,count=data.recover_trees(body,'Motion','for')
    a=trees[0]['structure'];b=a['children'][0];c=b['children'][0]
    assert count==3 and (a['side'],b['side'],c['side'])==('for','against','for')
    assert b['scores']['defense']==1.1 and c['argument']==['C']
    with pytest.raises(ValueError,match='Missing parent'):
        data.recover_trees(body.replace('Level-1','Level-3'),'Motion','for')


def test_historical_argument_recovery_never_uses_future_analysis():
    record=lambda mode,**kw:dict(mode=mode,stage='rebuttal',side='against',**kw)
    document=dict(motion='Motion',debate_thoughts={'for':[
        record('retrieve_on_prepared_tree',action_type='attack',target_claim='Target'),
        record('analyze_statement',claims=[dict(claim='Target',arguments=['Later premise'])]),
        record('retrieve_on_prepared_tree',action_type='attack',target_claim='Target')]})
    cases=data.historical_queries(document,'artifact.json')
    assert cases[0]['target_argument']=='' and cases[1]['target_argument']=='Later premise'


def test_generated_queries_require_balanced_distinct_targets():
    qs=[dict(action=a,target_claim=a+str(i),target_argument='Context') for a in ['attack','reinforce','rebut'] for i in range(5)]
    assert data.validate_generated(dict(queries=qs))==qs
    qs[1]['target_claim']=qs[0]['target_claim']
    with pytest.raises(ValueError,match='Duplicate'):data.validate_generated(dict(queries=qs))


def test_final_replay_retains_historical_context_and_removes_generated_premises():
    final=FIXTURES/'retrieval_all_motions_v2'
    cases=json.loads((final/'frozen_cases.json').read_text())
    historical=json.loads((FIXTURES/'retrieval_all_motions_v1/historical_cases.json').read_text())
    assert cases[:75]==historical
    supplemental=cases[75:]
    assert len(supplemental)==45
    assert all(c['query_origin']=='generated_supplement' for c in supplemental)
    assert all(c['target_argument']=='' and c['argument_source'] is None for c in supplemental)
    assert len({(c['slug'],c['target']) for c in supplemental})==45
