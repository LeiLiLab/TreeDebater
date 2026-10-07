"""Offline two-stage retrieval and agent-wiring regressions."""
import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from utils.rehearsal_retrieval import retrieve, relation_prompt
from test_rehearsal_retrieval import SRC, load_function, node, material_decisions


def pool(*nodes):
    return NS(get_node_by_side=lambda side: [n for n in nodes if n.side == side])


def decisions(candidates):
    return material_decisions(candidates, 'Labels protect creators.')


def run(nodes, validate=decisions, **kwargs):
    return retrieve('attack', 'Labels protect creators.', 'for', 'against',
                    [pool(*nodes)], [], 1, [1., 0.], validate=validate,
                    embed=lambda claims: [[0.7, 0.714] for _ in claims], **kwargs)


def objection(claim):
    return node(claim, [node('Reply to ' + claim, side='for')])


def test_low_similarity_useful_material_is_accepted_but_wrong_direction_is_rejected():
    good = objection('Labeling safeguards creators.')
    bad = objection('Labeling harms creators.')
    def judge(cs):
        return [material_decisions([c], 'Labels protect creators.',
                'challenges' if c['claim'] == good.claim else 'supports')[0] for c in cs]
    info, matches = retrieve('attack', 'Labels protect creators.', 'for', 'against',
                            [pool(bad, good)], [], 1, [1., 0.], validate=judge,
                            embed=lambda claims: [[1., 0.] if x == bad.claim else [0.7, 0.714] for x in claims])
    assert len(matches) == 1 and matches[0][3] == good.claim
    assert matches[0][4] < 0.8
    assert good.children[0].claim in info[0]


def test_top_k_is_global_bounded_and_outputs_are_capped():
    judge = Mock(side_effect=decisions)
    info, matches = run([objection(str(i)) for i in range(20)], validate=judge,
                        candidate_k=5, max_results=2)
    assert len(judge.call_args.args[0]) == 5
    assert len(info) == len(matches) == 2


def test_opponent_pool_participates_without_own_pool():
    target = objection('Labels protect creators.')
    info, matches = retrieve('attack', target.claim, 'for', 'against', None, [pool(target)], 1,
                            [1.], embed=lambda cs: [[1.] for _ in cs], validate=decisions)
    assert info and matches[0][0] == 'Prepared-Opponent-Tree-Retrieval'


def test_side_filter_and_unusable_materials_apply_before_recall():
    wrong = node('wrong side', [node('reply', side='for')], side='for')
    no_reply = node('leaf')
    wrong_reply = node('wrong reply', [node('still opponent', side='against')])
    judge = Mock(side_effect=decisions)
    assert run([wrong, no_reply, wrong_reply], judge) == ([], [])
    judge.assert_not_called()


@pytest.mark.parametrize('answer', [None, {}, [], [{'id': 0, 'relation': 'uncertain', 'usable_material_ids': [0]}],
    [{'id': 0, 'relation': 'equivalent', 'usable_material_ids': [99]}],
    [{'id': 99, 'relation': 'equivalent', 'usable_material_ids': [0]}],
    [{'id': 0, 'relation': 'equivalent', 'usable_material_ids': [0]}] * 2])
def test_invalid_or_unconfirmed_relations_fail_closed(answer):
    assert run([objection('target')], lambda cs: answer) == ([], [])


def test_no_validator_does_not_accept_even_exact_claim():
    assert run([objection('Labels protect creators.')], None) == ([], [])


def test_deduplication_does_not_consume_candidate_budget():
    first, second = objection('one'), objection('two')
    judge = Mock(side_effect=decisions)
    _, matches = run([first, first, second], judge, candidate_k=2)
    assert len(judge.call_args.args[0]) == len(matches) == 2


def test_reinforce_returns_only_selected_support_material():
    support = node('same claim', side='for')
    support.argument = 'supporting details'
    info, matches = retrieve('reinforce', 'same claim', 'for', 'against', [pool(support)], [], 1,
                            [1.], embed=lambda cs: [[1.] for _ in cs],
                            validate=lambda cs: material_decisions(cs, 'same claim', 'supports'))
    assert 'supporting details' in info[0]
    assert support.argument == 'supporting details'  # retrieval does not mutate nodes


def test_full_pools_are_prepared_without_root_similarity_gate():
    method = load_function(SRC / 'ouragents.py', '_get_prepared_tree',
                           {'PrepareTree': NS(from_json=lambda x: NS(root=NS(claim=x)))}, cls='TreeDebater')
    p = NS(side='for', claim_pool=[], rehearsal_claim_pool=[[{'tree_structure': str(i)}] for i in range(15)],
           oppo_claim_pool=[[{'tree_structure': str(i)}] for i in range(18)],
           main_claims=[], status='opening', debate_thoughts=[])
    assert len(method(p, 'for')) == 15
    assert len(method(p, 'against')) == 18


def test_relation_cache_includes_target_argument_and_reuses_identical_requests():
    api = Mock(return_value=([{'id': 0, 'relation': 'uncertain', 'usable_material_ids': []}], ''))
    method = load_function(SRC / 'ouragents.py', '_validate_rehearsal_candidates',
                           {'get_response_with_retry': api, 'RehearsalRelationResponse': object,
                            'logger': logging.getLogger('test')}, cls='TreeDebater')
    p = NS(motion='motion', helper_client=object())
    action = {'action': 'attack', 'target_claim': 'target', 'target_argument': 'condition A'}
    method(p, action, [])
    method(p, action, [])
    assert api.call_count == 1
    method(p, dict(action, target_argument='condition B'), [])
    assert api.call_count == 2


def test_listening_twice_keeps_opponent_pool():
    method = load_function(SRC / 'ouragents.py', 'listen', {}, cls='TreeDebater')
    p = NS(oppo_side='against', side='for', _add_message=Mock(), use_debate_flow_tree=False,
           use_rehearsal_tree=True, prepared_oppo_tree_list=None, _get_prepared_tree=Mock(return_value=['tree']))
    history = [{'side': 'against', 'stage': 'opening', 'content': 'text'}]
    method(p, history)
    method(p, history)
    assert p.prepared_oppo_tree_list == ['tree']
    p._get_prepared_tree.assert_called_once_with('against')


def test_prompt_distinguishes_polarity_and_material_applicability():
    prompt = relation_prompt('motion', 'rebut', 'claim', 'limited to adults', [])
    assert 'limited to adults' in prompt
    assert 'polarity' in prompt and 'causal direction' in prompt
    assert 'material_quote' in prompt and 'TARGET IS AN OPPONENT' in prompt


@pytest.mark.parametrize("limit", [3, 8, 10])
def test_loading_retains_full_rehearsal_pool_without_growing_planning_prompt(tmp_path, limit):
    import json
    import os
    own = [[{'tree_structure': str(i)}] for i in range(16)]
    other = [[{'tree_structure': str(i)}] for i in range(20)]
    file = tmp_path / 'pool_for.json'
    file.write_text(json.dumps(own))
    (tmp_path / 'pool_against.json').write_text(json.dumps(other))
    method = load_function(SRC / 'ouragents.py', 'claim_generation',
                           {'os': os, 'json': json}, cls='TreeDebater')
    p = NS(pool_file=str(file), side='for', oppo_side='against', config=NS(claim_pool_limit=limit))
    method(p, 50)
    assert len(p.claim_pool) == limit
    assert p.rehearsal_claim_pool == own and p.oppo_claim_pool == other


def test_agent_retrieval_initializes_pools_and_passes_relation_validator():
    from contextlib import nullcontext
    api = Mock(return_value=(['validated material'], []))
    method = load_function(SRC / 'ouragents.py', '_retrieve_on_prepared_tree', {
        'get_retrieval_from_rehearsal_tree': api,
        'REMAINING_ROUND_NUM': {'opening_for': 3},
        'timed_phase': lambda *a, **kw: nullcontext(),
        'logger': logging.getLogger('test'),
    }, cls='TreeDebater')
    p = NS(use_rehearsal_tree=True, prepared_tree_list=None, prepared_oppo_tree_list=None,
           _get_prepared_tree=lambda side: [side], side='for', oppo_side='against', status='opening',
           _get_embedding_from_cache=Mock(return_value=[1.]),
           debate_tree=NS(get_embedding_from_cache=Mock(), get_all_nodes=lambda: []),
           oppo_debate_tree=NS(get_all_nodes=lambda: []),
           _validate_rehearsal_candidates=Mock(return_value=[]), config=NS(rehearsal_mode="llm", rehearsal_candidate_k=7,
           rehearsal_max_results=2), debate_thoughts=[])
    action = {'action': 'attack', 'target_claim': 'claim'}
    assert method(p, action) == 'validated material'
    assert api.call_args.args[4:6] == (['for'], ['against'])
    assert api.call_args.kwargs['candidate_k'] == 7
    assert api.call_args.kwargs['max_results'] == 2
    api.call_args.kwargs['validate']([])
    enriched, candidates = p._validate_rehearsal_candidates.call_args.args
    assert enriched['action'] == action['action'] and enriched['target_claim'] == action['target_claim']
    assert enriched['target_context'] == {'argument': '', 'ancestors': [], 'constraints': []}
    assert candidates == []


def test_relation_response_schema_rejects_unknown_relation():
    from pydantic import ValidationError
    from utils.llm_schemas import RehearsalRelationResponse
    with pytest.raises(ValidationError):
        RehearsalRelationResponse.model_validate({'decisions': [
            {'id': 0, 'relation': 'similar', 'usable_material_ids': [0]}]})
