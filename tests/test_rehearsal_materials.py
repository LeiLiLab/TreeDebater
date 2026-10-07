"""Action-specific material selection: no paraphrase gate or sibling bundles."""
from types import SimpleNamespace as NS

import pytest

from utils.rehearsal_retrieval import retrieve, target_context
from test_rehearsal_retrieval import node, material_decisions
from test_rehearsal_relations import pool


TARGET = 'Labels and education are complementary.'


def select(nodes, decide, *, action='attack', target=TARGET, argument='', **kw):
    return retrieve(action, target, 'for', 'against', [pool(*nodes)], [], 1, [1., 0.],
                    embed=lambda cs: [[0.6, 0.8] for _ in cs], validate=decide,
                    target_argument=argument, **kw)


def test_related_parent_can_supply_useful_premise_and_explanation():
    reply = node('False confidence undermines learning.', side='for')
    reply.argument = ['Labels can reduce incentives to check sources and practise media literacy.']
    parent = node('We should not label AI.', [reply])
    def decide(cs):
        ds = material_decisions(cs, TARGET)
        ds[0]['materials'][0]['target_part'] = 'premise'
        ds[0]['materials'][0]['material_quote'] = 'reduce incentives to check sources'
        return ds
    info, matches = select([parent], decide)
    assert len(matches) == 1 and matches[0][4] == pytest.approx(0.6)
    assert reply.argument[0] in info[0]
    assert parent.claim != TARGET


def test_supporting_a_premise_does_not_require_a_paraphrase():
    material = node('Schools should teach programming.', side='for')
    material.argument = ['Programming develops critical thinking through evaluating solutions.']
    target = 'Critical thinking can be developed without writing essays.'
    def decide(cs):
        result = material_decisions(cs, target, 'supports')
        result[0]['materials'][0]['target_part'] = 'premise'
        return result
    info, _ = select([material], decide, action='reinforce', target=target)
    assert material.argument[0] in info[0]


@pytest.mark.parametrize('action,relation', [('attack', 'supports'), ('rebut', 'supports'),
                                            ('reinforce', 'challenges'), ('propose', 'answers')])
def test_action_direction_is_enforced_after_model_response(action, relation):
    material = node('Wrong-direction response', side='for')
    anchor = node(TARGET, [material]) if action in {'attack', 'rebut'} else material
    assert select([anchor], lambda cs: material_decisions(cs, TARGET, relation), action=action) == ([], [])


def test_only_selected_sibling_is_returned_with_its_explanation():
    good = node('Labels may undermine media literacy.', side='for')
    good.argument = ['People may stop checking sources when they trust labels blindly.']
    bad = node('Labels are costly to print.', side='for')
    anchor = node(TARGET, [good, bad])
    def decide(cs):
        result = material_decisions(cs, TARGET)
        result[0]['materials'][1]['relation'] = 'related'
        return result
    info, matches = select([anchor], decide)
    assert len(info) == len(matches) == 1
    assert good.argument[0] in info[0] and bad.claim not in info[0]


def test_output_limit_counts_materials_not_parent_bundles():
    parent = node(TARGET, [node(f'reply {i}', side='for') for i in range(5)])
    info, matches = select([parent], lambda cs: material_decisions(cs, TARGET), max_results=2)
    assert len(info) == len(matches) == 2
    assert all('reply 2' not in text for text in info)


@pytest.mark.parametrize('patch', [
    {'scope': 'incompatible'}, {'scope': 'uncertain'}, {'relation': 'uncertain'},
    {'target_quote': 'an invented premise'}, {'material_quote': 'fabricated supporting evidence'},
    {'material_quote': ''}, {'reason': ''}, {'id': True}, {'target_part': 'topic'},
])
def test_invalid_or_unverifiable_material_is_rejected(patch):
    anchor = node(TARGET, [node('reply', side='for')])
    def decide(cs):
        result = material_decisions(cs, TARGET)
        result[0]['materials'][0].update(patch)
        return result
    assert select([anchor], decide) == ([], [])


def test_duplicate_material_verdicts_fail_closed():
    anchor = node(TARGET, [node('reply', side='for')])
    def decide(cs):
        result = material_decisions(cs, TARGET)
        result[0]['materials'] *= 2
        return result
    assert select([anchor], decide) == ([], [])


def test_premise_quote_can_come_from_live_target_argument():
    anchor = node('Parent', [node('reply', side='for')])
    argument = 'Labels will teach every reader how to check sources.'
    decide = lambda cs: material_decisions(cs, 'teach every reader how to check sources')
    assert select([anchor], decide, argument=argument)[0]
    assert select([anchor], decide) == ([], [])


def test_direct_material_precedes_premise_material():
    anchor = node(TARGET, [node('premise', side='for'), node('direct', side='for')])
    def decide(cs):
        result = material_decisions(cs, TARGET)
        result[0]['materials'][0]['target_part'] = 'premise'
        return result
    info, _ = select([anchor], decide, max_results=1)
    assert 'direct' in info[0] and 'premise' not in info[0]


def tree(*nodes):
    return NS(get_all_nodes=lambda: nodes)


def test_context_uses_live_argument_constraints_and_explicit_tree():
    target = node(TARGET)
    target.node_id = 'live'
    target.argument = ['Only voluntary labels on political advertisements.']
    target.constraints = [{'kind': 'scope', 'quote': 'political advertisements'}]
    target.parent = node('A transparency proposal.', side='for')
    duplicate = node(TARGET)
    result = target_context({'target_claim': TARGET, 'targeted_debate_tree': 'opponent',
                             'target_argument': 'stale argument'},
                            {'you': tree(duplicate), 'opponent': tree(target)}, 'against')
    assert result['argument'] == target.argument[0]
    assert result['constraints'] == target.constraints
    assert result['ancestors'][0]['claim'] == target.parent.claim


def test_ambiguous_context_requires_id_and_never_uses_semantic_neighbor():
    first, second = node(TARGET), node(TARGET)
    first.node_id, second.node_id = 'first', 'second'
    second.argument = ['Specific exception.']
    trees = {'you': tree(first, second, node('A semantically similar claim.'))}
    assert target_context({'target_claim': TARGET}, trees, 'against')['argument'] == ''
    assert target_context({'target_claim': TARGET, 'target_node_id': 'second'}, trees,
                          'against')['argument'] == 'Specific exception.'
    assert target_context({'target_claim': 'No exact match'}, trees, 'against')['argument'] == ''


@pytest.mark.parametrize('status', ['withdrawn', 'superseded', 'needs_review'])
def test_stale_target_context_is_not_used(status):
    target = node(TARGET)
    target.position_status = status
    assert target_context({'target_claim': TARGET}, {'you': tree(target)}, 'against')['argument'] == ''


def test_changed_ancestor_invalidates_descendant_material():
    parent = node(TARGET, [node('response', side='for')])
    parent.parent = NS(position_status='withdrawn', parent=None)
    assert select([parent], lambda cs: material_decisions(cs, TARGET)) == ([], [])
