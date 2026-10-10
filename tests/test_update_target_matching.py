"""Update matching must preserve ownership, history and unresolved speech."""
from unittest.mock import Mock

import pytest

from debate_tree import DebateTree
from ouragents import TreeDebater
from streaming.tree_updates import apply_statements


def setup():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    a = own.update_node('propose', new_claim='Keep buses.', new_argument=[])
    b = other.update_node('propose', new_claim='Ban buses.', new_argument=[])
    return own, other, a, b


def statement(action, target, target_id=None, hint='opponent'):
    return dict(claim='Provide accessible transport.', content='Provide accessible transport.', arguments=['Access matters.'],
                purpose=[dict(action=action, target=target, target_id=target_id, targeted_debate_tree=hint)])


def apply(own, other, item, **kwargs):
    return apply_statements((own, other), [item], item['content'], 'for', **kwargs)


def test_id_resolves_paraphrase_without_embedding_or_changing_action():
    own, other, a, b = setup()
    other.get_most_similar_node = Mock(side_effect=AssertionError('No semantic mutation'))
    item = statement('rebut', 'Stop operating buses', b.node_id)
    events = apply(own, other, item)
    assert b.children[0].claim == item['claim'] and b.children[0].side == 'for'
    assert b.children[0].relation == 'reply' and b.argument == []
    assert any(e['action'] == 'TARGET_RESOLUTION' and e['method'] == 'id' and e['outcome'] == 'matched' for e in events)


@pytest.mark.parametrize('target', ['  BAN  buses! ', 'ban buses', 'Ban buses.'])
def test_legacy_text_recovers_only_formatting_variations(target):
    own, other, a, b = setup()
    apply(own, other, statement('attack', target))
    assert len(b.children) == 1


@pytest.mark.parametrize('target', ['Do not ban buses.', 'Ban some buses.', 'Ban buses after 2030.'])
def test_nearby_but_distinct_meaning_remains_unlinked(target):
    own, other, a, b = setup()
    events = apply(own, other, statement('attack', target))
    assert not b.children and own.root.children[-1].claim == 'Provide accessible transport.'
    assert any(e.get('reason') == 'text_not_found' for e in events)


def test_explicit_unknown_id_never_falls_back_to_matching_text():
    own, other, a, b = setup()
    events = apply(own, other, statement('attack', b.claim, 'missing'))
    assert not b.children and len(own.root.children) == 2
    assert any(e.get('reason') == 'unknown_id' for e in events)


def test_text_duplicate_resolves_by_speaker_but_not_by_arbitrary_branch():
    own, other, a, b = setup()
    same_side = own.update_node('propose', new_claim=b.claim)
    apply(own, other, statement('attack', b.claim))
    assert len(b.children) == 1 and not same_side.children
    duplicate = b.add_node(new_claim='A distinct objection', new_argument=[], side='for')
    duplicate.add_node(new_claim=b.claim, new_argument=[], side='against')
    events = apply(own, other, statement('attack', b.claim))
    assert len(b.children) == 2
    assert any(e.get('reason') == 'ambiguous_target' for e in events)


def test_unique_target_in_other_tree_recovers_incorrect_tree_hint():
    own, other, a, b = setup()
    apply(own, other, statement('attack', b.claim, hint='you'))
    assert len(b.children) == 1 and not a.children


def test_inactive_node_id_does_not_link_or_reactivate_old_branch():
    own, other, a, b = setup()
    b.position_status = 'withdrawn'
    events = apply(own, other, statement('attack', b.claim, b.node_id))
    assert not b.children and b.position_status == 'withdrawn'
    assert own.root.children[-1].claim == 'Provide accessible transport.'
    assert any(e.get('reason') == 'historical_target' for e in events)


def test_wrong_owner_rebut_is_never_reinterpreted_as_reinforce():
    own, other, a, b = setup()
    events = apply(own, other, statement('rebut', a.claim, a.node_id))
    assert not a.children and not a.argument
    assert any(e['action'] == 'REJECT_RELATION_OWNER' for e in events)


def test_direct_update_no_longer_converts_failed_rebut_or_calls_embeddings():
    own, other, a, b = setup()
    own.get_most_similar_node = Mock(side_effect=AssertionError('No semantic mutation'))
    assert own.update_node('rebut', new_claim='Reply', new_argument=['Reason'], target=a.claim) is None
    assert not a.argument and not a.children
    assert own.update_events[-1]['reason'] == 'owner_mismatch'


def test_corrections_disabled_does_not_turn_withdrawal_into_proposal():
    own, other, a, b = setup()
    events = apply(own, other, statement('retract', a.claim, a.node_id), allow_corrections=False)
    assert a.position_status == 'current' and len(own.root.children) == 1
    assert any(e['action'] == 'REJECT_CORRECTION_DISABLED' for e in events)


def test_regular_agent_without_planner_receives_registry_and_uses_id(monkeypatch):
    own, other, a, b = setup()
    item = statement('rebut', 'A paraphrased target', b.node_id)
    extraction = Mock(return_value=[item])
    monkeypatch.setattr('ouragents.extract_statement', extraction)
    p = TreeDebater.__new__(TreeDebater)
    p.side = 'for'; p.status = 'rebuttal'; p.motion = 'Transport'; p.use_debate_flow_tree = True
    p.debate_tree = own; p.oppo_debate_tree = other; p.debate_thoughts = []; p.helper_client = Mock()
    p._analyze_statement(item['content'], 'for')
    assert b.node_id in {n['node_id'] for n in extraction.call_args.kwargs['relation_targets']}
    assert len(b.children) == 1 and b.children[0].relation == 'reply'
    assert p.debate_thoughts[-1]['tree_updates']
