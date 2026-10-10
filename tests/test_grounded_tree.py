"""Real-tree offline regressions: attribution, target invalidation and scheduling."""
import copy
import json
from unittest.mock import Mock

import pytest

from debate_tree import DebateTree
from ouragents import TreeDebater
from streaming.argument_revisions import revise_claim
from streaming.grounding import parse_state
from streaming.branch_planning import parse_branch_state, planning_material
from streaming.planning import IncrementalPlanner, PlanningConfig
from streaming.tree_grounding import attach_source, tree_targets


def proposal(tree, claim, quote=None):
    node = tree.update_node('propose', new_claim=claim, new_argument=[], target=claim)
    quote = quote or claim
    assert attach_source(node, quote, quote, tree.side)
    return node


def state(node, quote=None):
    return {'claims': [{'node_id': node.node_id, 'quote': quote or node.source_spans[0]}],
            'limits': [], 'rebuttals': [{'target': 0, 'point': 'Ask about implementation.', 'assumptions': []}]}


def test_target_binding_rejects_another_nodes_quote_and_removed_ids():
    tree = DebateTree('Transport', 'for')
    a, b = proposal(tree, 'Limit cars.'), proposal(tree, 'Expand buses.')
    targets = tree_targets([tree], 'for')
    result = parse_state(json.dumps(state(a)), '', tree_targets=targets)
    assert result['rebuttals'][0]['target_node_id'] == a.node_id
    with pytest.raises(ValueError, match='attributed'):
        parse_state(json.dumps(state(a, b.claim)), b.claim, tree_targets=targets)
    tree.root.children.remove(a)
    with pytest.raises(ValueError, match='active'):
        parse_state(json.dumps(state(a)), a.claim, tree_targets=tree_targets([tree], 'for'))


def test_provenance_and_speaker_ownership_survive_serialization():
    tree = DebateTree('Transport', 'for')
    a = proposal(tree, 'Limit cars.')
    assert not attach_source(a, 'Allow ambulances.', 'Limit cars.', 'for')
    assert not attach_source(a, 'Limit cars.', 'Limit cars.', 'against')
    tree.update_node('attack', new_claim='Preserve access.', new_argument=[], target=a.claim)
    restored = DebateTree.from_json(tree.get_tree_info())
    assert tree_targets([restored], 'for') == tree_targets([tree], 'for')
    assert restored.root.children[0].side == 'for'
    assert restored.root.children[0].children[0].side == 'against'
    assert restored.root.children[0].source_spans == ['Limit cars.']


def test_structure_prioritizes_unanswered_attacks_without_marking_them_true():
    ours, theirs = DebateTree('Transport', 'against'), DebateTree('Transport', 'for')
    own = proposal(ours, 'Preserve access.')
    answered = proposal(theirs, 'Limit cars.')
    theirs.update_node('attack', new_claim='How will access work?', new_argument=[], target=answered.claim)
    direct = ours.update_node('attack', new_claim='Buses preserve access.', new_argument=[], target=own.claim)
    attach_source(direct, direct.claim, direct.claim, 'for')
    targets = tree_targets([theirs, ours], 'for')
    assert targets[0]['node_id'] == direct.node_id
    assert targets[0]['unanswered'] and targets[0]['attacks_our_claim']
    assert targets[0]['ancestors'][0]['node_id'] == own.node_id
    assert not targets[1]['unanswered']


def test_retained_historical_target_and_dependencies_invalidate_old_plan():
    tree = DebateTree('Transport', 'for')
    a = proposal(tree, 'Ban all cars.')
    tree.update_node('attack', new_claim='Ambulances need access.', new_argument=[], target=a.claim)
    p = IncrementalPlanner(PlanningConfig(mode='flat_tree'))
    p.start('for:opening')
    p.chunks, p.version, p.plan_version = [a.claim], 1, 1
    context = planning_material(tree_targets([tree], 'for'), [tree], 'for', topology=False)
    p.state = parse_branch_state('{"claims":[{"target":0}],"limits":[],"rebuttals":[]}', a.claim, context)
    p.plan = json.dumps(p.state)
    assert revise_claim([tree], target=a.claim, side='for', action='revise', claim='Only private cars.',
                        arguments=[], source='Only private cars.') == 1
    assert a.children and a.source_spans == ['Ban all cars.'] and a.position_status == 'superseded'
    assert tree.root.children[-1].source_spans == ['Only private cars.']
    p.revalidate_tree(planning_material(tree_targets([tree], 'for'), [tree], 'for', topology=False))
    assert not p.state and 'Ambulances need access' not in p.instructions()
    assert p.events[-1]['action'] == 'INVALID_TARGET'
    assert tree.revisions[0]['before']['children']


def player(monkeypatch, mode='flat_tree'):
    p = TreeDebater.__new__(TreeDebater)
    p.motion, p.side, p.oppo_side, p.status = 'Transport', 'against', 'for', 'opening'
    p.debate_tree, p.oppo_debate_tree = DebateTree(p.motion, p.side), DebateTree(p.motion, p.oppo_side)
    p.planner = IncrementalPlanner(PlanningConfig(mode=mode))
    p._planning_turn_snapshot = None
    p.use_debate_flow_tree = True
    p.debate_thoughts, p.high_quality_evidence_pool, p.conversation = [], [], []
    def extraction(client, motion, speech, **kwargs):
        return [{'claim': speech, 'content': speech, 'arguments': [],
                 'purpose': [{'action': 'propose', 'targeted_debate_tree': 'you', 'target': speech}]}]
    monkeypatch.setattr('ouragents.extract_statement', extraction)
    def plan(prompt, **kwargs):
        nodes = p.oppo_debate_tree.root.children
        targets = p._planning_context()['tree_targets']
        index = next(i for i, target in enumerate(targets) if target['node_id'] == nodes[-1].node_id)
        return [json.dumps({'claims': [{'target': index}], 'limits': [],
                            'rebuttals': [{'target': 0, 'move': 'challenge_inference',
                                          'point': 'Ask about implementation.', 'assumptions': []}]})]
    p.helper_client = Mock(side_effect=plan)
    return p


@pytest.mark.parametrize('mode', ['branch_tree', 'flat_tree'])
def test_live_observation_keeps_real_tree_and_bound_grounding(monkeypatch, mode):
    p = player(monkeypatch, mode)
    p.observe_opponent('Limit cars.', 'for', 'opening')
    assert p.use_debate_flow_tree and p.oppo_debate_tree.root.children
    node = p.oppo_debate_tree.root.children[0]
    assert p.planner.state['rebuttals'][0]['target_node_id'] == node.node_id
    assert 'GROUNDING CHECK' in p._current_planning_instructions(grounding=True)
    assert 'Prepare compact JSON rebuttal choices' in p.helper_client.call_args.args[0]
    assert 'Future exception' not in p.helper_client.call_args.args[0]


def test_flat_tree_replays_asr_replacement_and_preserves_checkpoint(monkeypatch):
    p = player(monkeypatch, 'flat_tree')
    p.observe_opponent('Limit cars except ambulances.', 'for', 'opening')
    assert p.oppo_debate_tree.root.children[0].claim == 'Limit cars except ambulances.'
    checkpoint = copy.deepcopy(p.planner)
    p.finalize_opponent('Allow all cars.', 'for', 'opening')
    assert [n.claim for n in p.oppo_debate_tree.root.children] == ['Allow all cars.']
    assert p.planner.state['claims'][0]['text'] == 'Allow all cars.'
    assert checkpoint.state['claims'][0]['text'] == 'Limit cars except ambulances.'


def test_cross_turn_targets_keep_observed_sources_without_future_input(monkeypatch):
    p = player(monkeypatch)
    p.observe_opponent('Limit cars.', 'for', 'opening')
    p.finalize_opponent('Limit cars.', 'for', 'opening')
    first = p.oppo_debate_tree.root.children[0]
    p.observe_opponent('Expand buses.', 'for', 'rebuttal')
    result = parse_state(json.dumps(state(first)), 'Expand buses.',
                         tree_targets=p._planning_context()['tree_targets'])
    assert result['claims'][0]['quote'] == 'Limit cars.'
    assert first.node_id in {t['node_id'] for t in p._planning_context()['tree_targets']}


def test_serialized_old_tree_without_sources_cannot_be_a_grounded_target():
    tree = DebateTree('Transport', 'for')
    tree.update_node('propose', new_claim='Limit cars.', new_argument=[], target='Limit cars.')
    assert tree_targets([tree], 'for') == []
