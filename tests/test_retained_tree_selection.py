"""Retained history must survive serialization without entering current tree views."""
import json
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from debate_tree import DebateTree
from ouragents import TreeDebater
from streaming.argument_revisions import revise_claim
from streaming.branch_planning import planning_material, parse_branch_state
from streaming.planning import IncrementalPlanner, PlanningConfig
from streaming.tree_grounding import attach_source, tree_targets
from streaming.tree_selection import select_nodes, selection_status
from streaming.tree_updates import apply_statements, target_registry


def add(tree, text, parent=None, side=None, relation='propose'):
    side = side or tree.side
    node = (parent or tree.root).add_node(new_claim=text, new_argument=[text], side=side)
    node.relation = relation
    node.update_status('proposed')
    assert attach_source(node, text, text, side)
    return node


def amend(trees, node, action, text):
    return revise_claim(trees, target=node.claim, target_id=node.node_id, side=node.side,
                        action=action, claim=text, arguments=[text], source=text)


def test_withdrawal_keeps_complete_path_but_selects_independent_live_branch_after_roundtrip():
    tree = DebateTree('Transport', 'for')
    old = add(tree, 'HISTORICAL_ALL_WEATHER_SERVICE')
    objection = add(tree, 'HISTORICAL_STORM_OBJECTION', old, 'against', 'attack')
    reply = add(tree, 'HISTORICAL_RESPONSE_TO_STORM', objection, 'for', 'reply')
    independent = add(tree, 'The service still reaches another pier.')
    assert amend([tree], old, 'retract', 'I no longer claim operation in all weather.') == 1
    assert old in tree.root.children and old.children == [objection] and objection.children == [reply]
    assert selection_status(old) == 'withdrawn'
    assert selection_status(objection) == selection_status(reply) == 'needs_review'
    assert objection.get_node_info()['position_status'] == 'current'
    assert objection.get_node_info()['selection_status'] == 'needs_review'
    restored = DebateTree.from_json(tree.get_tree_info())
    assert restored.get_tree_info() == tree.get_tree_info()
    targets = tree_targets([restored], 'for')
    assert [n['node_id'] for n in targets] == [independent.node_id]
    assert 'HISTORICAL_' not in json.dumps(targets)
    assert 'HISTORICAL_RESPONSE_TO_STORM' in restored.print_tree(include_status=True)
    assert 'Selection: needs_review' in restored.print_tree(include_status=True)
    statuses = {n['node_id']: n['selection_status'] for n in target_registry([restored])}
    assert statuses[old.node_id] == 'withdrawn' and statuses[reply.node_id] == 'needs_review'


def test_revision_adds_version_without_moving_old_responses_or_losing_old_evidence():
    tree = DebateTree('Transport', 'for')
    old = add(tree, 'Ban all cars.')
    old.evidence = [{'text': 'Original evidence'}]
    response = add(tree, 'Ambulances need roads.', old, 'against', 'attack')
    assert amend([tree], old, 'revise', 'Only private cars downtown; ambulances remain allowed.') == 1
    new = tree.root.children[-1]
    assert old.claim == 'Ban all cars.' and old.source_spans == ['Ban all cars.']
    assert old.children == [response] and old.evidence == [{'text': 'Original evidence'}]
    assert new.node_id != old.node_id and new.supersedes == old.node_id
    assert old.superseded_by == new.node_id and old.position_status == 'superseded'
    assert not new.children and not new.evidence and selection_status(response) == 'needs_review'
    assert [n['node_id'] for n in tree_targets([tree], 'for')] == [new.node_id]
    assert tree.revisions[-1]['replacement_id'] == new.node_id
    assert amend([tree], old, 'revise', new.claim) == 0  # no duplicate replacement


def test_reassertion_creates_current_node_without_reactivating_withdrawn_path():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    old = add(own, 'Run a pilot.')
    objection = add(own, 'Who pays?', old, 'against', 'attack')
    amend([own, other], old, 'retract', 'I no longer propose the pilot.')
    statement = {'claim': old.claim, 'arguments': [], 'content': 'I again propose: Run a pilot.',
                 'purpose': [{'action': 'propose', 'target': 'N/A', 'target_id': None}]}
    apply_statements((own, other), [statement], statement['content'], 'for')
    new = own.root.children[-1]
    assert new is not old and new.claim == old.claim and not new.children
    assert old.position_status == 'withdrawn' and old.children == [objection]
    assert new.source_spans == [statement['content']]
    assert [n['node_id'] for n in tree_targets((own, other), 'for')] == [new.node_id]


def test_reply_to_retained_inactive_target_is_preserved_without_a_false_live_link():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    old = add(other, 'A withdrawn objection.')
    amend([own, other], old, 'retract', 'I no longer make that objection.')
    statement = {'claim': 'A current explanation.', 'arguments': [], 'content': 'A current explanation.',
                 'purpose': [{'action': 'rebut', 'target': old.claim, 'target_id': old.node_id}]}
    events = apply_statements((own, other), [statement], statement['content'], 'for')
    assert not old.children and own.root.children[0].source_spans == [statement['content']]
    assert any(e['action'] == 'UNLINKED_CLAIM' for e in events)


def test_new_version_under_changed_ancestor_is_kept_as_unlinked_current_claim():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    parent = add(own, 'A broad policy.')
    old = add(own, 'A broad objection.', parent, 'against', 'attack')
    amend([own, other], parent, 'retract', 'I no longer propose this policy.')
    amend([own, other], old, 'revise', 'A narrower independent concern.')
    new = other.root.children[-1]
    assert old in parent.children and new.parent is other.root
    assert new.relation == 'propose' and new.supersedes == old.node_id
    assert selection_status(new) == 'current'


def test_selection_caps_targets_and_context_and_keeps_related_concession():
    ours, theirs = DebateTree('Transport', 'against'), DebateTree('Transport', 'for')
    objection = add(ours, 'A backup plan must be approved before launch.')
    reply = add(ours, 'The trial requires approval, but the operator is undecided.', objection, 'for', 'reply')
    concession = add(ours, 'I accept prior approval of the backup plan.', objection, 'for', 'concede')
    for i in range(20):
        add(theirs, f'Other independent claim {i}.')
    chosen, context = select_nodes((ours, theirs), 'for', max_targets=1, max_context_nodes=2)
    assert chosen == [reply] and context == [objection, concession]
    targets = tree_targets((ours, theirs), 'for', max_targets=1, max_context_nodes=2)
    rich = planning_material(targets, (ours, theirs), 'for', topology=True)
    flat = planning_material(targets, (ours, theirs), 'for', topology=False)
    assert rich['branch_briefs'][0]['other_replies_to_same_objection'][0]['node_id'] == concession.node_id
    assert concession.claim in rich['position_limits'] == flat['position_limits']
    assert 'Other independent claim' not in json.dumps(rich)
    assert len(theirs.root.children) == 20  # budget exclusion is not a mutation
    assert 'branch_briefs' not in flat and all('ancestors' not in n for n in flat['tree_targets'])


def test_attacked_or_unanswered_claims_are_not_automatically_retired():
    tree = DebateTree('Transport', 'for')
    attacked = add(tree, 'A contested current claim.')
    add(tree, 'An objection.', attacked, 'against', 'attack')
    attacked.status, attacked.scores = 'attacked', {'support': -100, 'defense': -100}
    silent = add(tree, 'A claim not mentioned again.')
    targets = tree_targets([tree], 'for')
    assert {n['node_id'] for n in targets} == {attacked.node_id, silent.node_id}
    assert targets[0]['node_id'] == silent.node_id


def test_recent_qualification_is_not_crowded_out_by_older_unanswered_claims():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    old = [add(own, f'Older claim {i}.') for i in range(12)]
    quote = 'Only a one-month trial with a published budget.'
    statement = {'claim': quote, 'arguments': [], 'content': quote,
                 'purpose': [{'action': 'propose', 'target': 'N/A', 'target_id': None}]}
    apply_statements((own, other), [statement], quote, 'for')
    targets = tree_targets((own, other), 'for', max_targets=1, max_context_nodes=1)
    assert len(targets) == 1 and targets[0]['claim'] == quote
    assert len(own.root.children) == 13 and all(selection_status(n) == 'current' for n in old)


def test_omitted_responses_are_reported_without_claiming_branch_is_unanswered():
    ours, theirs = DebateTree('Transport', 'against'), DebateTree('Transport', 'for')
    parent = add(ours, 'An objection.')
    target = add(ours, 'A current reply.', parent, 'for', 'reply')
    answer = add(ours, 'Our existing response.', target, 'against', 'attack')
    view = tree_targets((ours, theirs), 'for', max_targets=1, max_context_nodes=1)
    assert view[0]['responses'] == [] and view[0]['omitted_response_count'] == 1
    assert not view[0]['unanswered'] and answer in target.children


@pytest.mark.parametrize('mode', ['branch_tree', 'flat_tree'])
def test_generation_never_appends_retained_full_tree_even_when_selected_state_is_valid(mode):
    p = TreeDebater.__new__(TreeDebater)
    p.config = SimpleNamespace(streaming_tts=False)
    p.motion, p.side, p.oppo_side = 'Transport', 'against', 'for'
    p.act, p.counter_act, p.status = 'oppose', 'support', 'rebuttal'
    p.debate_tree, p.oppo_debate_tree = DebateTree(p.motion, p.side), DebateTree(p.motion, p.oppo_side)
    old = add(p.oppo_debate_tree, 'HISTORICAL_TARGET_MUST_NOT_BE_DELIVERED')
    amend([p.oppo_debate_tree], old, 'revise', 'Only a one-month trial is proposed.')
    new = p.oppo_debate_tree.root.children[-1]
    p.conversation, p.high_quality_evidence_pool, p.use_debate_flow_tree = [], [], True
    p.planner = IncrementalPlanner(PlanningConfig(mode=mode, max_tree_targets=1))
    p.planner.start('for:rebuttal')
    p.planner.chunks = [new.claim]
    p.planner.version = p.planner.plan_version = 1
    p.debate_tree.print_tree = p.oppo_debate_tree.print_tree = Mock(side_effect=AssertionError('Full tree leaked'))
    material = p._planning_context()
    p.planner.state = parse_branch_state('{"claims":[{"target":0}],"limits":[],"rebuttals":[]}', new.claim, material)
    p.planner.plan = json.dumps(p.planner.state)
    p.listen, p.speak, p._analyze_statement = Mock(), Mock(return_value='Delivered.'), Mock()
    p.rebuttal_generation([], 60)
    prompt = p.speak.call_args.args[0]
    assert new.claim in prompt and 'HISTORICAL_TARGET_MUST_NOT_BE_DELIVERED' not in prompt
    assert old in p.oppo_debate_tree.get_all_nodes()


def test_optional_retrieval_uses_selected_current_query_and_exemplar_nodes(monkeypatch):
    p = TreeDebater.__new__(TreeDebater)
    p.motion, p.side, p.status = 'Transport', 'for', 'rebuttal'
    p.planner = IncrementalPlanner(PlanningConfig(mode='flat_tree'))
    p.debate_tree = DebateTree(p.motion, p.side)
    old = add(p.debate_tree, 'HISTORICAL_QUERY_CLAIM')
    amend([p.debate_tree], old, 'retract', 'I no longer claim that.')
    current = add(p.debate_tree, 'Current trial proposal.')
    exemplar = DebateTree('Example', p.side)
    retired_example = add(exemplar, 'HISTORICAL_EXEMPLAR_CLAIM')
    amend([exemplar], retired_example, 'retract', 'No longer asserted.')
    example = add(exemplar, 'Current example claim.')
    p.debate_tree.print_tree = exemplar.print_tree = Mock(side_effect=AssertionError('Full tree leaked'))
    p._get_embedding_from_cache = Mock(return_value=[1.0, 0.0])
    p.pro_embeddings = [[[1.0, 0.0]]] * 3
    p.data_list = [{'motion': 'Example', 'pro_debate_tree_obj': exemplar,
                    'structured_arguments': [{'side': 'for', 'stage': 'rebuttal', 'claims': []}]}]
    p.debate_thoughts = []
    monkeypatch.setattr('ouragents.semantic_search', Mock(return_value=[[{'corpus_id': 0, 'score': 1.0}]]))
    _, feedback = p._get_retrieval_debate_tree()
    query = p._get_embedding_from_cache.call_args.args[0]
    assert current.claim in query and example.claim in feedback
    assert 'HISTORICAL_' not in query + feedback
    assert 'parent_id' not in query + feedback and 'relation' not in query + feedback


@pytest.mark.parametrize('topology', [True, False])
def test_indexed_limits_accept_verified_prior_sibling_sources_outside_current_prefix(topology):
    ours, theirs = DebateTree('Transport', 'against'), DebateTree('Transport', 'for')
    objection = add(ours, 'Approval is needed.')
    reply = add(ours, 'The operator remains undecided.', objection, 'for', 'reply')
    concession = add(ours, 'I accept approval before launch.', objection, 'for', 'concede')
    targets = tree_targets((ours, theirs), 'for', max_targets=1, max_context_nodes=2)
    assert [n['node_id'] for n in targets] == [reply.node_id]
    material = planning_material(targets, (ours, theirs), 'for', topology=topology)
    limit = material['position_limits'].index(concession.claim)
    raw = json.dumps({'claims': [{'target': 0}], 'limits': [limit], 'rebuttals': []})
    state = parse_branch_state(raw, reply.claim, material)
    assert state['limits'][0]['quote'] == concession.claim
    assert state['claims'][0]['node_id'] == reply.node_id


@pytest.mark.parametrize('mode', ['branch_tree', 'flat_tree'])
@pytest.mark.parametrize('select_claim', [True, False])
def test_bound_plan_invalidates_when_coverage_changes_outside_selected_claim_path(mode, select_claim):
    tree = DebateTree('Transport', 'for')
    target = add(tree, 'A current principal benefit.')
    boundary = add(tree, 'Only a one-month trial.')
    material = planning_material(tree_targets([tree], 'for'), [tree], 'for', topology=mode == 'branch_tree')
    index = next(i for i, n in enumerate(material['tree_targets']) if n['node_id'] == target.node_id)
    planner = IncrementalPlanner(PlanningConfig(mode=mode))
    planner.start('for:opening')
    planner.chunks = [target.claim, boundary.claim]
    planner.state = parse_branch_state(json.dumps({'claims': [{'target': index}] if select_claim else [], 'limits': [], 'rebuttals': []}),
                                       ' '.join(planner.chunks), material)
    planner.plan = json.dumps(planner.state)
    planner.version = planner.plan_version = 1
    old_version = next(n['version'] for n in material['tree_targets'] if n['node_id'] == target.node_id)
    planner.revalidate_tree(material)
    assert planner.state  # unchanged views retain both claim-bearing and ledger-only plans
    amend([tree], boundary, 'revise', 'Only a two-week trial.')
    changed = planning_material(tree_targets([tree], 'for'), [tree], 'for', topology=mode == 'branch_tree')
    assert next(n['version'] for n in changed['tree_targets'] if n['node_id'] == target.node_id) == old_version
    planner.revalidate_tree(changed)
    assert not planner.state
    assert planner.events[-1]['action'] == 'INVALID_TARGET'


def statement(node, action, quote):
    return {'claim': quote, 'arguments': [], 'content': quote,
            'purpose': [{'action': action, 'target': node.claim if node else 'N/A',
                         'target_id': node.node_id if node else None}]}


def test_earlier_support_in_same_chunk_does_not_resurrect_a_later_superseded_position():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    original = add(own, 'We will cover every home.')
    earlier = statement(original, 'reinforce', 'We will cover every home.')
    later = statement(original, 'revise', 'To clarify, coverage is limited to the clinic.')
    apply_statements((own, other), [earlier, later], earlier['content'] + ' ' + later['content'], 'for')
    current = tree_targets((own, other), 'for')
    assert [n['claim'] for n in current] == [later['claim']]
    assert original.claim == earlier['claim'] and original in own.root.children


def test_source_order_drives_recency_across_corrections_and_new_claims():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    first, second = add(own, 'Old first claim.'), add(own, 'Old second claim.')
    a = statement(first, 'revise', 'Only the first site will operate.')
    b = statement(second, 'revise', 'Only the second service is funded.')
    last = statement(None, 'propose', 'Before launch, approval must be published.')
    # Extraction order need not equal speech order.
    apply_statements((own, other), [last, b, a], ' '.join(x['content'] for x in (a, b, last)), 'for')
    selected = tree_targets((own, other), 'for', max_targets=1)
    assert selected[0]['claim'] == last['claim']


@pytest.mark.parametrize('reverse', [False, True])
def test_same_quote_replacement_wins_over_retract_across_separate_items(reverse):
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    original = add(own, 'Cover all homes.')
    quote = 'Actually, only the clinic receives coverage.'
    items = [statement(original, 'revise', quote), statement(original, 'retract', quote)]
    if reverse:
        items.reverse()
    events = apply_statements((own, other), items, quote, 'for')
    assert original.position_status == 'superseded'
    assert [n['claim'] for n in tree_targets((own, other), 'for')] == [quote]
    assert sum(e['action'] == 'APPLY_CORRECTION' for e in events) == 1


def test_later_reassertion_survives_earlier_withdrawal_in_same_chunk():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    original = add(own, 'Run a pilot.')
    withdrawal = statement(original, 'retract', 'I withdraw the pilot.')
    reassertion = statement(None, 'propose', 'After reconsideration, run a pilot.')
    reassertion['claim'] = original.claim
    apply_statements((own, other), [reassertion, withdrawal],
                     withdrawal['content'] + ' ' + reassertion['content'], 'for')
    assert original.position_status == 'withdrawn'
    current = tree_targets((own, other), 'for')
    assert len(current) == 1 and current[0]['node_id'] != original.node_id
    assert current[0]['sources'] == [reassertion['content']]


def test_unmatched_revision_does_not_jump_ahead_of_earlier_proposal():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    earlier = statement(None, 'propose', 'Only a trial is funded.')
    later = statement(None, 'revise', 'Approval must be published before launch.')
    later['purpose'][0]['target_id'] = 'missing'
    apply_statements((own, other), [later, earlier], earlier['content'] + ' ' + later['content'], 'for')
    assert tree_targets((own, other), 'for', max_targets=1)[0]['claim'] == later['claim']


def test_recent_correction_history_uses_speech_order_across_both_trees():
    own, other = DebateTree('Transport', 'for'), DebateTree('Transport', 'against')
    parent = add(other, 'An independent objection.')
    # Older events in the second tree must not evict newer events in the first.
    old = [add(other, f'Old claim {i}.', parent, 'for', 'reply') for i in range(6)]
    for i, node in enumerate(old):
        amend((own, other), node, 'retract', f'I withdraw old claim {i}.')
    latest = add(own, 'Newest claim.')
    amend((own, other), latest, 'retract', 'I withdraw the newest claim.')
    material = planning_material(tree_targets((own, other), 'for'), (own, other), 'for', topology=True)
    assert len(material['correction_history']) == 6
    assert material['correction_history'][-1]['quote'] == 'I withdraw the newest claim.'
    assert all(e['quote'] != 'I withdraw old claim 0.' for e in material['correction_history'])
    restored = tuple(DebateTree.from_json(t.get_tree_info()) for t in (own, other))
    assert planning_material(tree_targets(restored, 'for'), restored, 'for', topology=True) == material
