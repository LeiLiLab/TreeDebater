"""Offline ownership, revision, selected-view and final-review regressions."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from debate_tree import DebateTree
from ouragents import TreeDebater
from streaming.argument_revisions import revise_claim
from streaming.branch_planning import LIMIT_MARKERS, parse_branch_state, planning_material
from streaming.claim_constraints import bind_constraints, constraint_ledger, exported_constraints
from streaming.constraint_review import audit_feedback, current_checklist
from streaming.grounding import parse_state
from streaming.planning import IncrementalPlanner, PlanningConfig
from streaming.tree_grounding import tree_targets
from streaming.tree_updates import apply_statements, target_registry
from utils.llm_schemas import LinkedStatementsResponse


def condition(quote, kind='precondition', origin=None):
    return {'kind': kind, 'quote': quote, 'source_node_id': origin}


def statement(quote, conditions, action='propose', target=None):
    return {'claim': quote, 'content': quote, 'arguments': [], 'constraints': conditions,
            'purpose': [{'action': action, 'target': target.claim if target else 'N/A',
                         'target_id': target.node_id if target else None, 'targeted_debate_tree': 'you'}]}


def trees():
    return DebateTree('Community services', 'for'), DebateTree('Community services', 'against')


def propose(pair, quote, conditions):
    apply_statements(pair, [statement(quote, conditions)], quote, 'for')
    return pair[0].root.children[-1]


def material(pair, topology=True):
    return planning_material(tree_targets(pair, 'for'), pair, 'for', topology=topology)


def indexed():
    return json.dumps({'claims': [{'target': 0}], 'limits': [],
                       'rebuttals': [{'target': 0, 'move': 'challenge_support',
                                     'point': 'Ask for costings.', 'assumptions': []}]})


def player(pair, mode='branch_tree'):
    p = TreeDebater.__new__(TreeDebater)
    p.motion, p.side, p.oppo_side, p.status = 'Community services', 'against', 'for', 'rebuttal'
    p.oppo_debate_tree, p.debate_tree = pair
    p.conversation, p.high_quality_evidence_pool, p.debate_thoughts = [], [], []
    p.add_retrieval_feedback = False
    p.planner = IncrementalPlanner(PlanningConfig(mode=mode))
    p.planner.start('for:opening')
    p.planner.chunks = [' '.join(n.claim for n in pair[0].root.children)]
    p.planner.version = p.planner.plan_version = 1
    context = p._planning_context()
    if p.planner.config.branch_state:
        p.planner.state = parse_branch_state(indexed(), p.planner.chunks[0], context)
    elif p.planner.config.grounded_tree:
        n = context['tree_targets'][0]
        p.planner.state = parse_state(json.dumps({'claims': [{'node_id': n['node_id'], 'quote': n['sources'][-1]}],
                                                'limits': [], 'rebuttals': []}), '', tree_targets=context['tree_targets'])
    p.planner.plan = json.dumps(p.planner.state)
    return p


def test_keyword_free_condition_survives_serialization_all_planners_and_flat_ablation():
    pair = trees()
    quote = 'Scanning and the listening point need separate costings.'
    assert not LIMIT_MARKERS.search(quote)
    node = propose(pair, quote, [condition(quote)])
    restored = DebateTree.from_json(pair[0].get_tree_info())
    assert exported_constraints(restored.root.children[0]) == exported_constraints(node)
    assert target_registry(pair)[0]['constraints'] == exported_constraints(node)
    branch, flat = material(pair), material(pair, False)
    assert branch['constraints'] == flat['constraints']
    for view in (branch, flat):
        assert view['position_limits'] == [quote]
        state = parse_branch_state(indexed(), quote, view)
        assert state['limits'] == [] and state['constraints'][0]['quote'] == quote
        assert state['constraints'][0]['node_id'] == node.node_id
    assert 'branch_briefs' not in flat
    assert not {'ancestors', 'responses', 'unanswered', 'parent_id'} & flat['tree_targets'][0].keys()
    for mode in ('grounded_tree', 'light_tree', 'branch_tree', 'flat_tree'):
        p = player(pair, mode)
        assert p.planner.state['constraints'][0]['kind'] == 'precondition'
        assert quote in p._current_planning_instructions(grounding=True)
    old = pair[0].get_tree_info()
    old['structure']['children'][0].pop('constraints')
    assert DebateTree.from_json(old).root.children[0].constraints == []


def test_cross_branch_source_and_forged_owner_rejected_without_dropping_claim():
    pair = trees()
    a = propose(pair, 'Keep the access path clear.', [condition('Keep the access path clear.')])
    quote = 'Set up a listening point.'
    bad = statement(quote, [condition(a.claim), condition(a.claim, origin=a.node_id)])
    events = apply_statements(pair, [bad], a.claim + ' ' + quote, 'for')
    b = pair[0].root.children[-1]
    assert b.claim == quote and b.constraints == [] and a.constraints
    assert sum(e['action'] == 'REJECT_CONSTRAINT' for e in events) == 2


def test_revision_explicit_carry_keeps_exception_but_replaces_old_timing():
    pair = trees()
    old = propose(pair, 'Run for six weeks. Ambulances are exempt.',
                  [condition('Run for six weeks.', 'timing'), condition('Ambulances are exempt.', 'exception')])
    original = copy.deepcopy(old.constraints)
    item = statement('Run for three weeks.', [condition('Run for three weeks.', 'timing'),
                     condition('Ambulances are exempt.', 'exception', old.node_id)], 'revise', old)
    apply_statements(pair, [item], item['content'], 'for')
    new = pair[0].root.children[-1]
    assert old.constraints == original and old.position_status == 'superseded'
    assert pair[0].revisions[-1]['before']['constraints'] == original
    ledger = constraint_ledger(tree_targets(pair, 'for'), 'for')
    assert {c['quote'] for c in ledger} == {'Run for three weeks.', 'Ambulances are exempt.'}
    assert {c['node_id'] for c in ledger} == {new.node_id}
    assert next(c for c in ledger if c['kind'] == 'exception')['source_node_id'] == old.node_id
    assert new.source_spans[-1] == item['content']
    revise_claim(pair, target=new.claim, target_id=new.node_id, side='for', action='revise',
                 claim='End the trial.', arguments=[], source='End the trial.')
    assert pair[0].root.children[-1].constraints == []  # no implicit carry


@pytest.mark.parametrize('action', ['reinforce', 'rebut', 'concede'])
def test_conditions_attach_to_actual_speaker_node(action):
    pair = trees()
    owner = pair[0] if action == 'reinforce' else pair[1]
    target = owner.update_node('propose', new_claim='Access needs protection.', new_argument=[], target='Access needs protection.')
    quote = 'I accept keeping the access path clear.'
    apply_statements(pair, [statement(quote, [condition(quote, 'concession')], action, target)], quote, 'for')
    actual = target if action == 'reinforce' else target.children[0]
    assert actual.side == 'for' and actual.constraints[0]['source_node_id'] == actual.node_id
    if actual is not target:
        assert target.constraints == []


@pytest.mark.parametrize('mode', ['grounded_tree', 'light_tree', 'branch_tree', 'flat_tree'])
def test_condition_only_change_invalidates_plan_and_fallback_checklist_is_fresh(mode):
    pair = trees()
    node = propose(pair, 'A trial needs separate costings.', [])
    p = player(pair, mode)
    bind_constraints(node, [condition(node.claim)])
    p._current_planning_instructions(grounding=True)
    assert p.planner.state == {} and p.planner.events[-1]['action'] == 'INVALID_TARGET'
    assert current_checklist(p)[0]['quote'] == node.claim
    revise_claim(pair, target=node.claim, target_id=node.node_id, side='for', action='retract',
                 claim=node.claim, arguments=[], source='Withdraw the trial.')
    assert current_checklist(p) == []


def test_context_concession_constraints_are_equal_for_flat_and_branch_and_invalidate_plan():
    pair = trees()
    objection = pair[1].update_node('propose', new_claim='Staffing needs a budget.', new_argument=[], target='Staffing needs a budget.')
    for quote, action, conditions in [('I accept separate costings.', 'concede', [condition('I accept separate costings.', 'concession')]),
                                      ('Volunteers support the trial.', 'rebut', [])]:
        apply_statements(pair, [statement(quote, conditions, action, objection)], quote, 'for')
    target = tree_targets(pair, 'for', max_targets=1)
    branch = planning_material(target, pair, 'for', topology=True)
    flat = planning_material(target, pair, 'for', topology=False)
    assert branch['constraints'] == flat['constraints']
    assert branch['constraints'][0]['node_id'] == objection.children[0].node_id
    assert parse_branch_state(indexed(), '', flat)['constraints'] == branch['constraints']


def review_row(id_, status='preserved', quote='Separate costings are needed.'):
    return {'id': id_, 'status': status, 'draft_quote': quote, 'reason': 'The response addresses costing.', 'fix': ''}


def test_audit_validates_each_row_without_certifying_semantic_judgments():
    checklist = [dict(condition('Separate costings are needed.'), constraint_id=str(i)) for i in range(6)]
    rows = [review_row('0'), review_row('1', quote='Invented draft wording.'),
            review_row('2'), review_row('2'), review_row('3', 'missing', ''),
            review_row('4', 'not_applicable', ''), review_row('unknown')]
    audit = json.loads(audit_feedback(json.dumps({'checks': rows, 'issues': []}), checklist, 'Separate costings are needed.'))
    assert [r['status'] for r in audit['review_checks']] == ['preserved', 'unchecked', 'unchecked', 'missing', 'not_applicable', 'unchecked']
    assert audit['invalid_review_ids'] == 1 and 'not certification' in audit['evidence_validation']
    assert audit['review_checks'][1]['draft_quote'] == ''
    for malformed in ('Everything is preserved.', '{', 'null', '{"checks": null}'):
        audit = json.loads(audit_feedback(malformed, checklist, 'Draft.'))
        assert all(r['status'] == 'unchecked' for r in audit['review_checks'])
        assert audit['unverified_feedback'] == malformed and not audit['review_format_valid']


@pytest.mark.parametrize('mode', ['grounded_tree', 'light_tree', 'branch_tree', 'flat_tree'])
@pytest.mark.parametrize('quote,conditions', [
    ('Friday until eight, or the existing closing time if no volunteer attends. A four-week trial.',
     [condition('Friday until eight, or the existing closing time if no volunteer attends.', 'timing'),
      condition('A four-week trial.', 'scope')]),
    ('Two benches and three planters for three months. Keep access clear. A named waterer is needed.',
     [condition('Two benches and three planters for three months.', 'scope'),
      condition('Keep access clear.'), condition('A named waterer is needed.')]),
])
def test_failed_case_conditions_reach_existing_feedback_and_revision_calls(mode, quote, conditions):
    pair = trees()
    # Mocked extraction payload uses the real schema, transaction and plan parsers.
    response = LinkedStatementsResponse.model_validate({'statements': [statement(quote, conditions)]})
    extracted = response.model_dump()['statements']
    apply_statements(pair, extracted, quote, 'for')
    p = player(pair, mode)
    checklist = current_checklist(p)
    rows = [review_row(c['constraint_id'], 'missing', '') for c in checklist]
    audience = SimpleNamespace(feedback=Mock(return_value=json.dumps({'checks': rows, 'issues': []})))
    p.simulated_audience = [audience]
    p.helper_client = Mock(return_value=['The stated trial still needs an implementation budget.'])
    feedback, raw = p._get_feedback_from_audience('Discuss implementation.', [{'side': 'for', 'stage': 'opening', 'content': quote}])
    p._length_adjust('Discuss implementation.', feedback, [], '', 60, max_retry=1)
    assert audience.feedback.call_count == p.helper_client.call_count == 1
    for c in conditions:
        assert c['quote'] in audience.feedback.call_args.args[0]
        assert c['quote'] in p.helper_client.call_args.kwargs['prompt']
        assert c['quote'] in feedback
    assert json.loads(raw[0])['checks'] == rows
    assert 'Fresh condition checklist' in p.helper_client.call_args.kwargs['prompt']
    assert 'not certification' in feedback


def test_existing_extraction_call_requests_conditions_and_carry_registry(monkeypatch):
    from utils import helper
    pair = trees()
    node = propose(pair, 'Ambulances are exempt.', [condition('Ambulances are exempt.', 'exception')])
    quote = 'Run for three weeks.'
    payload = statement(quote, [condition(quote, 'timing'), condition(node.claim, 'exception', node.node_id)], 'revise', node)
    client = Mock()
    request = Mock(return_value=([payload], json.dumps({'statements': [payload]})))
    monkeypatch.setattr(helper, 'get_response_with_retry', request)
    result = helper.extract_statement(client, pair[0].motion, quote, tree=['', ''], side='for', stage='rebuttal',
                                      allow_corrections=True, relation_targets=target_registry(pair))
    assert request.call_count == 1
    assert request.call_args.kwargs['response_model'] is LinkedStatementsResponse
    prompt = request.call_args.args[1]
    assert 'CLAIM-OWNED QUALIFICATIONS' in prompt and node.node_id in prompt and node.claim in prompt
    apply_statements(pair, result, quote, 'for')
    assert {c['quote'] for c in pair[0].root.children[-1].constraints} == {quote, node.claim}


def test_revision_rebuilds_checklist_after_feedback_conditions_retire():
    pair = trees()
    old = propose(pair, 'Trial for six weeks.', [condition('Trial for six weeks.', 'timing')])
    p = player(pair)
    p.simulated_audience = [SimpleNamespace(feedback=Mock(return_value='Retain six weeks.'))]
    feedback, _ = p._get_feedback_from_audience('Discuss the trial.', [])
    revise_claim(pair, target=old.claim, target_id=old.node_id, side='for', action='revise',
                 claim='Trial for three weeks.', arguments=[], source='Trial for three weeks.',
                 constraints=[condition('Trial for three weeks.', 'timing')])
    p.planner.chunks = ['Trial for three weeks.']
    p.helper_client = Mock(return_value=['The three-week trial needs a budget.'])
    p._length_adjust('Discuss the trial.', feedback, [], '', 60, max_retry=1)
    prompt = p.helper_client.call_args.kwargs['prompt']
    fresh = json.loads(prompt.split('Fresh condition checklist (data):\n')[-1])
    assert [c['quote'] for c in fresh] == ['Trial for three weeks.']
    assert p.planner.state == {} and p.helper_client.call_count == 1


@pytest.mark.parametrize('mode', ['grounded_linear', 'light_linear'])
def test_linear_source_limits_are_reviewed_without_inventing_tree_ownership(mode):
    p = player(trees(), mode)
    p.planner.state = parse_state(json.dumps({'claims': [], 'limits': [
        {'kind': 'exception', 'quote': 'Ambulances are exempt.'}], 'rebuttals': []}), 'Ambulances are exempt.')
    ledger = current_checklist(p)
    assert len(ledger) == 1 and ledger[0]['node_id'] is None and ledger[0]['source_node_id'] is None
    assert ledger[0]['quote'] == 'Ambulances are exempt.'
