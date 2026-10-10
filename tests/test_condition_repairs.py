"""Regressions from real condition-retention failures, with no model requests."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streaming.constraint_review import audit_feedback, current_checklist, draft_units, opponent_sources
from streaming.grounding import LIMIT_KINDS, parse_state
from test_claim_constraints import condition, material, player, propose, trees


def raw_state(quote, kind):
    return json.dumps({'claims': [], 'limits': [{'kind': kind, 'quote': quote}], 'rebuttals': []})


@pytest.mark.parametrize('kind', LIMIT_KINDS)
def test_planning_accepts_condition_vocabulary_without_weakening_source_check(kind):
    quote = 'Six weeks, with separate costings before commitment.'
    assert parse_state(raw_state(quote, kind), quote)['limits'][0]['kind'] == kind
    with pytest.raises(ValueError, match='Source quote'):
        parse_state(raw_state('Invented prerequisite.', kind), quote)


def test_unknown_kind_is_still_rejected():
    with pytest.raises(ValueError, match='Invalid limit'):
        parse_state(raw_state('Six weeks.', 'proven_fact'), 'Six weeks.')


def test_limits_use_only_selected_opponent_context_claims_keep_own_sources():
    target = {'node_id': 'reply', 'side': 'for', 'claim': 'Staffing is open.', 'version': 'v1',
              'sources': ['Staffing is open.'], 'ancestors': [
                  {'node_id': 'ours', 'side': 'against', 'claim': 'Publish a budget.',
                   'sources': ['Our claimed finding.'], 'responses': [
                       {'node_id': 'grant', 'side': 'for', 'claim': 'Cost it first.',
                        'sources': ['Separate costings precede commitment.']}]}]}
    result = parse_state(raw_state('Separate costings precede commitment.', 'precondition'), '', tree_targets=[target])
    assert result['limits'][0]['kind'] == 'precondition'
    with pytest.raises(ValueError, match='Source quote'):
        parse_state(raw_state('Our claimed finding.', 'precondition'), '', tree_targets=[target])
    state = {'claims': [{'node_id': 'reply', 'quote': 'Separate costings precede commitment.'}],
             'limits': [], 'rebuttals': []}
    with pytest.raises(ValueError, match='attributed'):
        parse_state(json.dumps(state), '', tree_targets=[target])


@pytest.mark.parametrize('mode', ['flat_tree', 'branch_tree'])
def test_empty_extracted_conditions_do_not_hide_a_keyword_free_prerequisite(mode):
    pair = trees()
    quote = 'Scanning and the listening point need separate costings.'
    node = propose(pair, quote, [])
    p = player(pair, mode)
    entries = current_checklist(p)
    assert len(entries) == 1 and entries[0]['quote'] == quote
    assert entries[0]['kind'] == 'source_candidate' and entries[0]['node_id'] == node.node_id
    assert entries[0]['planned_target']
    if p.planner.config.branch_state:
        assert quote in p.planner.state['position_limits']
        assert material(pair, mode == 'branch_tree')['position_limits'] == [quote]


def check_row(status='not_applicable', draft='Staffing needs agreement.', exclusion='School visits are a separate proposal.'):
    return {'id': 'c', 'status': status, 'draft_quote': draft, 'exclusion_quote': exclusion,
            'reason': 'The draft addresses the other proposal.', 'fix': ''}


@pytest.mark.parametrize('planned,quote,exclusion,expected', [
    (True, 'Staffing needs agreement.', 'School visits are a separate proposal.', 'unchecked'),
    (False, '', 'School visits are a separate proposal.', 'unchecked'),
    (False, 'Staffing needs agreement.', '', 'unchecked'),
    (False, 'Staffing needs agreement.', 'Invented independence.', 'unchecked'),
    (False, 'Staffing needs agreement.', 'School visits are a separate proposal.', 'not_applicable'),
])
def test_irrelevance_needs_verified_evidence_and_cannot_silently_drop_planned_target(planned, quote, exclusion, expected):
    checklist = [dict(condition('School visits by appointment.', 'scope'), constraint_id='c', planned_target=planned)]
    raw = json.dumps({'checks': [check_row(draft=quote, exclusion=exclusion)], 'issues': []})
    result = json.loads(audit_feedback(raw, checklist, 'Staffing needs agreement.', sources=['School visits are a separate proposal.']))
    assert result['review_checks'][0]['status'] == expected


def assertion(index, status='supported', quote='Funding is unsettled.'):
    return {'sentence': index, 'status': status, 'source_quote': quote, 'reason': 'Source states the gap.', 'fix': ''}


def test_assertion_review_covers_every_draft_unit_and_rejects_forged_or_missing_support():
    draft = ('Funding is unsettled. Volunteers lack training. Who will fund it? '
             'Delays are inevitable. The trial will fail.')
    raw = json.dumps({'checks': [], 'issues': [], 'assertions': [
        assertion(0), assertion(1, quote='Volunteers lack training.'),
        assertion(2, 'nonfactual', ''), assertion(3), assertion(3), assertion(True), assertion(99)]})
    result = json.loads(audit_feedback(raw, [], draft, sources=['Funding is unsettled.']))
    assert len(result['assertion_checks']) == len(draft_units(draft)) == 5
    assert [c['status'] for c in result['assertion_checks']] == ['supported', 'unchecked', 'nonfactual', 'unchecked', 'unchecked']
    assert result['assertion_checks'][1]['draft_quote'] == 'Volunteers lack training.'
    assert result['assertion_checks'][1]['source_quote'] == ''
    assert result['invalid_sentence_ids'] == 2
    missing = json.loads(audit_feedback('{"checks":[],"issues":[]}', [], draft))
    assert all(c['status'] == 'unchecked' for c in missing['assertion_checks'])


def test_opponent_sources_do_not_turn_our_prior_assertions_into_evidence():
    p = player(trees(), 'linear')
    p.planner.chunks = ['A named volunteer attends.']
    history = [{'side': 'against', 'content': 'Volunteers lack training.'},
               {'side': 'for', 'content': 'A named volunteer attends.'}]
    sources = opponent_sources(p, history)
    assert sources == ['A named volunteer attends.']
    raw = json.dumps({'checks': [], 'assertions': [assertion(0, quote='Volunteers lack training.')], 'issues': []})
    assert json.loads(audit_feedback(raw, [], 'Volunteers lack training.', sources=sources))['assertion_checks'][0]['status'] == 'unchecked'


def test_real_feedback_revision_handoff_keeps_unsupported_sentence_checks_in_existing_calls():
    pair = trees()
    quote = 'A named volunteer attends. Complaints are recorded and published after the trial.'
    propose(pair, quote, [condition('A named volunteer attends.')])
    p = player(pair)
    p.simulated_audience = [SimpleNamespace(feedback=Mock(return_value=json.dumps({
        'checks': [], 'assertions': [assertion(0, 'unsupported', '')], 'issues': []})))]
    p.helper_client = Mock(return_value=['How will the volunteer respond to complaints during the trial?'])
    feedback, _ = p._get_feedback_from_audience('Volunteers lack training.', [{'side': 'for', 'stage': 'opening', 'content': quote}])
    p._length_adjust('Volunteers lack training.', feedback, [], '', 60, max_retry=1)
    assert p.simulated_audience[0].feedback.call_count == p.helper_client.call_count == 1
    prompt = p.helper_client.call_args.kwargs['prompt']
    assert 'assertion_checks' in prompt and 'unsupported' in prompt and 'Volunteers lack training.' in prompt
    assert 'Assertion review data' in p.simulated_audience[0].feedback.call_args.args[0]


def test_provided_evidence_can_support_facts_but_not_condition_irrelevance():
    from streaming.constraint_review import supplied_evidence
    p = player(trees(), 'linear')
    p.evidence_pool = [{'content': 'Staff received training.'}, {'content': ''}]
    evidence = supplied_evidence(p)
    assert evidence == ['Staff received training.']
    raw = json.dumps({'checks': [], 'assertions': [assertion(0, quote=evidence[0])], 'issues': []})
    assert json.loads(audit_feedback(raw, [], evidence[0], evidence_sources=evidence))['assertion_checks'][0]['status'] == 'supported'
    checklist = [dict(condition('School visits by appointment.'), constraint_id='c')]
    raw = json.dumps({'checks': [check_row(exclusion=evidence[0])], 'issues': []})
    assert json.loads(audit_feedback(raw, checklist, 'Staffing needs agreement.', evidence_sources=evidence))['review_checks'][0]['status'] == 'unchecked'
