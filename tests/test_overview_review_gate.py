"""Lightweight overview gate: protocol regressions, not semantic model evaluation."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from streaming.overview_review import CHECKS, INSTRUCTIONS, audit, review


@pytest.fixture
def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/overview_review_live_v5.json').read_text())['cases']


def inputs(case):
    payload = case['payload']
    candidate = dict(text=payload['draft'], target_ids=payload['target_ids'], framework=payload['framework'])
    data = dict(payload['context'], condition_candidates=payload['checklist'],
                opponent_sources=payload['sources'], supplied_evidence=[])
    return candidate, data


def passing(draft='', side='neutral'):
    return dict(stance_ok=True,
                stance_assessment=dict(expressed_side=side, quote=draft,
                    reason='The paragraph does not endorse the opposing case.'),
                ready_to_speak_ok=True, latest_input_ok=True, conflicts=[])


def unpack(prompt):
    return json.loads(prompt.rsplit('\n', 1)[1])


def test_long_real_payload_needs_one_compact_review_with_full_transcript(recorded):
    candidate, data = inputs(recorded[1])
    data['our_definition'] = 'Our proposed scope allows private verification.'
    def helper(*, prompt, max_tokens):
        payload = unpack(prompt)
        assert payload['context']['final_transcript'] == data['final_transcript']
        assert payload['context']['debate_history'] == data['debate_history']
        assert payload['context']['our_definition'] == data['our_definition']
        assert data['our_definition'] not in payload['opponent_sources']
        assert data['our_definition'] not in payload['sources']
        assert data['opponent_sources'][0] in payload['sources']
        assert payload['draft'] == candidate['text']
        assert 'checklist' not in payload and 'final_input_units' not in payload
        assert 'Ordinary advocacy' in prompt and 'need not be proved' in prompt
        assert max_tokens == 1000
        return [json.dumps(passing())]
    helper = Mock(side_effect=helper)
    result = review(candidate, data, helper, endpoint=True)
    assert result['accepted'] and helper.call_count == 1
    assert set(result['checks']) == set(CHECKS)


def test_old_incomplete_report_is_inconclusive_then_repaired_without_text_change(recorded):
    case = recorded[0]
    candidate, data = inputs(case)
    prompts = []
    def helper(*, prompt, **kwargs):
        prompts.append(prompt)
        assert unpack(prompt)['draft'] == candidate['text']
        if len(prompts) == 1:
            return [case['responses'][0]['content']]
        assert 'REVIEW FORMAT REPAIR:' in prompt and 'ready_to_speak_ok must be a boolean' in prompt
        return [json.dumps(passing())]
    result = review(candidate, data, helper)
    assert result['accepted'] and len(prompts) == 2
    assert not result['format_attempts'][0]['review_format_valid']
    assert result['format_attempts'][0]['conflicts'] == []


@pytest.mark.parametrize('kind', CHECKS)
def test_explicit_conflict_is_rejected_without_format_retry(recorded, kind):
    candidate, data = inputs(recorded[1])
    result = passing()
    result[kind + '_ok'] = False
    if kind == 'stance':
        result['stance_assessment'] = dict(expressed_side='for', quote=candidate['text'],
                                          reason='The draft supports the motion, contrary to its assignment.')
    result['conflicts'] = [dict(kind=kind, basis='stance_mismatch' if kind == 'stance' else 'invalidated_premise', draft_quote=candidate['text'],
        source_quote=data['final_transcript'], reason='This span conflicts with the stated stance or proposal.')]
    helper = Mock(return_value=[json.dumps(result)])
    checked = review(candidate, data, helper, endpoint=True)
    assert checked['review_format_valid'] and not checked['accepted']
    assert helper.call_count == 1


@pytest.mark.parametrize('fault', ['vague', 'wrong_draft_quote', 'wrong_source_quote', 'not_boolean',
                                  'null', 'duplicate', 'missing_basis', 'advocacy_disagreement'])
def test_inconclusive_or_forged_review_is_never_silently_accepted(recorded, fault):
    candidate, data = inputs(recorded[1])
    result = passing()
    result['ready_to_speak_ok'] = False
    row = dict(kind='ready_to_speak', basis='misstated_commitment',
               draft_quote=candidate['text'], source_quote='', reason='Concrete framing mismatch.')
    result['conflicts'] = [row]
    if fault == 'vague':
        result['conflicts'] = []
    elif fault == 'wrong_draft_quote':
        row['draft_quote'] = 'not a span of the draft'
    elif fault == 'wrong_source_quote':
        row['source_quote'] = 'not a span of any source'
    elif fault == 'not_boolean':
        result['ready_to_speak_ok'] = 'false'
    elif fault == 'null':
        result['ready_to_speak_ok'] = None
    elif fault == 'missing_basis':
        row.pop('basis')
    elif fault == 'advocacy_disagreement':
        row['basis'] = 'opponent_disagrees'
    else:
        result['conflicts'].append(dict(row))
    helper = Mock(return_value=[json.dumps(result)])
    checked = review(candidate, data, helper, endpoint=True)
    assert not checked['accepted'] and not checked['review_format_valid']
    assert helper.call_count == 2


def test_unrequested_detailed_rows_are_not_a_gate(recorded):
    payload = recorded[1]['payload']
    raw = dict(passing(), checks=[], endpoint_checks=[])  # old optional reports do not matter
    assert audit(json.dumps(raw), payload, endpoint=True)['accepted']


def test_early_readiness_stage_is_part_of_review_payload(recorded):
    candidate, data = inputs(recorded[1])
    data['prefix_handoff'] = True
    helper = Mock(return_value=[json.dumps(passing())])
    assert review(candidate, data, helper)['accepted']
    prompt = helper.call_args.kwargs['prompt']
    assert unpack(prompt)['prefix_handoff'] is True
    assert 'ALREADY HEARD opponent point' in prompt
    assert 'reject attributions unsupported by heard opponent input' in prompt
    assert unpack(prompt)['opponent_sources'] == list(dict.fromkeys(data['opponent_sources']))
    assert 'before the final audio batch is transcribed' in prompt


def test_review_still_obeys_shared_call_budget(recorded):
    from streaming.config import OutputConfig
    from streaming.listening_prefix import PrefixPreparation, PreparationStopped
    candidate, data = inputs(recorded[1])
    helper = Mock(return_value=['{}'])
    prep = PrefixPreparation(1, helper, OutputConfig(listening_prefix_max_calls=1))
    with pytest.raises(PreparationStopped):
        review(candidate, data, prep._complete, endpoint=True)
    assert helper.call_count == prep._calls == 1


def test_v37_advocacy_uses_identical_decision_contract_before_and_after_endpoint():
    """Recorded draft/objection exercise prompt wiring, not live model accuracy."""
    case = json.loads((Path(__file__).parent / 'fixtures/overview_review_v37_advocacy.json').read_text())
    candidate = dict(text=case['draft'])
    base = dict(motion='Governments should require identity verification for social media accounts.',
        our_side='for', stage='closing', debate_history=[],
        final_transcript='', heard_transcript='', opponent_sources=[], supplied_evidence=[])
    prompts = []
    for endpoint in (False, True):
        source = case['opponent_objection'] if endpoint else ''
        data = dict(base, prefix_handoff=not endpoint,
                    heard_transcript=source, final_transcript=source,
                    opponent_sources=[source] if source else [])
        helper = Mock(return_value=[json.dumps(passing())])
        assert review(candidate, data, helper, endpoint=endpoint)['accepted']
        prompt = helper.call_args.kwargs['prompt']
        marker = 'ENDPOINT GATE: ' if endpoint else 'PREPARATION GATE: '
        rules, payload = prompt.rsplit(marker + '\n', 1)
        assert rules == INSTRUCTIONS
        assert json.loads(payload)['draft'] == case['draft']
        assert json.loads(payload)['context']['final_transcript'] == source
        assert 'not an isolated phrase' in rules
        assert 'wrong endorsed stance must still fail the separate stance check' in rules
        assert 'does not by itself invalidate that overview' in rules
        prompts.append(rules)
    assert prompts[0] == prompts[1]


@pytest.mark.parametrize('endpoint', [False, True])
@pytest.mark.parametrize('basis,draft,source', [
    ('stance_mismatch', 'We oppose the motion.', ''),
    ('fabricated_evidence', 'A proven 95 percent reduction settles this debate.', ''),
    ('misstated_commitment', 'Their policy requires a public identity register.',
     'Our policy keeps identities private.'),
    ('unheard_input', 'Our opponents concede the privacy issue.', ''),
    ('invalidated_premise', 'They provide no privacy safeguards.',
     'Our policy keeps identities private.'),
])
def test_concrete_defects_remain_blocking_at_both_stages(recorded, endpoint, basis, draft, source):
    candidate, data = inputs(recorded[1])
    candidate['text'] = draft
    data.update(prefix_handoff=True, opponent_sources=[source] if source else [])
    kind = ('stance' if basis == 'stance_mismatch' else
            'latest_input' if basis == 'invalidated_premise' else 'ready_to_speak')
    response = passing()
    if kind == 'stance':
        data['our_side'] = 'for'
        candidate['framework'] = dict(candidate['framework'], position='for')
        response['stance_assessment'] = dict(expressed_side='against', quote=draft,
                                            reason='The spoken paragraph opposes the motion.')
    response[kind + '_ok'] = False
    response['conflicts'] = [dict(kind=kind, basis=basis, draft_quote=draft,
        source_quote=source, reason='Correct the factual premise, attribution or assigned stance.')]
    helper = Mock(return_value=[json.dumps(response)])
    result = review(candidate, data, helper, endpoint=endpoint)
    assert result['review_format_valid'] and not result['accepted']
    assert result['conflicts'] == response['conflicts'] and helper.call_count == 1


@pytest.mark.parametrize('fault', ['missing_source', 'wrong_basis'])
def test_latest_input_rejection_needs_grounded_premise_conflict(recorded, fault):
    candidate, data = inputs(recorded[1])
    response = passing()
    response['latest_input_ok'] = False
    response['conflicts'] = [dict(kind='latest_input',
        basis='misstated_commitment' if fault == 'wrong_basis' else 'invalidated_premise',
        draft_quote=candidate['text'],
        source_quote='' if fault == 'missing_source' else data['final_transcript'],
        reason='Correct the attributed commitment.')]
    helper = Mock(return_value=[json.dumps(response)])
    result = review(candidate, data, helper, endpoint=True)
    assert not result['review_format_valid'] and not result['accepted']
    assert helper.call_count == 2
