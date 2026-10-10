"""Offline stance-gate regressions; model judgments are controlled fixtures."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from streaming.flat_speaking import SegmentRejected
from streaming.overview_review import audit, review, review_payload
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, FRAMEWORK, prepared_player, prefix_helper, trace
from test_overview_review_gate import passing

pytestmark = pytest.mark.usefixtures('word_length_modes')
RECORDED = json.loads((Path(__file__).parent/'fixtures/v43_prefix_stance_reversal.json').read_text())


def inputs():
    candidate = copy.deepcopy(RECORDED['candidate'])
    data = dict(motion=RECORDED['motion'], our_side='for', stage='closing',
        debate_history=[], final_transcript='', heard_transcript='',
        opponent_sources=[], supplied_evidence=[])
    return candidate, data


@pytest.mark.parametrize('endpoint', [False, True])
def test_v43_wrong_framework_cannot_be_approved_by_original_or_new_all_true_review(endpoint):
    candidate, data = inputs()
    payload = review_payload(candidate, data, endpoint=endpoint)
    for verdict in (RECORDED['original_review'], passing(candidate['text'], 'for')):
        checked = audit(json.dumps(verdict), payload, endpoint=endpoint)
        assert checked['review_format_valid'] and not checked['accepted']
        assert checked['checks']['stance'] is False
        assert checked['conflicts'][0]['origin'] == 'local_framework_check'
    helper = Mock(side_effect=AssertionError('Known opposite framework must be rejected locally'))
    checked = review(candidate, data, helper, endpoint=endpoint)
    assert not checked['accepted'] and checked['format_attempts'] == []
    helper.assert_not_called()


@pytest.mark.parametrize('assigned,position', [
    ('for', 'AGAINST'), ('for', ' oppose. '), ('against', 'FOR'), ('against', 'support')])
def test_explicit_opposite_labels_are_normalized_and_rejected(assigned, position):
    candidate, data = inputs()
    data['our_side'] = assigned
    candidate['framework']['position'] = position
    helper = Mock()
    assert not review(candidate, data, helper)['accepted']
    helper.assert_not_called()


@pytest.mark.parametrize('text,expressed,quote', [
    ('How should we balance transparency and privacy?', 'neutral', ''),
    ('Our definition covers public posts.', 'neutral', 'Our definition covers public posts.'),
    ('They oppose labels, but we support this motion to help users.', 'for', 'we support this motion'),
    ('Labels cannot solve every problem, but they provide useful transparency.', 'for', 'they provide useful transparency'),
])
def test_freeform_framework_and_neutral_or_quoted_opposition_are_not_keyword_rejected(text, expressed, quote):
    candidate, data = inputs()
    candidate.update(text=text, framework=dict(candidate['framework'], position='Support safeguards against deception.'))
    response = passing(quote, expressed)
    helper = Mock(return_value=[json.dumps(response)])
    assert review(candidate, data, helper)['accepted']
    assert helper.call_count == 1


def test_matching_framework_does_not_replace_review_of_actual_spoken_stance():
    candidate, data = inputs()
    candidate['framework']['position'] = 'for'
    response = passing(candidate['text'], 'against')
    response['stance_ok'] = False
    response['conflicts'] = [dict(kind='stance', basis='stance_mismatch',
        draft_quote=candidate['text'], source_quote='',
        reason='The paragraph endorses the case against labeling despite the FOR assignment.')]
    helper = Mock(return_value=[json.dumps(response)])
    result = review(candidate, data, helper)
    assert result['review_format_valid'] and not result['accepted']
    assert helper.call_count == 1
    prompt = helper.call_args.kwargs['prompt']
    assert 'Check the actual spoken text even when the framework position matches' in prompt
    assert review_payload(candidate, data)['context']['our_position'] == 'support the motion'


def test_opposite_assessment_cannot_be_overridden_by_true_stance_flag():
    candidate, data = inputs()
    candidate['framework']['position'] = 'for'
    helper = Mock(return_value=[json.dumps(passing(candidate['text'], 'against'))])
    checked = review(candidate, data, helper)
    assert checked['review_format_valid'] and not checked['accepted']
    assert checked['checks']['stance'] is False
    assert checked['conflicts'][0]['origin'] == 'local_assessment_check'
    assert helper.call_count == 1


@pytest.mark.parametrize('fault', ['old_schema', 'missing_assessment', 'missing_stance', 'fake_quote',
    'empty_quote', 'no_reason', 'unclear', 'unknown_side', 'string_bool', 'contradictory_rejection'])
def test_incomplete_or_inconsistent_stance_review_is_never_accepted(fault):
    candidate, data = inputs()
    candidate['framework']['position'] = 'for'
    result = passing(candidate['text'], 'for')
    if fault == 'old_schema':
        result = copy.deepcopy(RECORDED['original_review'])
    elif fault == 'missing_assessment':
        result.pop('stance_assessment')
    elif fault == 'missing_stance':
        result.pop('stance_ok')
    elif fault == 'fake_quote':
        result['stance_assessment']['quote'] = 'Not in the speech.'
    elif fault == 'empty_quote':
        result['stance_assessment']['quote'] = ''
    elif fault == 'no_reason':
        result['stance_assessment']['reason'] = ''
    elif fault in ('unclear', 'unknown_side'):
        result['stance_assessment']['expressed_side'] = 'unclear' if fault == 'unclear' else 'both'
    elif fault == 'string_bool':
        result['stance_ok'] = 'true'
    else:
        result['stance_ok'] = False
        result['conflicts'] = [dict(kind='stance', basis='stance_mismatch', draft_quote=candidate['text'],
            source_quote='', reason='Reject, despite classifying the speech as FOR.')]
    helper = Mock(return_value=[json.dumps(result)])
    checked = review(candidate, data, helper)
    assert not checked['accepted'] and not checked['review_format_valid']
    assert helper.call_count == 2  # Existing bounded format retry, not approval.
