"""Reproduce v29 invalid handoff metadata without publishing or paid requests."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from streaming.flat_speaking import SegmentRejected
from streaming.listening_prefix import PrefixFormatRejected, PrefixPreparation, material, prepare
from test_full_speech import audio as audio
from test_listening_prefix import prepared_player, prefix_helper, conflict, prompt_data, TAIL
from test_listening_prefix import PREFIX, FRAMEWORK
from test_listening_overview import wait_for, event_count

pytestmark = pytest.mark.usefixtures('word_length_modes')
RECORDED = json.loads((Path(__file__).parent / 'fixtures/v29_prefix_target_ids.json').read_text())
MISSING_REASON = json.loads((Path(__file__).parent / 'fixtures/listening_v51_missing_framework_reason.json').read_text())


@pytest.mark.parametrize('record', MISSING_REASON, ids=lambda record: str(record['call_id']))
def test_recorded_v51_missing_explanation_keeps_speech_and_still_reviews_it(tmp_path, record):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.first_chunk_seconds = 16
    data = material(p, 'closing', history)
    original = copy.deepcopy(record['candidate'])
    base = prefix_helper(None)

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            return [json.dumps(original)]
        assert 'REPAIR:' not in prompt
        payload = prompt_data(prompt)
        assert payload['draft'] == original['text']
        return base(prompt=prompt, **kwargs)

    helper = Mock(side_effect=helper)
    result = prepare(data, helper, p.streaming_output_config)
    assert result['text'] == original['text']
    assert result['body_preparation']['draft'] == original['draft']
    assert 'reason' not in original['framework']
    assert result['framework']['reason'] == 'Author did not provide a framework explanation.'
    assert result['audits'][-1]['semantic_review'] and result['audits'][-1]['accepted']
    assert result['audits'][-1]['normalizations'][0]['field'] == 'framework.reason'
    assert helper.call_count == 2  # One authoring call and the unchanged publication review.


def test_framework_only_repair_names_missing_field_and_preserves_authored_text(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    candidate = dict(text=PREFIX, draft=TAIL, framework=copy.deepcopy(FRAMEWORK))
    candidate['framework'].pop('core_dispute')
    base = prefix_helper(None)

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            return [json.dumps(candidate)]
        if prompt.startswith('LISTENING FRAMEWORK REPAIR:'):
            payload = prompt_data(prompt)
            assert payload['issues'] == ['framework.core_dispute is required']
            assert payload['text'] == PREFIX and payload['draft'] == TAIL
            assert 'Do not rewrite text or draft.' in prompt
            # A provider returning extra speech fields still cannot overwrite it.
            return [json.dumps(dict(framework=FRAMEWORK, text='Wrong replacement.', draft='Wrong tail.'))]
        assert 'LISTENING PREFIX REPAIR:' not in prompt
        return base(prompt=prompt, **kwargs)

    helper = Mock(side_effect=helper)
    result = prepare(material(p, p.status, history), helper, p.streaming_output_config)
    assert result['text'] == PREFIX and result['body_preparation']['draft'] == TAIL
    assert result['framework'] == FRAMEWORK and result['audits'][-1]['accepted']
    assert helper.call_count == 3


def test_framework_only_repair_is_bounded_and_does_not_fabricate_missing_stance(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    framework = copy.deepcopy(FRAMEWORK)
    framework.pop('position')
    candidate = dict(text=PREFIX, draft=TAIL, framework=framework)

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            return [json.dumps(candidate)]
        assert prompt.startswith('LISTENING FRAMEWORK REPAIR:')
        assert 'framework.position is required' in prompt
        return [json.dumps(dict(framework=framework))]

    helper = Mock(side_effect=helper)
    with pytest.raises(PrefixFormatRejected):
        prepare(material(p, p.status, history), helper, p.streaming_output_config)
    assert helper.call_count == 2


def test_missing_reason_does_not_bypass_semantic_rejection(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    candidate = dict(text=PREFIX, draft=TAIL, framework=copy.deepcopy(FRAMEWORK))
    candidate['framework'].pop('reason')
    rejecting = prefix_helper(None, review_edit=lambda verdict, payload: conflict(verdict, payload))

    def helper(*, prompt, **kwargs):
        if prompt.startswith(('LISTENING PREFIX DRAFT:', 'LISTENING PREFIX REPAIR:')):
            return [json.dumps(candidate)]
        return rejecting(prompt=prompt, **kwargs)

    with pytest.raises(SegmentRejected, match='publication review'):
        prepare(material(p, p.status, history), helper, p.streaming_output_config)


def inputs(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    cfg = p.streaming_output_config
    cfg.first_chunk_seconds = 16
    cfg.listening_prefix_overlap_final_update = True
    data = material(p, p.status, history)
    # Recorded v29 paragraphs are FOR drafts; the shared dummy player is AGAINST.
    # This fixture exercises incomplete-speech recovery, not a stance reversal.
    data['our_side'] = RECORDED[0]['framework']['position']
    data['current_targets'] = [dict(node_id=RECORDED[0]['target_ids'][0])]
    return p, data


def test_recorded_incomplete_speech_is_repaired_without_legacy_target_ids(tmp_path):
    (p, data) = inputs(tmp_path)
    (repairs, reviews) = ([], [])

    def helper(*, prompt, **kwargs):
        payload = prompt_data(prompt) if 'LISTENING PREFIX' in prompt else {}
        if 'LISTENING PREFIX DRAFT:' in prompt:
            instructions = prompt.rsplit('\n', 1)[0]
            assert '"target_ids"' not in instructions
            return [json.dumps(RECORDED[0])]
        if 'LISTENING PREFIX REPAIR:' in prompt:
            repairs.append(payload)
            assert '"target_ids"' not in prompt
            assert 'Use only current targets and preserve' not in prompt
            if len(repairs) == 1:
                assert 'target_ids' not in payload['draft']
                assert payload['issues']
            return [json.dumps(dict(RECORDED[1], target_ids=[], draft=TAIL))]
        reviews.append(payload)
        result = dict(stance_ok=True, stance_assessment=dict(expressed_side=payload['context']['our_side'], quote=payload['draft'], reason='This paragraph supports the assigned side.'), ready_to_speak_ok=True, latest_input_ok=True, conflicts=[])
        return [json.dumps(result)]
    candidate = prepare(data, helper, p.streaming_output_config)
    assert 'target_ids' not in candidate and candidate['handoff_prechecked']
    assert candidate['audits'][-1]['accepted'] and len(reviews) == 1


@pytest.mark.parametrize('fault', ['missing_body', 'json'])
def test_new_heard_input_can_recover_same_framework_after_format_failure(tmp_path, audio, fault):
    p, data = inputs(tmp_path)
    _, encoded = audio
    p.streaming_output_config.listening_prefix_pre_synthesize = True
    base = prefix_helper(None)
    drafts = []
    def helper(*, prompt, **kwargs):
        payload = prompt_data(prompt) if 'LISTENING PREFIX' in prompt else {}
        if 'LISTENING PREFIX DRAFT:' in prompt or 'LISTENING PREFIX REPAIR:' in prompt:
            drafts.append(prompt)
            if 'NEW_HEARD_QUALIFICATION' not in kwargs['history_messages'][-1]['content']:
                return ['invalid JSON' if fault == 'json' else json.dumps(RECORDED[min(len(drafts)-1, 1)])]
            return [json.dumps(dict(RECORDED[1], target_ids=[], draft=TAIL))]
        return base(prompt=prompt, **kwargs)
    helper = Mock(side_effect=helper)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config,
                             audio_preparer=lambda *args: encoded(2))
    try:
        prep.offer(data)
        wait_for(lambda: event_count(prep) == 1)
        assert prep.peek() is None
        calls = helper.call_count
        # A tree-only update is not another heard batch and cannot trigger retries.
        same_heard = copy.deepcopy(data)
        same_heard['current_targets'].append(dict(node_id='updated-tree'))
        prep.offer(same_heard)
        wait_for(lambda: event_count(prep) == 2)
        assert helper.call_count == calls
        assert prep.events[-1]['status'] == 'same_failed_format_input'
        fresh = copy.deepcopy(data)
        fresh['heard_transcript'] += ' NEW_HEARD_QUALIFICATION.'
        prep.offer(fresh)
        wait_for(lambda: prep.peek() is not None)
        candidate = prep.peek()
        assert 'target_ids' not in candidate and candidate['audits'][-1]['accepted']
        assert candidate['framework'] == RECORDED[1]['framework']
        wait_for(lambda: prep.prepared_audio(candidate['text'], p.streaming_output_config) is not None)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff is not None
        assert handoff['audio']['text'] == candidate['text']
        assert handoff['candidate']['handoff_prechecked']
    finally:
        prep.close()
    assert len(drafts) == 3  # Initial draft + one repair, then one fresh-input retry.
    assert prep._calls <= prep.config.listening_prefix_max_calls


@pytest.mark.parametrize('call_cap', [3, 48])
def test_repeated_bad_format_has_only_one_new_input_retry_and_respects_call_cap(tmp_path, call_cap):
    p, data = inputs(tmp_path)
    p.streaming_output_config.listening_prefix_max_calls = call_cap
    helper = Mock(return_value=[json.dumps(RECORDED[0])])
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    try:
        for i in range(5):
            fresh = copy.deepcopy(data)
            fresh['heard_transcript'] += f' Heard batch {i}.'
            count = event_count(prep)
            prep.offer(fresh)
            if prep._calls < call_cap:
                wait_for(lambda: event_count(prep) == count + 1)
            elif prep._thread is not None:
                prep._thread.join(timeout=3)
        assert prep.peek() is None
        assert helper.call_count == min(call_cap, 4)
        if call_cap == 48:
            assert prep.events[-1]['status'] == 'format_retry_exhausted'
    finally:
        prep.close()
    calls = helper.call_count
    data['heard_transcript'] += ' After freeze.'
    prep.offer(data)
    assert helper.call_count == calls
