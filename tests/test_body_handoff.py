"""Draft values survive feedback delays and handover without bypassing final review."""
import copy
import json
import threading
from concurrent.futures import Future
from dataclasses import replace
from types import MethodType
from unittest.mock import Mock

import pytest

from ouragents import TreeDebater
from streaming.listening_prefix import PrefixPreparation, material
from streaming.flat_speaking import SegmentRejected
from test_final_input_overlap import ready_handoff
from test_full_speech import audio as audio
from test_listening_overview import wait_for
from test_listening_prefix import PREFIX, TAIL, prepared_player, prefix_helper, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')






def test_inflight_draft_arrives_during_final_input_and_reuses_existing_review(audio, tmp_path):
    (p, history, _) = prepared_player(tmp_path)
    (_, encoded) = audio
    p.streaming_output_config = replace(p.streaming_output_config, listening_prefix_overlap_final_update=True, listening_prefix_pre_synthesize=True, listening_single_body_revision=True)
    (draft_started, release_draft, first_audio) = (threading.Event() for _ in range(3))
    base = prefix_helper(None)
    (reviewed, published) = ([], [])
    final_history = copy.deepcopy(history)
    final_history[-1]['content'] += ' FINAL_NEW_CONDITION.'

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:'):
            draft_started.set()
            assert release_draft.wait(3)
        if prompt.startswith('LISTENING WHOLE SPEECH FEEDBACK:'):
            assert 'FINAL_NEW_CONDITION' in prompt
            assert TAIL in prompt
            reviewed.append('feedback')
            return ['{"points":[],"other_corrections":["Preserve the final new condition."]}']
        if prompt.startswith('LISTENING FINAL BODY GATE:'):
            assert 'FINAL_NEW_CONDITION' in prompt and reviewed
            reviewed.append('gate')
        return base(prompt=prompt, **kwargs)
    p.helper_client = Mock(side_effect=helper)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    (p.high_quality_evidence_pool, p.used_evidence) = ([], set())
    p._get_response.side_effect = AssertionError('Completed transferred draft must not be regenerated')
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config, lambda *args: encoded(2))
    p._listening_prefix = prep
    from test_listening_body_cadence import offer_and_drain, add_words
    data = material(p, p.status, history)
    offer_and_drain(prep, data)
    original_body = copy.deepcopy(prep._body_latest)
    add_words(data, 100)
    prep.offer(data)
    try:
        assert draft_started.wait(3)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff and handoff['candidate']['body_preparation']['draft'] == TAIL
        future = handoff['body_future']
        assert future is not None and (not future.done())

        def complete_input():
            assert first_audio.wait(3), 'In-flight draft blocked first audio'
            release_draft.set()
            assert json.loads(future.result(timeout=3))['feedback'] is None
            p.planner.chunks = [final_history[-1]['content']]
            return final_history

        def emit(index, path, text, duration):
            published.append(text)
            if index == 0:
                assert not reviewed
                first_audio.set()
            else:
                assert reviewed == ['feedback'], 'Transferred draft bypassed complete-input feedback'
        p.tts_chunk_callback = emit
        call = lambda : p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff, listening_input_completion=complete_input, listening_recognized_input=lambda: final_history)
        assert TAIL in call()
        assert published == [PREFIX, TAIL]
        saved = trace(tmp_path)
        assert saved['status'] == 'completed'
        assert 'review_warnings' not in saved
        assert 'body_reviews' not in saved
        assert 'final_body_repairs' not in saved
        assert saved['body_transfer']['adopted']
        assert saved['body_preparation_source'] == 'inflight_transfer'
        assert saved['body_preparation']['feedback_status'] == 'deferred_to_final_input'
        assert saved['body_preparation']['draft'] == TAIL
        assert prep._body_latest == original_body, 'Late result mutated the frozen preparation'
        p._get_response.assert_not_called()
        p._length_adjust.assert_called()
        assert not any((c.kwargs['prompt'].startswith('LISTENING BODY FEEDBACK:') for c in p.helper_client.call_args_list))
    finally:
        release_draft.set()
        prep.close()


@pytest.mark.parametrize('mode', ['still_running', 'failed', 'bad_format', 'wrong_prefix', 'wrong_framework', 'wrong_turn'])
def test_unavailable_or_mismatched_transfer_falls_back_without_wait(audio, tmp_path, mode):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    prep, handoff = ready_handoff(p, history, encoded)
    handoff['candidate'].pop('body_preparation', None)
    future = Future()
    handoff['body_future'] = future
    if mode == 'failed':
        future.set_result(None)
    elif mode == 'bad_format':
        future.set_result('invalid JSON')
    elif mode != 'still_running':
        value = dict(draft=TAIL, prefix_text=PREFIX, stage=p.status, turn=p.planner.turn,
                     framework=copy.deepcopy(handoff['candidate']['framework']), feedback=None)
        if mode == 'wrong_prefix':
            value['prefix_text'] = 'Different already-spoken opening.'
        elif mode == 'wrong_framework':
            value['framework']['response_axes'] = ['Different promised issue']
        else:
            value['turn'] = 'another:turn'
        future.set_result(json.dumps(value))
    try:
        p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=lambda: history, listening_recognized_input=lambda: history)
        p._get_response.assert_called_once()
        assert trace(tmp_path)['body_preparation_source'] == 'cold_draft'
        assert not trace(tmp_path)['body_transfer']['adopted']
        if mode == 'still_running':
            assert not future.done()
    finally:
        prep.close()


def test_feedback_failure_preserves_valid_draft_but_invalid_draft_is_not_saved(tmp_path):
    p, history, node = prepared_player(tmp_path)
    base = prefix_helper(node)
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY FEEDBACK:'):
            raise RuntimeError('Optional preparatory feedback failed')
        return base(prompt=prompt, **kwargs)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    try:
        prep.offer(material(p, p.status, history))
        wait_for(lambda: any(e['status'] == 'body_saved_unreviewed' for e in prep.events))
        assert prep.freeze()['body_preparation']['draft'] == TAIL
    finally:
        prep.close()
    def invalid(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:'):
            return ['{"draft":""}']
        return base(prompt=prompt, **kwargs)
    prep = PrefixPreparation(p.planner.turn, invalid, p.streaming_output_config)
    try:
        from test_listening_body_cadence import offer_and_drain, add_words
        data = material(p, p.status, history)
        offer_and_drain(prep, data)
        add_words(data, 100)
        prep.offer(data)
        wait_for(lambda: any(e['status'] == 'body_rejected' for e in prep.events))
        candidate = prep.freeze()
        assert candidate['body_preparation']['draft'] == TAIL
        assert prep.pending_body(candidate).result(timeout=1) is None
    finally:
        prep.close()
