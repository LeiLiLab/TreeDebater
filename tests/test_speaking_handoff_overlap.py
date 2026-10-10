"""Event-gated audio/feedback overlap and latest completed body adoption."""
import copy
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from streaming.flat_speaking import SegmentRejected
from test_full_speech import audio as audio
from test_final_input_overlap import ready_handoff
from test_listening_prefix import TAIL, prepared_player, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')


def configure(p):
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True,
        listening_body_snapshot_delivery=True)
    p.high_quality_evidence_pool, p.used_evidence = [], set()


@pytest.mark.parametrize('outcome', ['success', 'tts_failure', 'binding_mismatch'])
def test_feedback_starts_before_pending_prefix_audio_and_failure_releases_workers(audio, tmp_path, outcome):
    p, history, _ = prepared_player(tmp_path)
    query, encoded = audio
    _, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    configure(p)
    audio_value = copy.deepcopy(handoff['audio'])
    pending_audio = Future()
    handoff.update(audio=None, audio_future=pending_audio)
    reviewed, first_audio = threading.Event(), threading.Event()
    revisions = []
    base = p.helper_client

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert not pending_audio.done(), 'Feedback waited for prefix synthesis'
            reviewed.set()
            return [json.dumps(dict(points=[], other_corrections=['Keep the qualification.']))]
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            assert pending_audio.done()
            return [TAIL]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda index, *args: first_audio.set() if index == 0 else None
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(p.rebuttal_generation, history, 60, time_control=True,
            listening_handoff=handoff, listening_recognized_input=lambda: history,
            listening_input_completion=lambda: history)
        try:
            assert reviewed.wait(3), 'Complete ASR feedback was blocked behind pending TTS'
            assert not first_audio.is_set()
            assert not revisions  # Duration-dependent revision still waits for real audio.
            if outcome == 'tts_failure':
                pending_audio.set_exception(RuntimeError('Prefix synthesis failed'))
            else:
                if outcome == 'binding_mismatch':
                    audio_value['text'] = 'Audio belongs to a different prefix.'
                pending_audio.set_result(audio_value)
            if outcome == 'success':
                assert TAIL in result.result(timeout=5)
                assert first_audio.is_set() and len(revisions) == 1
                saved = trace(tmp_path)
                assert saved['body_publication_mode'] == 'complete_asr_snapshot'
                assert saved['parallel_body_feedback']['start_seconds'] < saved['prefix_audio_transfer']['ready_seconds']
                assert saved['prefix_audio_transfer']['status'] == 'reused'
            else:
                error = RuntimeError if outcome == 'tts_failure' else SegmentRejected
                with pytest.raises(error, match='Prefix synthesis failed|does not match'):
                    result.result(timeout=5)
                assert not first_audio.is_set() and not revisions
                saved = trace(tmp_path)
                assert saved['status'] == 'failed'
                assert saved['prefix_audio_transfer']['status'] == 'failed'
                assert saved['chunks'] == []
            assert all(c.args[1] != handoff['candidate']['text'] for c in query.call_args_list)
        finally:
            if not pending_audio.done():
                pending_audio.set_result(audio_value)


@pytest.mark.parametrize('variant', ['newer', 'older', 'wrong_prefix', 'wrong_framework',
                                     'wrong_stage', 'wrong_turn', 'failed', 'pending', 'late'])
def test_body_is_selected_once_after_asr_without_waiting_for_unfinished_work(audio, tmp_path, variant):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    configure(p)
    pending = Future()
    handoff['body_future'] = pending
    original = copy.deepcopy(handoff['candidate']['body_preparation'])
    newer = copy.deepcopy(original)
    newer.update(draft='NEWEST_FINISHED_BODY contains the newly heard qualification.',
                 ready_monotonic=max(time.perf_counter(), original['ready_monotonic']) + 1)
    if variant == 'older':
        newer['ready_monotonic'] = original['ready_monotonic'] - 1
    elif variant.startswith('wrong_'):
        key = {'wrong_prefix': 'prefix_text', 'wrong_framework': 'framework',
               'wrong_stage': 'stage', 'wrong_turn': 'turn'}[variant]
        newer[key] = {} if key == 'framework' else 'mismatched'
    expected = newer['draft'] if variant == 'newer' else original['draft']
    observed = []
    base = p.helper_client

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            observed.append(prompt)
            assert expected in prompt
            if variant == 'late' and not pending.done():
                # Completion after task binding must never replace that task.
                pending.set_result(json.dumps(newer))
            return [json.dumps(dict(points=[], other_corrections=['Keep the qualification.']))]
        if kwargs.get('json_mode') is False:
            assert expected in prompt
            return [TAIL]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)

    def recognized():
        # This runs after the initial handoff check and before task binding.
        if not pending.done() and variant not in ('pending', 'late'):
            if variant == 'failed':
                pending.set_exception(RuntimeError('Speculative body failed'))
            else:
                pending.set_result(json.dumps(newer))
        return copy.deepcopy(history)

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(p.rebuttal_generation, history, 60, time_control=True,
            listening_handoff=handoff, listening_recognized_input=recognized,
            listening_input_completion=lambda: history)
        try:
            assert TAIL in result.result(timeout=5)
        finally:
            if not pending.done():
                pending.set_result(None)
    saved = trace(tmp_path)
    assert len(observed) == 1
    assert saved['body_publication_snapshot']['draft'] == expected
    assert saved['body_transfer']['adopted'] is (variant == 'newer')
    assert saved['body_preparation']['draft'] == expected
