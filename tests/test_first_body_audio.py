"""Offline event ordering and exact reuse across the final-input handover."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading
import time
from types import MethodType
from unittest.mock import Mock

import pytest

from ouragents import TreeDebater
from streaming.body_audio import FirstBodyAudio
from streaming.body_task import BodyTask
from test_final_input_overlap import ready_handoff
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, prepared_player, trace, prefix_helper
import tts_streaming as tts

pytestmark = pytest.mark.usefixtures('word_length_modes')
CHANGED_TAIL = 'The corrected transcript requires safeguards before any funding commitment.'


def player_with_handoff(tmp_path, encoded, *, stage=None):
    player, history, _ = prepared_player(tmp_path)
    if stage is not None:
        player.status = stage
        player.planner.turn = f'{player.oppo_side}:{stage}'
        history[-1]['stage'] = stage
    _, handoff = ready_handoff(player, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback=None,
        prefix_text=PREFIX, stage=player.status, turn=player.planner.turn)
    player.streaming_output_config = replace(player.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    player._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, player)
    player._length_adjust = MethodType(TreeDebater._length_adjust, player)
    player.high_quality_evidence_pool, player.used_evidence = [], set()
    return player, history, handoff


@pytest.mark.parametrize('needs_revision', [False, True])
def test_immediate_closing_feedback_waits_for_actual_prefix_duration(audio, tmp_path, monkeypatch, needs_revision):
    """Replay v51: closing feedback finishes before the first audio callback."""
    from test_listening_overview import wait_for
    query, encoded = audio
    player, history, handoff = player_with_handoff(tmp_path, encoded, stage='closing')
    handoff['candidate']['body_preparation']['needs_fit'] = needs_revision
    handoff['body_future'] = None  # Keep this test's exact preparatory length decision.
    handoff['audio']['tts_out']['audio_seconds'] = 2.75  # Decoded audio is exactly 2s.
    monkeypatch.setattr(tts, 'estimate_statement_seconds', lambda *args, **kwargs: 58.)
    first, audio_started, committed = [threading.Event() for _ in range(3)]
    revisions, delivered = [], []
    base = player.helper_client
    convert = tts.convert_text_to_speech_streaming

    def delayed_prefix(*args, **kwargs):
        wait_for(lambda: trace(tmp_path).get('parallel_body_feedback', {}).get('end_seconds') is not None)
        assert not delivered
        return convert(*args, **kwargs)

    def helper(*, prompt, **kwargs):
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            assert not committed.is_set()
            return [TAIL]
        assert 'LISTENING WHOLE SPEECH FEEDBACK:' not in prompt
        return base(prompt=prompt, **kwargs)

    def synthesize(client, text, **kwargs):
        assert text == TAIL and not committed.is_set()
        audio_started.set()
        return encoded(.1)

    def emit(index, path, text, seconds):
        delivered.append(text)
        if index == 0:
            assert seconds == 2.
            first.set()
        else:
            assert committed.is_set()

    def complete():
        assert first.wait(3)
        assert audio_started.wait(3), 'Immediate closing feedback skipped early body work'
        committed.set()
        return copy.deepcopy(history)

    monkeypatch.setattr(tts, 'convert_text_to_speech_streaming', delayed_prefix)
    player.helper_client = Mock(side_effect=helper)
    player.tts_chunk_callback = emit
    query.side_effect = synthesize
    assert player.closing_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete,
        listening_recognized_input=lambda: copy.deepcopy(history)) == PREFIX + '\n\n' + TAIL
    saved = trace(tmp_path)
    assert saved['parallel_body_feedback']['end_seconds'] < saved['chunks'][0]['ready_seconds']
    assert saved['prefix_duration_wait']['audio_seconds'] == 2.
    assert saved['first_body_audio']['start_seconds'] < saved['final_input_ready_seconds']
    assert saved['first_body_audio']['reused'] and query.call_count == 1
    assert len(revisions) == int(needs_revision)
    if needs_revision:
        assert saved['parallel_body_revision']['reused']
    assert delivered == [PREFIX, TAIL]


def test_prefix_failure_releases_early_duration_waiter(audio, tmp_path, monkeypatch):
    from test_listening_overview import wait_for
    query, encoded = audio
    player, history, handoff = player_with_handoff(tmp_path, encoded, stage='closing')
    handoff['candidate']['body_preparation']['needs_fit'] = True
    handoff['body_future'] = None
    player.tts_chunk_callback = Mock()

    def fail_before_prefix(*args, **kwargs):
        wait_for(lambda: trace(tmp_path).get('parallel_body_feedback', {}).get('end_seconds') is not None)
        raise RuntimeError('First audio decoding failed')

    monkeypatch.setattr(tts, 'convert_text_to_speech_streaming', fail_before_prefix)
    with pytest.raises(RuntimeError, match='First audio decoding failed'):
        player.closing_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=lambda: copy.deepcopy(history),
            listening_recognized_input=lambda: copy.deepcopy(history))
    saved = trace(tmp_path)
    assert saved['status'] == 'failed' and saved['chunks'] == []
    assert 'First audio decoding failed' in saved['parallel_body_feedback']['worker_error']
    player.tts_chunk_callback.assert_not_called()
    query.assert_not_called()


@pytest.mark.parametrize('case', ['match', 'changed', 'failed', 'no_revision'])
def test_synthesis_precedes_handover_but_publication_waits(audio, tmp_path, monkeypatch, case):
    query, encoded = audio
    player, history, handoff = player_with_handoff(tmp_path, encoded)
    first, audio_started, handover_finished = [threading.Event() for _ in range(3)]
    delivered, synthesized = [], []
    final = copy.deepcopy(history)
    final[-1]['content'] += ' FINAL_ASR_QUALIFICATION.'
    if case == 'no_revision':
        monkeypatch.setattr(BodyTask, 'needs_revision', lambda *args, **kwargs: False)

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            return prefix_helper(None)(prompt=prompt, **kwargs)
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert first.wait(3)
            return ['No changes' if case == 'no_revision' else 'Keep the final qualification.']
        assert kwargs.get('json_mode') is False
        tail = CHANGED_TAIL if 'CORRECTED_TRANSCRIPT' in kwargs['history_messages'][-1]['content'] else TAIL
        # Exercise exactly the same envelope/cleanup as committed revision.
        return [json.dumps({'speech': 'Revised Statement:\n' + tail})]

    def synthesize(client, text, *args, **kwargs):
        synthesized.append(text)
        if len(synthesized) == 1:
            assert text == TAIL and not handover_finished.is_set()
            audio_started.set()
            assert handover_finished.wait(3), 'Speculative synthesis blocked input handover'
            if case == 'failed':
                raise RuntimeError('Optional TTS failed')
        else:
            assert handover_finished.is_set()
        return encoded(.1)

    def emit(index, path, text, seconds):
        delivered.append(text)
        if index == 0:
            first.set()
        else:
            assert handover_finished.is_set(), 'Unconfirmed body escaped to playback'

    def complete():
        assert audio_started.wait(3), 'Body TTS waited for final tree/plan handover'
        assert delivered == [PREFIX]
        resolved = copy.deepcopy(final)
        if case == 'changed':
            resolved[-1]['content'] += ' CORRECTED_TRANSCRIPT.'
        player.planner.chunks = [resolved[-1]['content']]
        handover_finished.set()
        return resolved

    player.helper_client = Mock(side_effect=helper)
    player.tts_chunk_callback = emit
    query.side_effect = synthesize
    try:
        result = player.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=lambda: copy.deepcopy(final))
    finally:
        handover_finished.set()
    expected = CHANGED_TAIL if case == 'changed' else TAIL
    assert result == PREFIX + '\n\n' + expected
    assert delivered == [PREFIX, expected]
    assert synthesized == ([TAIL, expected] if case in ('changed', 'failed') else [TAIL])
    saved = trace(tmp_path)
    prep = saved['first_body_audio']
    assert prep['start_seconds'] < saved['final_input_ready_seconds']
    assert prep['matched'] is (case != 'changed')
    assert prep['reused'] is (case in ('match', 'no_revision'))
    assert prep['status'] == ('failed' if case == 'failed' else 'ready')


@pytest.mark.parametrize('change', ['text', 'voice', 'model'])
def test_audio_reuse_requires_exact_final_chunk_and_voice(audio, change):
    query, _ = audio
    config = tts.OutputConfig()
    prepared = FirstBodyAudio(config, time.perf_counter())
    try:
        prepared.prepare(TAIL, 58)
        original = [TAIL, config.voice, config.model]
        assert prepared.match(*original).result(timeout=3)
        changed = list(original)
        changed[['text', 'voice', 'model'].index(change)] += ' changed'
        assert prepared.match(*changed) is None
        assert query.call_count == 1
    finally:
        prepared.close()


def test_inflight_matching_audio_is_shared_without_a_second_request(audio):
    query, encoded = audio
    started, release = threading.Event(), threading.Event()
    def synthesize(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return encoded(.1)
    query.side_effect = synthesize
    config = tts.OutputConfig()
    prepared = FirstBodyAudio(config, time.perf_counter())
    ctx = tts._ChunkRefineContext(Mock(), TAIL, 58, 1, 1, [PREFIX], '',
        config.voice, 0, 1, '', config=config)
    try:
        prepared.prepare(TAIL, 58)
        assert started.wait(3)
        ctx.prepared_audio = prepared
        candidate = ctx.add_candidate(TAIL, 5, 'normal', 0)
        assert not candidate.future.done()
        release.set()
        assert candidate.future.result(timeout=3)['audio_seconds'] == .1
        assert candidate.audio_reused and query.call_count == 1
    finally:
        release.set()
        ctx.executor.shutdown(wait=True)
        prepared.close()


def test_failed_handover_never_publishes_speculative_body_and_settles_request(audio, tmp_path):
    query, encoded = audio
    player, history, handoff = player_with_handoff(tmp_path, encoded)
    first, started, failed, release, finished = [threading.Event() for _ in range(5)]
    delivered = []
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            return prefix_helper(None)(prompt=prompt, **kwargs)
        assert first.wait(3)
        return ['Keep the qualification.' if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt else TAIL]
    def synthesize(*args, **kwargs):
        started.set()
        try:
            assert release.wait(3)
            return encoded(.1)
        finally:
            finished.set()
    def complete():
        assert started.wait(3)
        failed.set()
        raise RuntimeError('Final handover failed')
    def emit(index, path, text, seconds):
        delivered.append(text)
        first.set()
    query.side_effect = synthesize
    player.helper_client = Mock(side_effect=helper)
    player.tts_chunk_callback = emit
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(player.rebuttal_generation, history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=lambda: copy.deepcopy(history))
        try:
            assert failed.wait(3)
            assert delivered == [PREFIX] and not finished.is_set() and not job.done()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='Final handover failed'):
            job.result(timeout=3)
    assert finished.is_set() and query.call_count == 1
    assert trace(tmp_path)['committed_text'] == PREFIX
