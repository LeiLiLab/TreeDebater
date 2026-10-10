"""Offline synchronization tests: overlap, verified candidates and deadline fallback."""
import threading
import time
from unittest.mock import Mock

import pytest

from streaming.config import OutputConfig
from test_full_speech import audio as audio
import tts_streaming as tts

pytestmark = pytest.mark.usefixtures('word_length_modes')


def context(*, workers=8):
    return tts._ChunkRefineContext(Mock(), 'Original body.', 6, .1, .1, [], '', 'echo', 1, 1, '',
        config=OutputConfig(adaptive_delivery=True, allow_expansion=True,
            max_parallel_tts=workers))


def test_verified_edit_synthesizes_and_publishes_while_raw_audio_is_pending(audio, tmp_path, monkeypatch):
    query, encoded = audio
    raw_started, raw_finished, release_raw = [threading.Event() for _ in range(3)]
    prefix, raw, edited = 'Opening.', 'Original body.', 'Reviewed shorter body.'
    audio_values = {prefix: encoded(2), raw: encoded(12), edited: encoded(6)}
    def synthesize(client, text, *args, **kwargs):
        if text == raw:
            raw_started.set()
            try:
                assert release_raw.wait(3), 'Edited audio could not publish while raw synthesis was pending'
            finally:
                raw_finished.set()
        elif text == edited:
            assert not raw_finished.is_set()
        return audio_values[text]
    query.side_effect = synthesize
    monkeypatch.setattr(tts, '_estimate_duration', lambda text, **kw: 12 if text == raw else 6)
    def revise(*args, **kwargs):
        assert raw_started.wait(2) and not raw_finished.is_set()
        return edited
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    heard = []
    def emit(index, path, text, seconds):
        heard.append(text)
        if index == 1:
            assert not raw_finished.is_set()
            release_raw.set()
    try:
        tts.convert_text_to_speech_streaming(prefix, str(tmp_path/'speech.mp3'), 8,
            config=OutputConfig(adaptive_delivery=True, allow_expansion=True,
                normalize_seams=False, first_chunk_seconds=2, max_refinements=1, early_max_refinements=1,
                max_parallel_tts=8, speed_adjust_min=1, speed_adjust_max=1),
            tail_supplier=lambda: raw, on_chunk=emit)
        assert heard == [prefix, edited] and query.call_count == 3
    finally:
        release_raw.set()


def test_empty_edit_never_enters_parallel_tts(audio, monkeypatch):
    query, encoded = audio
    started, release = threading.Event(), threading.Event()
    def synthesize(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return encoded(12)
    query.side_effect = synthesize
    monkeypatch.setattr(tts, '_estimate_duration', lambda *a, **kw: 12)
    def revise(*args, **kwargs):
        assert started.wait(2)
        return ''
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    ctx = context()
    try:
        tts._refine_worker(ctx, 'normal')
        assert len(ctx.candidates) == query.call_count == 1
        assert not ctx.rewrite_checks[0]['accepted']
    finally:
        release.set()
        ctx.executor.shutdown(wait=True)


def test_deadline_publishes_raw_without_waiting_for_slow_edited_tts(audio, tmp_path, monkeypatch):
    query, encoded = audio
    release, edit_started, edit_finished = threading.Event(), threading.Event(), threading.Event()
    values = {'Opening.': encoded(2), 'Original body.': encoded(12), 'Reviewed edit.': encoded(6)}
    def synthesize(client, text, *args, **kwargs):
        if text == 'Reviewed edit.':
            edit_started.set()
            assert release.wait(3), 'Late optimization blocked original audio publication'
            edit_finished.set()
        return values[text]
    query.side_effect = synthesize
    monkeypatch.setattr(tts, '_estimate_duration', lambda text, **kw: 12 if text == 'Original body.' else 6)
    monkeypatch.setattr(tts, '_revise_to_n_words', Mock(return_value='Reviewed edit.'))
    heard = []
    def emit(index, path, text, duration):
        heard.append(text)
        if index == 1:
            assert edit_started.is_set() and not edit_finished.is_set()
            release.set()
    try:
        tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'), 8,
            config=OutputConfig(adaptive_delivery=True, normalize_seams=False,
                max_parallel_tts=8, early_max_refinements=1, max_refinements=1,
                speed_adjust_min=1, speed_adjust_max=1),
            tail_supplier=lambda: 'Original body.', on_chunk=emit)
        assert heard == ['Opening.', 'Original body.']
    finally:
        release.set()


def test_shared_raw_audio_is_not_synthesized_twice(audio):
    query, _ = audio
    ctx = context()
    try:
        first = ctx.add_candidate('Original body.', 12, 'prestart', 0)
        second = ctx.add_candidate('Original  body.', 12, 'normal', 0)
        assert first is second
        first.future.result(timeout=2)
        assert query.call_count == 1
    finally:
        ctx.executor.shutdown(wait=True)


def test_finished_writer_still_waits_for_pending_verified_audio(audio):
    query, encoded = audio
    release = threading.Event()
    raw_audio, edited_audio = encoded(12), encoded(6)
    def synthesize(client, text, *args, **kwargs):
        if text == 'Reviewed edit.':
            assert release.wait(2)
            return edited_audio
        return raw_audio
    query.side_effect = synthesize
    ctx = context()
    ctx.refinement_deadline = time.perf_counter() + 3
    finished = threading.Event()
    def wait():
        tts._wait_for_refinement(ctx, ctx.refinement_deadline)
        finished.set()
    try:
        ctx.add_candidate('Original body.', 12, 'normal', 0).future.result(timeout=2)
        edit = ctx.add_candidate('Reviewed edit.', 6, 'normal', 1)
        waiter = threading.Thread(target=wait)
        waiter.start()
        assert not finished.wait(.05), 'Pending candidate was ignored after writer finished'
        release.set()
        waiter.join(2)
        assert finished.is_set() and ctx.chosen_cand is edit
    finally:
        release.set()
        ctx.executor.shutdown(wait=True)


def test_late_parallel_edit_cannot_replace_raw_audio(audio, monkeypatch):
    query, encoded = audio
    clock = [0.]
    monkeypatch.setattr(tts, '_now', lambda: clock[0])
    release = threading.Event()
    raw_audio, edited_audio = encoded(12), encoded(6)
    def synthesize(client, text, *args, **kwargs):
        if text == 'Reviewed edit.':
            assert release.wait(2)
            return edited_audio
        return raw_audio
    query.side_effect = synthesize
    ctx = context()
    ctx.refinement_deadline = 10.
    try:
        raw = ctx.add_candidate('Original body.', 12, 'normal', 0)
        raw.future.result(timeout=2)
        edit = ctx.add_candidate('Reviewed edit.', 6, 'normal', 1)
        clock[0] = 11.
        release.set()
        try:
            edit.future.result(timeout=2)
        except Exception:
            pass  # A still-queued edit may be cancelled before synthesis starts.
        assert ctx.chosen_cand is None
        chosen, _ = tts._pick_best_completed(ctx.candidates, 6, deadline=10.)
        assert chosen is raw
    finally:
        release.set()
        ctx.executor.shutdown(wait=True)
