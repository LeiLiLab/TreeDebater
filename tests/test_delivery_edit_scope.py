"""Offline publication tests. Stub verdicts test enforcement, not model accuracy."""
import json
import threading
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import Mock

import pytest

import tts_streaming as tts
from streaming.config import OutputConfig
from test_full_speech import audio as audio

pytestmark = pytest.mark.usefixtures('word_length_modes')
CASES = json.loads((Path(__file__).parent/'fixtures/v44_delivery_boundary_repetition.json').read_text())


def context(source='Original argument.', previous=('Opening.',), following='Next argument.'):
    return tts._ChunkRefineContext(Mock(), source, 6, .1, .1, list(previous), following,
        'echo', 1, 1, 'prestart', config=OutputConfig(adaptive_delivery=True,
        max_parallel_tts=8, allow_expansion=True))






@pytest.mark.parametrize('change_during', ['write', 'estimate'])
def test_context_change_cannot_mix_writer_and_synthesis(audio, monkeypatch, change_during):
    query, encoded = audio
    query.side_effect = lambda *a, **kw: encoded(12)
    ctx = context()
    captured = ctx.edit_scope()
    def change():
        ctx.update_context(['Opening.', 'Newly committed point.'], 'Changed next argument.')
    def revise(*args, **kwargs):
        assert args[3:5] == (list(captured.preceding), captured.following)
        assert kwargs['source_text'] == captured.source
        if change_during == 'write':
            change()
        return 'A reviewed expansion.'
    def estimate(text, **kwargs):
        if text != ctx.original_text and change_during == 'estimate':
            change()
        return 12
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    monkeypatch.setattr(tts, '_estimate_duration', estimate)
    try:
        tts._refine_worker(ctx, 'prestart')
        ctx.executor.shutdown(wait=True)
        assert query.call_count == 1, 'Stale edit reached synthesis'
        assert not ctx.rewrite_checks[0]['accepted']
        assert ctx.rewrite_checks[0]['edit_scope'] == captured.payload()
        assert captured.preceding == ('Opening.',)
    finally:
        ctx.executor.shutdown(wait=True)


@pytest.mark.parametrize('already_finished', [False, True])
def test_changed_context_expires_ready_or_pending_prestart_and_keeps_raw(audio, already_finished):
    query, encoded = audio
    started, release = threading.Event(), threading.Event()
    raw_audio, edit_audio = encoded(12), encoded(6)
    def synthesize(client, text, *a, **kw):
        if text == 'Speculative edit.':
            started.set()
            assert release.wait(3)
            return edit_audio
        return raw_audio
    query.side_effect = synthesize
    ctx = context()
    try:
        raw = ctx.add_candidate(ctx.original_text, 12, 'prestart', 0)
        raw.future.result(timeout=2)
        old_scope = ctx.edit_scope()
        edit = ctx.add_candidate('Speculative edit.', 6, 'prestart', 1, edit_scope=old_scope)
        assert started.wait(2)
        if already_finished:
            release.set()
            result = edit.future.result(timeout=2)
            ctx.try_adopt(edit, result)
            assert ctx.chosen_cand is edit
        ctx.update_context(['Opening.', 'Actual prior delivery.'], 'Next argument.')
        assert edit.obsolete and not raw.obsolete and ctx.chosen_cand is None
        assert not ctx.try_adopt(edit, edit_audio)
        # Fallback must not wait for or choose the obsolete six-second audio.
        chosen, _ = tts._pick_best_completed(ctx.candidates, 6)
        assert chosen is raw
        release.set()
        edit.future.result(timeout=2)
        assert not ctx.try_adopt(edit, edit_audio)
        assert ctx.add_candidate('Late old approval.', 6, 'prestart', 2, edit_scope=old_scope) is None
    finally:
        release.set()
        ctx.executor.shutdown(wait=True)


def test_identical_text_requires_fresh_approval_after_context_change(audio):
    query, encoded = audio
    query.side_effect = lambda *a, **kw: encoded(12)  # No candidate auto-adopts.
    ctx = context()
    try:
        old = ctx.add_candidate('Same words.', 12, 'prestart', 1, edit_scope=ctx.edit_scope())
        old.future.result(timeout=2)
        ctx.update_context(['Different preceding speech.'], 'Next argument.')
        new = ctx.add_candidate('Same words.', 12, 'normal', 1, edit_scope=ctx.edit_scope())
        new.future.result(timeout=2)
        assert old is not new and old.obsolete and not new.obsolete
        assert old.edit_scope.revision < new.edit_scope.revision
        assert new.audio_reused and old.future is new.future
        assert query.call_count == 1
        chosen, _ = tts._pick_best_completed(ctx.candidates, 6)
        assert chosen is new
    finally:
        ctx.executor.shutdown(wait=True)


def test_normal_worker_rebinds_prestart_without_extra_review_and_reuses_audio(audio, monkeypatch):
    (query, encoded) = audio
    query.side_effect = lambda client, text, *a, **kw: encoded(6 if text == 'Prepared edit.' else 12)
    monkeypatch.setattr(tts, '_estimate_duration', lambda *a, **kw: 12)
    writer = Mock(side_effect=AssertionError('Existing proposal should be reviewed before regenerating'))
    monkeypatch.setattr(tts, '_revise_to_n_words', writer)
    ctx = context()
    try:
        old_scope = ctx.edit_scope()
        old = ctx.add_candidate('Prepared edit.', 6, 'prestart', 1, edit_scope=old_scope)
        result = old.future.result(timeout=2)
        ctx.try_adopt(old, result)
        ctx.update_context(['Opening.', 'Actually delivered predecessor.'], 'Next argument.')
        tts._refine_worker(ctx, 'normal')
        ctx.executor.shutdown(wait=True)
        writer.assert_not_called()
        assert query.call_count == 2
        eligible = [c for c in ctx.candidates if not c.obsolete]
        (chosen, _) = tts._pick_best_completed(eligible, 6)
        assert chosen.text == 'Prepared edit.'
        assert chosen.future is old.future and chosen.edit_scope != old_scope
        assert chosen.audio_reused
    finally:
        ctx.executor.shutdown(wait=True)


def test_unchanged_context_preserves_prestart_approval_and_shared_audio(audio):
    query, encoded = audio
    query.side_effect = lambda *a, **kw: encoded(6)
    ctx = context()
    try:
        scope = ctx.edit_scope()
        edit = ctx.add_candidate('Approved edit.', 6, 'prestart', 1, edit_scope=scope)
        result = edit.future.result(timeout=2)
        ctx.update_context(list(scope.preceding), scope.following)
        assert ctx.edit_scope() is scope and not edit.obsolete
        assert ctx.chosen_cand is edit or ctx.try_adopt(edit, result)
        assert ctx.add_candidate('Approved edit.', 6, 'normal', 1, edit_scope=scope) is edit
        assert query.call_count == 1
    finally:
        ctx.executor.shutdown(wait=True)


def test_failed_and_cancelled_speculative_audio_do_not_break_fresh_worker():
    ctx = context()
    failed, cancelled = Future(), Future()
    failed.set_exception(RuntimeError('TTS failed'))
    cancelled.cancel()
    ctx.candidates = [tts._TtsCandidate(i, 'Edit.', 6, future,
        edit_scope=ctx.edit_scope(), obsolete=True) for i, future in enumerate((failed, cancelled))]
    try:
        assert ctx.reusable_proposal() is None
    finally:
        ctx.executor.shutdown(wait=True)
