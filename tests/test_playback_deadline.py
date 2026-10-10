"""Offline regressions for the v23 opening/body gap and wasted rewrites."""
import csv
import json
import time
from unittest.mock import Mock

import pytest

from streaming.config import OutputConfig
from test_full_speech import audio as audio
import tts_streaming as tts

pytestmark = pytest.mark.usefixtures('word_length_modes')


def test_slow_tail_and_edit_publish_existing_audio_before_opening_ends(audio, tmp_path, monkeypatch):
    query, encoded = audio
    opening_audio, body_audio = encoded(2), encoded(.1)
    query.side_effect = lambda client, text, *a, **kw: opening_audio if text == 'Opening.' else body_audio
    def tail():
        time.sleep(1.2)  # Already uses most of the opening playback window.
        return 'Body argument.'
    def revise(*args, **kwargs):
        time.sleep(.9)  # Optional edit cannot finish before the remaining deadline.
        return 'A much longer revised argument.'
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    arrivals = []
    tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'), 10,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False, allow_expansion=True,
            max_refinements=1, early_max_refinements=1, speed_adjust_min=1, speed_adjust_max=1),
        tail_supplier=tail, on_chunk=lambda i, p, text, d: arrivals.append((time.monotonic(), text)))
    assert arrivals[1][0] < arrivals[0][0] + 2
    assert arrivals[1][1] == 'Body argument.'
    assert query.call_count == 2  # Late optional rewrite never gets synthesized.


@pytest.mark.parametrize('body_wait', [4.89, 20.0])
def test_tail_wait_is_deducted_and_queued_audio_extends_deadline(audio, tmp_path, monkeypatch, body_wait):
    clock = [100.]
    monkeypatch.setattr(tts, '_now', lambda: clock[0])
    query, encoded = audio
    query.side_effect = lambda client, text, *a, **kw: encoded(17.042 if text == 'Opening.' else 5)
    rewrite = Mock(side_effect=AssertionError('No length edits enabled'))
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    def tail():
        clock[0] += body_wait
        return 'First body paragraph.\n\nSecond body paragraph.'
    heard = []
    tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'), 40,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False, min_chunk_words=1,
            max_refinements=0, early_max_refinements=0, speed_adjust_min=1, speed_adjust_max=1),
        tail_supplier=tail, on_chunk=lambda i, p, text, d: heard.append(text))
    rows = list(csv.DictReader((tmp_path/'speech_chunks/chunk_profile.csv').open()))
    remaining = max(0, 17.042 - body_wait)
    assert float(rows[1]['time_budget_s']) == pytest.approx(remaining)
    assert float(rows[2]['time_budget_s']) == pytest.approx(remaining + 5)
    assert heard == ['Opening.', 'First body paragraph.', 'Second body paragraph.']
    assert query.call_count == 3  # Even an exhausted opening buffer still permits mandatory raw TTS.


@pytest.mark.parametrize('rewrite_result', ['Body argument.', '  Body   argument.  '])
def test_unchanged_rewrite_reuses_audio_and_publishes_without_waiting(audio, tmp_path, monkeypatch, rewrite_result):
    query, encoded = audio
    query.side_effect = lambda client, text, *a, **kw: encoded(17 if text == 'Opening.' else 1)
    rewrite = Mock(return_value=rewrite_result)
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    arrivals = []
    tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'), 40,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False, allow_expansion=True,
            max_refinements=3, early_max_refinements=3,
            speed_adjust_min=1, speed_adjust_max=1), tail_supplier=lambda: 'Body argument.',
        on_chunk=lambda *args: arrivals.append(time.monotonic()))
    assert arrivals[1] - arrivals[0] < 2  # Old code waited about15s and synthesized identical text repeatedly.
    assert query.call_count == 2
    rewrite.assert_called_once()
    audit = json.loads((tmp_path/'speech_chunks/rewrite_audit.json').read_text())
    assert audit[0]['checks'][0]['reason'].startswith('Unchanged')
    rows = list(csv.DictReader((tmp_path/'speech_chunks/chunk_profile.csv').open()))
    assert rows[1]['timed_out'] == 'False'


def test_edit_finishing_after_deadline_does_not_trigger_review_or_tts(audio, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(tts, '_now', lambda: clock[0])
    query, encoded = audio
    query.side_effect = lambda *a, **kw: encoded(1)
    def revise(*args, **kwargs):
        clock[0] = 11.
        return 'A much longer revised argument.'
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    ctx = tts._ChunkRefineContext(Mock(), 'Body argument.', 10, 1, 1, [], '', 'echo', 3, 1, '',
        config=OutputConfig(adaptive_delivery=True, allow_expansion=True))
    ctx.refinement_deadline = 10.
    try:
        tts._refine_worker(ctx, 'normal')
        assert query.call_count == 1
        assert not ctx.rewrite_checks[0]['accepted']
    finally:
        ctx.executor.shutdown(wait=True)
