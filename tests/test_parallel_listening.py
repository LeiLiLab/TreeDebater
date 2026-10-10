"""Played audio recognition overlaps ordered analysis; no provider calls."""
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import pytest

from test_listening_motion_live import live, fake_turn


def test_next_asr_runs_during_previous_analysis_and_handover_drains_both(tmp_path, monkeypatch):
    speaker, listener, asr, _ = fake_turn(tmp_path, monkeypatch)
    analysis_started = threading.Event()
    second_asr_done = threading.Event()
    second_analysis_started = threading.Event()
    release_analysis = threading.Event()
    endpoint, transcript = Future(), Future()
    order = []
    provider_ready = []
    original = asr.audio.transcriptions.create

    def transcribe(**kwargs):
        result = original(**kwargs)
        if result.text == 'Heard slice2.':
            assert analysis_started.is_set()
            provider_ready.append(time.perf_counter())
            second_asr_done.set()
        return result

    def observe(text, side, stage):
        order.append(text)
        if len(order) == 1:
            analysis_started.set()
            assert second_asr_done.wait(3), 'Tree analysis blocked next ASR'
        else:
            second_analysis_started.set()
            assert release_analysis.wait(3)

    asr.audio.transcriptions.create = transcribe
    listener.observe_opponent.side_effect = observe
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(live.play_turn, 0, speaker, listener, [], 240, None,
            SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter(),
            playback_complete=endpoint, transcript_ready=transcript)
        try:
            assert transcript.result(timeout=3) == 'Heard slice1. Heard slice2.'
            endpoint.result(timeout=3)
            assert second_analysis_started.wait(3)
            assert not job.done(), 'Mutable state handed over before final analysis'
        finally:
            release_analysis.set()
        row = job.result(timeout=3)
    assert order == ['Heard slice1.', 'Heard slice2.']
    first, second = row['heard']
    # Releasing the fake provider may let analysis finish before receive() gets
    # CPU time to stamp asr_ready. Measure overlap before releasing that latch.
    assert first['analysis_start_monotonic'] < provider_ready[0] < first['analysis_end_monotonic']
    assert first['analysis_end_monotonic'] <= second['analysis_start_monotonic']
    assert row['listener_drained_monotonic'] >= second['analysis_end_monotonic']
    assert row['full_asr_ready_monotonic'] < second['analysis_end_monotonic']


def test_analysis_failure_blocks_inflight_asr_from_scheduling_more_analysis(tmp_path, monkeypatch):
    speaker, listener, asr, _ = fake_turn(tmp_path, monkeypatch)
    meter = live.Meter()
    second_asr_started = threading.Event()
    original = asr.audio.transcriptions.create
    transcript = Future()

    def transcribe(**kwargs):
        result = original(**kwargs)
        if result.text == 'Heard slice2.':
            second_asr_started.set()
            assert meter.stopped.wait(3)
        return result

    def observe(*args):
        assert second_asr_started.wait(3)
        raise RuntimeError('Tree analysis failed')

    asr.audio.transcriptions.create = transcribe
    listener.observe_opponent.side_effect = observe
    with pytest.raises((RuntimeError, live.BudgetExceeded)):
        live.play_turn(0, speaker, listener, [], 240, None,
            SimpleNamespace(artifact={'blocked_dispatches': []}), asr, meter, transcript_ready=transcript)
    with pytest.raises(RuntimeError, match='Tree analysis failed'):
        transcript.result(timeout=1)
    listener.observe_opponent.assert_called_once()
    assert meter.stopped.is_set()
