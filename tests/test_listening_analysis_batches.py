"""ASR cadence and ordered tree/planning batches are independent, offline."""
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from types import SimpleNamespace

import pytest

from test_listening_motion_live import live, fake_turn


def words(label, count):
    return ' '.join(f'{label}{i}' for i in range(count)) + '.'


def run(speaker, listener, asr, **kwargs):
    return live.play_turn(0, speaker, listener, [], 240, None,
        SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter(), **kwargs)


def test_default_tree_analysis_threshold_is_100_words():
    assert live.INPUT_CONFIG.min_text_words == 100
    assert live.INPUT_CONFIG.max_text_wait_seconds == 60


@pytest.mark.parametrize('sizes,expected,final_flush', [
    ([40, 59, 1, 30], [[0, 1, 2], [3]], [False, True]),
    ([60, 60, 20], [[0, 1], [2]], [False, True]),
    ([60, 40], [[0, 1]], [False]),
    ([10, 20], [[0, 1]], [True]),
])
def test_accumulate_whole_chunks_and_drain_final_partial_once(tmp_path, monkeypatch, sizes, expected, final_flush):
    parts = [words(str(i) + '-', size) for i, size in enumerate(sizes)]
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch,
        min_text_words=100, transcripts=parts)
    original = asr.audio.transcriptions.create

    def transcribe(**kwargs):
        # No analysis is scheduled while fewer than100 words have arrived.
        if sum(sizes[:len(calls)]) < 100:
            listener.observe_opponent.assert_not_called()
        return original(**kwargs)

    asr.audio.transcriptions.create = transcribe
    row = run(speaker, listener, asr)
    assert len(calls) == len(parts)
    assert row['listener_transcript'] == ' '.join(parts)
    batches = row['analysis_batches']
    assert [b['sequences'] for b in batches] == expected
    assert [b['final_flush'] for b in batches] == final_flush
    assert [c.args[0] for c in listener.observe_opponent.call_args_list] == [
        ' '.join(parts[i] for i in group) for group in expected]
    assert ' '.join(b['text'] for b in batches) == row['listener_transcript']
    for batch in batches:
        assert batch['analysis_start_monotonic'] >= max(row['heard'][i]['asr_ready_monotonic'] for i in batch['sequences'])
        for i in batch['sequences']:
            assert row['heard'][i]['analysis_batch'] == batch['index']
            assert row['heard'][i]['analysis_end_monotonic'] == batch['analysis_end_monotonic']
    assert row['listener_drained_monotonic'] >= batches[-1]['analysis_end_monotonic']


def test_complete_asr_ready_while_final_batched_analysis_is_still_blocked(tmp_path, monkeypatch):
    parts = [words(str(i) + '-', n) for i, n in enumerate([60, 40, 10, 10])]
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch,
        min_text_words=100, transcripts=parts)
    first_started, third_recognized = threading.Event(), threading.Event()
    final_started, release_final = threading.Event(), threading.Event()
    endpoint, transcript = Future(), Future()
    original = asr.audio.transcriptions.create
    observed = []

    def transcribe(**kwargs):
        result = original(**kwargs)
        if len(calls) == 3:
            assert first_started.wait(3)
            third_recognized.set()
        return result

    def observe(text, *args):
        observed.append(text)
        if len(observed) == 1:
            first_started.set()
            assert third_recognized.wait(3), 'Analysis blocked later ASR'
        else:
            final_started.set()
            assert release_final.wait(3)

    asr.audio.transcriptions.create = transcribe
    listener.observe_opponent.side_effect = observe
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(run, speaker, listener, asr,
            playback_complete=endpoint, transcript_ready=transcript)
        try:
            assert transcript.result(timeout=3) == ' '.join(parts)
            endpoint.result(timeout=3)
            assert final_started.wait(3)
            assert not job.done(), 'Mutable handover must wait for the final partial batch'
        finally:
            release_final.set()
        row = job.result(timeout=3)
    assert observed == [' '.join(parts[:2]), ' '.join(parts[2:])]
    first, final = row['analysis_batches']
    assert first['analysis_start_monotonic'] < row['heard'][2]['asr_ready_monotonic'] < first['analysis_end_monotonic']
    assert first['analysis_end_monotonic'] <= final['analysis_start_monotonic']
    assert row['full_asr_ready_monotonic'] < final['analysis_end_monotonic']


def test_failure_in_final_partial_analysis_fails_the_turn(tmp_path, monkeypatch):
    parts = [words('a', 15), words('b', 15)]
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch,
        min_text_words=100, transcripts=parts)
    listener.observe_opponent.side_effect = RuntimeError('Final analysis failed')
    with pytest.raises(RuntimeError, match='Final analysis failed'):
        run(speaker, listener, asr)
    assert len(calls) == 2
    listener.observe_opponent.assert_called_once_with(' '.join(parts), 'for', 'opening')


def test_short_text_timeout_runs_while_next_asr_is_in_flight(tmp_path, monkeypatch):
    from dataclasses import replace
    parts = [words('first', 20), words('second', 25)]
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch,
        min_text_words=100, transcripts=parts)
    monkeypatch.setattr(live, 'INPUT_CONFIG', replace(live.INPUT_CONFIG, max_text_wait_seconds=.25))
    original = asr.audio.transcriptions.create
    analyzed = threading.Event()
    listener.observe_opponent.side_effect = lambda *args: analyzed.set()

    def transcribe(**kwargs):
        if len(calls) == 1:
            assert analyzed.wait(3), 'Pending text timeout waited for in-flight ASR'
        return original(**kwargs)

    asr.audio.transcriptions.create = transcribe
    row = run(speaker, listener, asr)
    assert [b['flush_reason'] for b in row['analysis_batches']] == ['timeout', 'final']
    assert [b['sequences'] for b in row['analysis_batches']] == [[0], [1]]
    assert row['analysis_batches'][0]['buffered_seconds'] >= .25
    assert row['analysis_batches'][0]['analysis_end_monotonic'] <= row['heard'][1]['asr_ready_monotonic']
    assert row['listener_transcript'] == ' '.join(parts)
