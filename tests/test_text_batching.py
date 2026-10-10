"""Deterministic timer races and shutdown without model or audio calls."""
from unittest.mock import Mock

import pytest

from streaming.config import InputConfig
from streaming.text_batching import TimedTextBatcher


def make_batcher(wait=60):
    timers, now = [], [0.]
    def timer_factory(seconds, callback):
        timer = Mock(callback=callback, seconds=seconds)
        timers.append(timer)
        return timer
    emit, error = Mock(), Mock()
    batcher = TimedTextBatcher(InputConfig(min_text_words=3, max_text_wait_seconds=wait),
        emit, error, timer_factory=timer_factory, clock=lambda: now[0])
    return batcher, timers, now, emit, error


def test_timeout_is_from_first_recognition_and_flushes_short_batch_once():
    batcher, timers, now, emit, error = make_batcher()
    first, second = {'text': 'First'}, {'text': 'Second'}
    batcher.append(first)
    now[0] = 40
    batcher.append(second)
    assert len(timers) == 1 and timers[0].seconds == 60
    now[0] = 60
    timers[0].callback()
    emit.assert_called_once_with([first, second], 'timeout', 60)
    batcher.close(drain=True)
    timers[0].callback()
    assert emit.call_count == 1
    error.assert_not_called()


def test_cancelled_timer_cannot_flush_newer_batch_or_emit_after_close():
    batcher, timers, now, emit, error = make_batcher()
    batcher.append({'text': 'One'})
    batcher.append({'text': 'two three'})
    assert emit.call_args.args[1] == 'threshold'
    timers[0].cancel.assert_called_once()
    batcher.append({'text': 'New'})
    timers[0].callback()  # Simulate callback already waiting on the lock at cancel.
    assert emit.call_count == 1
    batcher.close(drain=True)
    assert emit.call_args.args[1] == 'final' and emit.call_count == 2
    timers[1].callback()
    assert emit.call_count == 2
    with pytest.raises(RuntimeError, match='closed'):
        batcher.append({'text': 'Too late'})


def test_disabled_timeout_and_aborted_pending_buffer():
    batcher, timers, now, emit, error = make_batcher(wait=0)
    batcher.append({'text': 'Pending'})
    assert not timers
    batcher.close()
    batcher.close(drain=True)
    emit.assert_not_called()


def test_timer_dispatch_failure_latches_closed_and_reports_error():
    batcher, timers, now, emit, error = make_batcher()
    failure = RuntimeError('Analysis executor closed')
    emit.side_effect = failure
    batcher.append({'text': 'Short'})
    timers[0].callback()
    error.assert_called_once_with(failure)
    assert batcher.closed
    timers[0].callback()
    assert emit.call_count == 1
