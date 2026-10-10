"""Offline accounting and model-routing tests for overlapping speech work."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location('adaptive_speech_benchmark', Path(__file__).resolve().parents[1]
                                             / 'experiments/incremental_planning/benchmark_adaptive_speech.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_background_text_calls_use_thread_owned_connections_and_stable_labels(tmp_path, monkeypatch):
    client = benchmark.ThreadedClient(tmp_path, label='generation')
    def transport(self, messages, **kwargs):
        # sqlite rejects this if the connection was inherited from another thread.
        assert self.db.execute('SELECT cap FROM budget').fetchone()[0] == 200
        return self.label
    monkeypatch.setattr(benchmark.BudgetedClient, 'complete', transport)
    with ThreadPoolExecutor(max_workers=2) as executor:
        a = executor.submit(client.complete, [{'role': 'user', 'content': 'Remaining speech.'}])
        b = executor.submit(client.complete_at, 'refinement', [{'role': 'user', 'content': 'Shorten.'}])
        assert a.result() == 'generation'
        assert b.result() == 'refinement'
    assert client.label == 'generation'


def test_refinement_cannot_bypass_run_budget_from_worker_thread(tmp_path, monkeypatch):
    client = benchmark.ThreadedClient(tmp_path)
    client.db.execute("INSERT INTO calls(label,reserved,state) VALUES(?,48,'error')", (benchmark.RUN+'/prior',))
    client.db.commit()
    transport = Mock()
    monkeypatch.setattr(benchmark.BudgetedClient, 'complete', transport)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.complete_at, benchmark.RUN+'/refinement',
                                 [{'role': 'user', 'content': 'Do not dispatch.'}])
        with pytest.raises(benchmark.BudgetExceeded):
            future.result()
    transport.assert_not_called()
