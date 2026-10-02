import importlib.util
from pathlib import Path
import sqlite3

import httpx
import pytest

from streaming.experiment_client import BudgetedClient, BudgetExceeded

spec = importlib.util.spec_from_file_location("audio_probe", Path(__file__).resolve().parents[1]
                                            / "experiments/incremental_planning/audio_probe.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
AudioGuard = module.AudioGuard


def test_audio_subbudget_is_reserved_in_shared_ledger_before_provider_call(tmp_path):
    c = BudgetedClient(tmp_path)
    seen = []
    def provider(request):
        assert c.summary()["reserved_upper_usd"] == 1
        seen.append(request)
        return httpx.Response(200, content=b"audio")
    guard = AudioGuard(tmp_path, "audio", httpx.MockTransport(provider))
    client = httpx.Client(transport=guard)
    client.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "Hello"})
    assert guard.artifact["external_calls"][0]["state"] == "ok"
    guard.finish()
    assert c.summary()["reported_usage_estimate_usd"] == pytest.approx(5*15/1e6)
    assert c.summary()["uncertain_calls"] == 0
    assert len(seen) == 1


def test_provider_error_latches_audio_closed_and_retains_full_reservation(tmp_path):
    c = BudgetedClient(tmp_path)
    calls = []
    def provider(request):
        calls.append(request)
        return httpx.Response(503, content=b"unavailable")
    guard = AudioGuard(tmp_path, "audio", httpx.MockTransport(provider))
    client = httpx.Client(transport=guard)
    with pytest.raises(RuntimeError):
        client.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "Hello"})
    with pytest.raises(BudgetExceeded):
        client.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "Hello"})
    guard.finish()
    assert len(calls) == 1
    assert c.summary()["reserved_upper_usd"] == 1 and c.summary()["uncertain_calls"] == 1


def test_audio_cannot_bypass_existing_global_cap(tmp_path):
    c = BudgetedClient(tmp_path)
    c.db.execute("INSERT INTO calls(label,reserved,state) VALUES('prior',199.5,'ok')")
    c.db.commit()
    with pytest.raises(BudgetExceeded):
        AudioGuard(tmp_path, "audio", httpx.MockTransport(lambda r: httpx.Response(200)))
    assert c.summary()["calls"] == 1
