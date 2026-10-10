import importlib.util
from pathlib import Path

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


def test_smaller_audio_bundle_preserves_per_request_bound(tmp_path):
    c = BudgetedClient(tmp_path)
    guard = AudioGuard(tmp_path, "audio", httpx.MockTransport(lambda r: httpx.Response(200, content=b"audio")), allowance=.1)
    client = httpx.Client(transport=guard)
    assert c.summary()["reserved_upper_usd"] == .1
    # Each 1000-character request reserves $0.06; the second cannot fit.
    client.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "a"*1000})
    with pytest.raises(BudgetExceeded):
        client.post("https://api.openai.com/v1/audio/speech", json={"model": "tts-1", "input": "a"*1000})
    guard.finish()
    assert len(guard.artifact["external_calls"]) == 1


def test_proposed_cap_cannot_change_existing_ledger(tmp_path):
    c = BudgetedClient(tmp_path)
    with pytest.raises(BudgetExceeded):
        AudioGuard(tmp_path, 'proposed', approved_cap=220)
    assert c.summary()['cap_usd'] == 200
    assert c.summary()['calls'] == 0


def test_audio_requires_exact_approved_cap_and_enforces_it(tmp_path):
    c = BudgetedClient(tmp_path, cap=220)
    with pytest.raises(BudgetExceeded):
        AudioGuard(tmp_path, 'wrong-cap')
    guard = AudioGuard(tmp_path, 'approved', approved_cap=220)
    guard.finish()
    c.db.execute("INSERT INTO calls(label,reserved,state) VALUES('prior',218.5,'ok')")
    c.db.commit()
    with pytest.raises(BudgetExceeded):
        AudioGuard(tmp_path, 'over-cap', approved_cap=220)
    assert c.summary()['cap_usd'] == 220


def test_full_legacy_audio_capacity_keeps_cost_and_request_stops(tmp_path):
    c = BudgetedClient(tmp_path)
    seen = []
    def provider(request):
        seen.append(request)
        return httpx.Response(200, content=b'audio')
    guard = AudioGuard(tmp_path, 'full', httpx.MockTransport(provider), allowance=3, max_requests=128)
    client = httpx.Client(transport=guard)
    for _ in range(128):
        client.post('https://api.openai.com/v1/audio/speech', json={'model': 'tts-1', 'input': 'Hi'})
    with pytest.raises(BudgetExceeded):
        client.post('https://api.openai.com/v1/audio/speech', json={'model': 'tts-1', 'input': 'Hi'})
    guard.finish()
    assert len(seen) == 128
    assert guard.artifact['blocked_dispatches'][0]['requests_used'] == 128
    assert c.summary()['accounted_exposure_usd'] == 3
    from streaming.experiment_accounting import reconcile_success
    assert reconcile_success(c.db, guard.request_id, guard.path)
    assert c.summary()['accounted_exposure_usd'] == pytest.approx(128*2*15e-6*4)


def test_asr_reserves_decoded_upload_duration_with_fourfold_margin(tmp_path):
    from io import BytesIO
    from pydub import AudioSegment
    c = BudgetedClient(tmp_path)
    guard = AudioGuard(tmp_path, 'asr', httpx.MockTransport(lambda r: httpx.Response(200, content=b'{"text":"Heard."}')))
    wav = BytesIO();AudioSegment.silent(duration=15200).export(wav, format='wav')
    httpx.Client(transport=guard).post('https://api.openai.com/v1/audio/transcriptions',
        data={'model':'whisper-1'}, files={'file':('speech.wav',wav.getvalue(),'audio/wav')})
    entry = guard.artifact['external_calls'][0]
    assert entry['reserved_usd'] == pytest.approx(4*16/60*.006)
    guard.finish()
    original = guard.path.read_bytes()
    guard.finish()
    assert guard.path.read_bytes() == original
    with pytest.raises(BudgetExceeded, match='finished'):
        httpx.Client(transport=guard).post('https://api.openai.com/v1/audio/speech',json={'model':'tts-1','input':'Late'})
    assert guard.path.read_bytes() == original
    from streaming.experiment_accounting import reconcile_success
    assert reconcile_success(c.db, guard.request_id, guard.path)
    assert c.summary()['accounted_exposure_usd'] == pytest.approx(entry['reserved_usd'])
