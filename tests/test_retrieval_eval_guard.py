import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def module(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'experiments/retrieval_eval_v1/run.py'
    spec = importlib.util.spec_from_file_location('retrieval_eval_guard', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'HERE', tmp_path)
    monkeypatch.setattr(mod, 'ROOT', tmp_path)
    keys = tmp_path/'src/configs/api_key.json'
    keys.parent.mkdir(parents=True)
    keys.write_text(json.dumps({'OPENAI_API_KEY': 'test', 'DEEPSEEK_API_KEY': 'test'}))
    return mod


def config(cap=5):
    return dict(approved=True, approval='test-only', cap_usd=cap,
                rates_per_million={'text-embedding-3-small': [0.02, 0]})


def test_approval_required(module):
    with pytest.raises(RuntimeError, match='approval'):
        module.Guard(dict(approved=False))


def test_cap_prevents_dispatch(module, monkeypatch):
    network = Mock()
    monkeypatch.setattr(module, 'urlopen', network)
    guard = module.Guard(config(0.000001))
    with pytest.raises(RuntimeError, match='Budget stop'):
        guard.post('text-embedding-3-small', {'model': 'text-embedding-3-small', 'input': ['hello']})
    network.assert_not_called()


def test_network_failure_retains_reservation_and_blocks_retry(module, monkeypatch):
    network = Mock(side_effect=TimeoutError())
    monkeypatch.setattr(module, 'urlopen', network)
    guard = module.Guard(config())
    body = {'model': 'text-embedding-3-small', 'input': ['hello']}
    with pytest.raises(TimeoutError):
        guard.post('text-embedding-3-small', body)
    assert guard.summary()['exposure_usd'] > 0
    restarted = module.Guard(config())
    with pytest.raises(RuntimeError, match='uncertain cost'):
        restarted.post('text-embedding-3-small', body)
    assert network.call_count == 1
    assert restarted.summary() == guard.summary()


def test_stop_file_prevents_dispatch(module, monkeypatch):
    network = Mock()
    monkeypatch.setattr(module, 'urlopen', network)
    guard = module.Guard(config())
    (module.HERE/'STOP').touch()
    with pytest.raises(RuntimeError, match='Budget stop'):
        guard.post('text-embedding-3-small', {'model': 'text-embedding-3-small', 'input': ['hello']})
    network.assert_not_called()
