import json
from unittest.mock import Mock

import pytest

from streaming.experiment_client import BudgetedClient, BudgetExceeded


def response():
    result = Mock()
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.read.return_value = json.dumps({"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 2}})
    return result


def test_reserve_before_dispatch_and_cap_survives_restart(tmp_path, monkeypatch):
    http = Mock(return_value=response())
    monkeypatch.setattr("streaming.experiment_client.urlopen", http)
    client = BudgetedClient(tmp_path, cap=0.04)
    assert client.text("Hi", 10) == "OK"
    assert client.summary()["reported_usage_estimate_usd"] > 0
    reloaded = BudgetedClient(tmp_path, cap=0.04)
    with pytest.raises(BudgetExceeded):
        reloaded.text("Hi", 10)
    assert http.call_count == 1
    assert reloaded.summary()["calls"] == 1
    with pytest.raises(ValueError):
        BudgetedClient(tmp_path, cap=200)


def test_failed_requests_remain_reserved_and_do_not_retry(tmp_path, monkeypatch):
    http = Mock(side_effect=TimeoutError("lost response"))
    monkeypatch.setattr("streaming.experiment_client.urlopen", http)
    client = BudgetedClient(tmp_path, cap=0.04)
    with pytest.raises(TimeoutError):
        client.text("Hi", 10)
    assert http.call_count == 1
    assert client.summary()["reserved_upper_usd"] > 0
    assert client.summary()["uncertain_calls"] == 1
    with pytest.raises(BudgetExceeded):
        client.text("Hi", 10)


def test_concurrent_clients_share_one_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("streaming.experiment_client.urlopen", Mock(return_value=response()))
    one = BudgetedClient(tmp_path, cap=0.04)
    two = BudgetedClient(tmp_path, cap=0.04)
    one.text("Hi", 10)
    with pytest.raises(BudgetExceeded):
        two.text("Hi", 10)


def test_usage_is_attributed_to_its_job_under_shared_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr("streaming.experiment_client.urlopen", Mock(return_value=response()))
    one = BudgetedClient(tmp_path, label="one")
    two = BudgetedClient(tmp_path, label="two")
    one.text("Hi", 10)
    two.text("Hi", 10)
    assert one.summary("one")["calls"] == 1
    assert two.summary("two")["calls"] == 1
    assert one.summary()["calls"] == 2
