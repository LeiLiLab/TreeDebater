"""Budget invariants with mocked transport: these tests make no paid requests."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sqlite3
from unittest.mock import Mock

import pytest
from fastapi import HTTPException


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'experiments/debater_baseline_gemma4/gateway.py'
    spec = importlib.util.spec_from_file_location('test_experiment_gateway',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = module.db.execute("SELECT sql FROM sqlite_master WHERE name='calls'").fetchone()[0]
    module.db.close()
    module.ROOT = tmp_path
    module.keys = {}
    module.db = sqlite3.connect(tmp_path/'cost.sqlite',check_same_thread=False)
    module.db.execute(schema)
    module.db.commit()
    transport = Mock(side_effect=AssertionError('Unexpected transport dispatch'))
    monkeypatch.setattr(module.requests,'post',transport)
    yield module, transport
    module.db.close()


class Request:
    def __init__(self,model='google.gemma-4-26b-a4b'):
        self.body = {'model':model,'messages':[{'role':'user','content':'Test.'}],'max_tokens':128}

    async def json(self):
        return dict(self.body)


def test_committed_budget_blocks_dispatch_and_latches_stop(gateway):
    module,transport = gateway
    module.db.execute("INSERT INTO calls(charged,status) VALUES(199.99,'pending')")
    module.db.commit()
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.forward('v1/chat/completions',Request()))
    assert error.value.status_code == 402
    transport.assert_not_called()
    assert (module.ROOT/'STOP').exists()


def test_unknown_model_cannot_bypass_budget(gateway):
    module,transport = gateway
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.forward('v1/chat/completions',Request('unpriced-model')))
    assert error.value.status_code == 400
    transport.assert_not_called()
    assert module.db.execute('SELECT count(*) FROM calls').fetchone()[0] == 0


def test_failed_request_retains_durable_pre_dispatch_reservation(gateway):
    module,transport = gateway
    def fail(*args,**kwargs):
        with sqlite3.connect(module.ROOT/'cost.sqlite') as connection:
            row = connection.execute('SELECT bound,charged,status FROM calls').fetchone()
        assert row[0] == row[1] and row[1] > 0 and row[2] == 'pending'
        raise ConnectionError('Mocked transport failure')
    transport.side_effect = fail
    with pytest.raises(HTTPException) as error:
        asyncio.run(module.forward('v1/chat/completions',Request()))
    assert error.value.status_code == 502
    bound,charged,status = module.db.execute('SELECT bound,charged,status FROM calls').fetchone()
    assert bound == charged and status == 'error'


def test_historical_gateway_settles_reported_usage_and_releases_reservation(gateway):
    module,transport = gateway
    payload = {'model':module.MODEL,'usage':{'prompt_tokens':100,'completion_tokens':20},
               'choices':[{'message':{'content':'Test response.'},'finish_reason':'stop'}]}
    response = Mock(content=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    response.json.return_value = payload
    transport.side_effect = None
    transport.return_value = response
    asyncio.run(module.forward('v1/chat/completions',Request()))
    bound,charged,estimate,status = module.db.execute('SELECT bound,charged,estimate,status FROM calls').fetchone()
    assert estimate == pytest.approx((100*.13+20*.4)/1e6)
    # This historical gateway settles successful calls at reported usage.
    # The live motion gateway has a separate conservative settlement policy.
    assert charged == estimate < bound and status == 'ok'
