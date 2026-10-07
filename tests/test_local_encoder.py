"""Worker lifecycle, caching, and offline configuration contracts."""
import io
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from utils.local_encoder import LocalEncoder


def worker(monkeypatch, response):
    import utils.local_encoder as module
    p=NS(stdin=io.StringIO(),stdout=io.StringIO(json.dumps(response)+'\n'),
         poll=Mock(return_value=None),terminate=Mock(),wait=Mock(),kill=Mock())
    launch=Mock(return_value=p)
    monkeypatch.setattr(module.subprocess,'Popen',launch)
    monkeypatch.setattr(module.select,'select',lambda *args:([p.stdout],[],[]))
    return p,launch


def test_resident_encoder_deduplicates_and_caches_without_network(monkeypatch):
    p,launch=worker(monkeypatch,{'vectors':[[1.,0.],[0.,1.]]})
    encoder=LocalEncoder()
    try:
        result=encoder.encode(['first','second','first'])
        assert result.shape==(3,2)
        assert json.loads(p.stdin.getvalue())==['first','second']
        assert np.array_equal(encoder.encode(['second']),[[0.,1.]])
        launch.assert_called_once()
        env=launch.call_args.kwargs['env']
        assert env['HF_HUB_OFFLINE']==env['TRANSFORMERS_OFFLINE']=='1'
        assert env['OMP_NUM_THREADS']=='2'
    finally:
        encoder.close()
    p.terminate.assert_called_once()
    assert encoder.process is None


def test_worker_error_closes_process_without_fallback(monkeypatch):
    p,_=worker(monkeypatch,{'error':'Offline encoder unavailable'})
    encoder=LocalEncoder()
    with pytest.raises(RuntimeError,match='Offline encoder unavailable'):
        encoder.encode(['claim'])
    p.terminate.assert_called_once()
    assert encoder.process is None and not encoder.cache


def test_worker_timeout_cleans_up(monkeypatch):
    import utils.local_encoder as module
    p,_=worker(monkeypatch,{'vectors':[[1.]]})
    monkeypatch.setattr(module.select,'select',lambda *args:([],[],[]))
    encoder=LocalEncoder()
    with pytest.raises(TimeoutError):encoder.encode(['claim'])
    p.terminate.assert_called_once()
