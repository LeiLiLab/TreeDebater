import json

import httpx
import pytest

from test_audio_probe_budget import AudioGuard
from streaming.experiment_client import BudgetedClient
from streaming.experiment_accounting import AuditMismatch, reconcile_success


def bundle(tmp_path, text='Hello'):
    client = BudgetedClient(tmp_path)
    guard = AudioGuard(tmp_path, 'audio', httpx.MockTransport(lambda r: httpx.Response(200, content=b'audio')))
    if text:
        httpx.Client(transport=guard).post('https://api.openai.com/v1/audio/speech',
            json={'model': 'tts-1', 'input': text})
    guard.finish()
    return client, guard


def test_release_only_unused_capacity_and_preserve_utf8_bound(tmp_path):
    client, guard = bundle(tmp_path, '你好')
    original = client.db.execute('select * from calls').fetchall()
    assert reconcile_success(client.db, guard.request_id, guard.path)
    assert client.summary()['accounted_exposure_usd'] == pytest.approx(6 * 15e-6 * 4)
    assert client.summary()['reported_usage_estimate_usd'] == pytest.approx(2 * 15e-6)
    assert not reconcile_success(client.db, guard.request_id, guard.path)
    assert client.db.execute('select * from calls').fetchall() == original
    assert client.summary()['cap_usd'] == 200


@pytest.mark.parametrize('state', ['pending', 'error', 'usage_missing'])
def test_incomplete_audio_retains_whole_reservation(tmp_path, state):
    client, guard = bundle(tmp_path)
    client.db.execute('update calls set state=?', (state,)); client.db.commit()
    assert not reconcile_success(client.db, guard.request_id, guard.path)
    assert client.summary()['accounted_exposure_usd'] == 1


def test_unfinished_subrequest_cannot_release_capacity(tmp_path):
    client, guard = bundle(tmp_path)
    artifact = json.loads(guard.path.read_text())
    artifact['external_calls'][0]['state'] = 'pending'
    guard.path.write_text(json.dumps(artifact))
    assert not reconcile_success(client.db, guard.request_id, guard.path)
    assert client.summary()['accounted_exposure_usd'] == 1


def test_mismatched_receipt_is_rejected(tmp_path):
    client, guard = bundle(tmp_path)
    artifact = json.loads(guard.path.read_text())
    artifact['external_calls'][0]['characters'] = 200
    guard.path.write_text(json.dumps(artifact))
    with pytest.raises(AuditMismatch):
        reconcile_success(client.db, guard.request_id, guard.path)
    assert client.summary()['accounted_exposure_usd'] == 1


def test_completed_zero_dispatch_bundle_releases_capacity(tmp_path):
    client, guard = bundle(tmp_path, '')
    assert reconcile_success(client.db, guard.request_id, guard.path)
    assert client.summary()['accounted_exposure_usd'] == 1e-9
