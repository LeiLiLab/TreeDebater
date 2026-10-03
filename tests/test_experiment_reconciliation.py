import json
import sqlite3
from unittest.mock import Mock

import pytest

from streaming.experiment_accounting import (AuditMismatch, accounted_exposure, initialize_accounting,
                                             reconcile_success, successful_charge)
from streaming.experiment_client import BudgetedClient, BudgetExceeded
from test_experiment_budget import response


def test_completed_usage_is_settled_once_without_rewriting_original_row(tmp_path, monkeypatch):
    monkeypatch.setattr('streaming.experiment_client.urlopen', Mock(return_value=response()))
    c = BudgetedClient(tmp_path)
    c.text('Hi', 10)
    original = c.db.execute('select * from calls').fetchone()
    s = c.summary()
    assert s['accounted_exposure_usd'] == pytest.approx(4*s['reported_usage_estimate_usd'])
    assert s['reserved_upper_usd'] > s['accounted_exposure_usd']
    assert not reconcile_success(c.db, 1, tmp_path/'call_000001.json')
    assert c.db.execute('select * from calls').fetchone() == original
    assert c.db.execute('select count(*) from budget_settlements').fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError, match='append-only'):
        c.db.execute('delete from budget_settlements')
    c.db.rollback()


def test_changed_artifact_cannot_silently_replace_a_settlement(tmp_path, monkeypatch):
    monkeypatch.setattr('streaming.experiment_client.urlopen', Mock(return_value=response()))
    c = BudgetedClient(tmp_path)
    c.text('Hi', 10)
    path = tmp_path/'call_000001.json'
    a = json.loads(path.read_text());a['response']['usage']['completion_tokens'] = 1
    path.write_text(json.dumps(a))
    with pytest.raises(AuditMismatch):
        reconcile_success(c.db, 1, path)


@pytest.mark.parametrize('state', ['pending', 'error', 'usage_missing'])
def test_unsettled_work_preserves_full_original_reservation(tmp_path, state):
    c = BudgetedClient(tmp_path)
    c.db.execute('insert into calls(label,reserved,state) values(?,?,?)', ('unknown',199.9,state))
    c.db.commit()
    assert accounted_exposure(c.db) == 199.9
    with pytest.raises(BudgetExceeded):
        c.complete([], model='gpt-5.6-sol', max_tokens=1600)


def test_missing_or_negative_usage_never_creates_budget_credit(tmp_path, monkeypatch):
    for usage in ({}, {'prompt_tokens': 1, 'completion_tokens': -2}):
        r = response();r.read.return_value = json.dumps({'choices':[{'message':{'content':'ok'}}], 'usage':usage})
        monkeypatch.setattr('streaming.experiment_client.urlopen', Mock(return_value=r))
        c = BudgetedClient(tmp_path)
        c.text('Hi')
        assert c.db.execute('select count(*) from budget_settlements').fetchone()[0] == 0
        assert c.summary()['accounted_exposure_usd'] == c.summary()['reserved_upper_usd']


def test_legacy_ledger_migration_is_conservative_and_does_not_reset_cap(tmp_path):
    d = sqlite3.connect(tmp_path/'cost.sqlite')
    d.execute('create table calls(id integer primary key,label text,reserved real,state text)')
    d.execute("insert into calls values(1,'legacy',199.9,'ok')")
    d.commit()
    initialize_accounting(d)
    assert accounted_exposure(d) == 199.9  # 'ok' alone does not grant a credit
    assert d.execute('select * from calls').fetchone() == (1,'legacy',199.9,'ok')
