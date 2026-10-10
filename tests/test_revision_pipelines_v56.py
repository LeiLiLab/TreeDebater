"""Offline protocol/budget checks for the isolated streaming comparison."""
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

D=Path(__file__).resolve().parents[1]/'experiments/incremental_planning'
sys.path.insert(0,str(D))
import compare_revision_pipelines_v56 as probe
from streaming.experiment_client import BudgetedClient


def test_paragraph_stream_retains_partial_chunks_and_flushes_final_paragraph():
    got=[]
    parser=probe.ParagraphStream(got.append)
    for delta in ['First sent','ence.\n','\nSecond', ' sentence.\n\nThi','rd.']:
        parser.feed(delta)
    assert got==['First sentence.','Second sentence.']
    parser.finish()
    assert got==['First sentence.','Second sentence.','Third.']


class SSE(BytesIO):
    headers={'Content-Type':'text/event-stream'}


def packets(usage=True):
    values=[dict(model=probe.MODEL,choices=[dict(delta={'content':'First.\n\n'},finish_reason=None)]),
            dict(model=probe.MODEL,choices=[dict(delta={'content':'Last.'},finish_reason='stop')])]
    if usage:values.append(dict(choices=[],usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110)))
    return ''.join('data: '+json.dumps(p)+'\n\n' for p in values).encode()+b'data: [DONE]\n\n'


def test_stream_reserves_before_network_and_reconciles_reported_usage(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'LEDGER',tmp_path)
    client=BudgetedClient(tmp_path,cap=probe.GLOBAL_CAP,label='test')
    def open_response(*a,**k):
        assert client.db.execute("select state from calls").fetchone()[0]=='pending'
        return SSE(packets())
    got=[]
    with patch.object(probe,'urlopen',side_effect=open_response):
        text,ident,ttft=probe.stream_complete(client,[dict(role='user',content='Write.')],got.append)
    assert text=='First.\n\nLast.' and got==['First.\n\n','Last.']
    assert client.db.execute('select state from calls').fetchone()[0]=='ok'
    assert client.db.execute('select count(*) from budget_settlements').fetchone()[0]==1
    client.db.close()


def test_missing_stream_usage_stops_and_keeps_reservation(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'LEDGER',tmp_path)
    client=BudgetedClient(tmp_path,cap=probe.GLOBAL_CAP,label='test')
    with patch.object(probe,'urlopen',return_value=SSE(packets(False))):
        with pytest.raises(ValueError,match='usage missing'):
            probe.stream_complete(client,[dict(role='user',content='Write.')],lambda _:None)
    assert client.db.execute('select state from calls').fetchone()[0]=='error'
    assert client.db.execute('select count(*) from budget_settlements').fetchone()[0]==0
    client.db.close()


def test_stream_subcap_rejects_before_network(tmp_path,monkeypatch):
    monkeypatch.setattr(probe,'LEDGER',tmp_path)
    client=BudgetedClient(tmp_path,cap=probe.GLOBAL_CAP,label='test/stream')
    probe.arm(client.db,'test/',.001)
    with patch.object(probe,'urlopen') as dispatch:
        with pytest.raises(Exception,match='budget exceeded'):
            probe.stream_complete(client,[dict(role='user',content='Write.')],lambda _:None)
    dispatch.assert_not_called()
    assert client.db.execute('select count(*) from calls').fetchone()[0]==0
    client.db.close()


def test_parent_cap_raise_requires_recorded_authorization_and_keeps_old_spend(tmp_path):
    client=BudgetedClient(tmp_path,cap=probe.GLOBAL_CAP,label='test')
    probe.arm(client.db,probe.PARENT,20)
    client.db.execute("insert into calls(label,reserved,state) values(?,10,'error')",(probe.PARENT+'/old',))
    client.db.commit()
    with pytest.raises(ValueError,match='authorization'):
        probe.raise_authorized_parent_cap(client.db,dict(cumulative_cap_usd=50,authorization=''))
    probe.raise_authorized_parent_cap(client.db,dict(cumulative_cap_usd=50,authorization='User approved 50'))
    assert probe.accounted_exposure(client.db)==10
    with pytest.raises(Exception,match='budget exceeded'):
        client.db.execute("insert into calls(label,reserved,state) values(?,41,'pending')",(probe.PARENT+'/new',))
    client.db.rollback()
    client.db.close()


def test_playback_gaps_are_not_sum_of_generation_durations():
    case=dict(prefix_seconds=12,feedback_delay_seconds=3,speech_budget_seconds=60)
    result=dict(chunks=[dict(index=0,text='First.',text_ready_seconds=2,audio_ready_seconds=5,audio_seconds=20),
                        dict(index=1,text='Last.',text_ready_seconds=25,audio_ready_seconds=30,audio_seconds=20)])
    m=probe.metrics(result,case)
    assert m['first_body_gap_seconds']==0 and m['total_playback_gap_seconds']==1
    assert m['signed_duration_error_seconds']==-8
