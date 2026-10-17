"""Offline mixed-match, publication and durable budget regressions."""

from optional_experiments import require_local_experiment
require_local_experiment('experiments/tools/matrix_4_1', module=True)

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/'experiments/tools/matrix_4_1'
sys.path.insert(0,str(SCRIPTS))
from engine import run_match
from gateway import Ledger,Stop
from launch import select,validate_approval


class FakePlayer:
    motion='Motion'
    def __init__(self,name,side):
        self.name,self.side=name,side
        self.streaming_listen=name=='Our'
        self.observed,self.calls,self.delivered=[],[],[]
        self.closed=False
    def prepare(self):pass
    def generate(self,stage,history,budget,folder,emit,incoming):
        if incoming:
            history=incoming.completion.result()
            assert incoming.recognized==history
        self.calls.append((stage,history))
        for index in range(2):
            emit(index,folder/f'{index}.mp3',f'{self.side} {stage} public {index}.',.01)
        return 'PRIVATE UNPUBLISHED DRAFT'
    def observe(self,text,side,stage):
        assert side!=self.side
        assert 'PRIVATE' not in text
        self.observed.append((stage,side,text))
    def handoff(self,stage,prior,budget):return None
    def fit_last_chunk(self,item,remaining,folder):return None
    def references(self,text):return ''
    def record_delivered(self,text):self.delivered.append(text)
    def close(self):self.closed=True


@pytest.mark.parametrize('opponent',['Debater','Baseline','DeepDebater','TreeDebater'])
@pytest.mark.parametrize('ours_side',['for','against'])
def test_six_turns_every_opponent_and_stance_no_judge(tmp_path,opponent,ours_side):
    other='against' if ours_side=='for' else 'for'
    players={ours_side:FakePlayer('Our',ours_side),other:FakePlayer(opponent,other)}
    result=run_match(players,dict(opening=1,rebuttal=1,closing=1),tmp_path,pace=False)
    assert result['status']=='complete'
    assert [(x['stage'],x['side']) for x in result['history']]==[
        (phase,side) for phase in ('opening','rebuttal','closing') for side in ('for','against')]
    for side,player in players.items():
        assert len(player.calls)==3 and player.closed
        for stage,history in player.calls:
            expected=2*['opening','rebuttal','closing'].index(stage)+(side=='against')
            assert len(history)==expected
        assert all('PRIVATE' not in text for text in player.delivered)
    assert players[ours_side].observed
    assert not players[other].observed
    assert all(len(row['chunks'])==2 for row in result['turns'])
    assert 'PRIVATE' not in (tmp_path/'judge_input.json').read_text()
    assert not any('judgment' in key for key in result)


def test_truncated_tail_and_later_chunks_never_enter_public_history(tmp_path):
    players={s:FakePlayer('Our' if s=='for' else 'Debater',s) for s in ('for','against')}
    result=run_match(players,dict(opening=.015,rebuttal=.015,closing=.015),tmp_path,pace=False)
    assert all(row['truncated'] and row['audio_seconds']==.01 for row in result['turns'])
    assert all('public 1' not in speech['content'] for speech in result['history'])


def test_observer_error_is_a_failed_match_not_a_successful_empty_turn(tmp_path):
    ours=FakePlayer('Our','against')
    ours.observe=Mock(side_effect=RuntimeError('listener failed'))
    with pytest.raises(RuntimeError,match='listener failed'):
        run_match({'for':FakePlayer('Baseline','for'),'against':ours},dict(opening=1,rebuttal=1,closing=1),tmp_path,pace=False)
    assert json.loads((tmp_path/'result.json').read_text())['status']=='failed'


def test_ledger_preserves_study_cost_and_rejects_match_overrun(tmp_path):
    path=tmp_path/'cost.sqlite'
    first=Ledger(path,5,'first',1)
    rid=first.reserve('text','ours',.6)
    first.finish(rid,seconds=.01,estimate=.2)
    first.db.close()
    second=Ledger(path,5,'second',.4)
    assert second.status()['conservative_usd']==pytest.approx(.2)
    assert second.status()['match_conservative_usd']==0
    second.reserve('tts','ours',.3)
    with pytest.raises(Stop):second.reserve('tts','ours',.2)
    assert second.status()['stopped']
    assert second.status()['conservative_usd']==pytest.approx(.5)
    second.db.close()


def test_pending_request_retains_charge_and_prevents_restart(tmp_path):
    path=tmp_path/'cost.sqlite'
    first=Ledger(path,5,'first',1)
    first.reserve('text','ours',.6)
    first.db.close()
    with pytest.raises(Stop,match='pending'):
        Ledger(path,5,'first',1)


def test_approval_template_cannot_launch(tmp_path):
    path=tmp_path/'study.json';path.write_text('{}')
    with pytest.raises(ValueError,match='approval'):
        validate_approval(path,dict(approved=False),[dict(run_id='a')])
    from preflight import sha
    approval=dict(approved=True,currency='USD',rates_confirmed=True,study_sha256=sha(path),total_cap_usd=10,match_caps={'a':1})
    assert validate_approval(path,approval,[dict(run_id='a')])==10
    with pytest.raises(ValueError):validate_approval(path,approval,[dict(run_id='b')])
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError):validate_approval(path,approval,[dict(run_id='a')])


def test_env_separates_player_output_and_does_not_construct_old_judges(monkeypatch):
    import env as module
    from agents import DebaterConfig,JudgeConfig,AudienceConfig
    created=[]
    def fake(config,motion):
        player=NS(config=config)
        created.append(player)
        return player
    forbidden=Mock(side_effect=AssertionError('Old evaluation client created'))
    monkeypatch.setattr(module,'TreeDebater',fake)
    monkeypatch.setattr(module,'Judge',forbidden)
    monkeypatch.setattr(module,'Audience',forbidden)
    config=module.EnvConfig(motion='Motion',judge_num=1,audience_num=3,
        judge_config=JudgeConfig(),audience_config=AudienceConfig(),
        debater_config=[DebaterConfig(type='treedebater',side='for',streaming_output={'speech_mode':'listening_prefix'}),
                        DebaterConfig(type='treedebater',side='against',streaming_output={'speech_mode':'full_script'})])
    environment=module.Env(config,False)
    assert environment.judge is None and environment.audiences==[]
    forbidden.assert_not_called()
    ours,tree=created
    assert ours.streaming_output_config.speech_mode=='listening_prefix'
    assert tree.streaming_output_config.speech_mode=='full_script'
    ours.streaming_output_config.first_chunk_seconds=99
    assert tree.streaming_output_config.first_chunk_seconds!=99


def test_matrix_gateway_prices_approved_tts():
    from gateway import classify
    kind,bound,_,cost,margin=classify('/v1/audio/speech',dict(model='tts-1',voice='echo',input='Hello.'))
    assert kind=='tts' and cost==pytest.approx(6*15/1e6) and bound==cost*margin
    with pytest.raises(Stop,match='Unapproved TTS'):
        classify('/v1/audio/speech',dict(model='tts-1-hd',voice='echo',input='Hello.'))


def test_large_text_reserves_by_payload_and_still_obeys_cap(tmp_path):
    from gateway import classify, MODEL
    large=dict(model=MODEL,messages=[dict(role='user',content='Evidence. '*40000)],max_tokens=8192)
    kind,bound,_,_,margin=classify('/v1/chat/completions',large)
    assert kind=='text' and bound>(262144*.13+8192*.40)/1e6*1.75
    ledger=Ledger(tmp_path/'large.sqlite',bound/2,'match',bound/2)
    with pytest.raises(Stop,match='Budget'):
        ledger.reserve(kind,'test',bound)
    assert ledger.status()['requests']==0
    with pytest.raises(Stop,match='transport'):
        classify('/v1/chat/completions',dict(model=MODEL,messages=[dict(role='user',content='x'*9000000)]))


def test_settled_request_cannot_be_overwritten_by_cleanup(tmp_path):
    ledger=Ledger(tmp_path/'settled.sqlite',10,'match',10)
    rid=ledger.reserve('tts','test',.3)
    with pytest.raises(Stop):ledger.finish(rid,seconds=.1,status=429)
    ledger.finish(rid,seconds=.2,status=502)
    assert ledger.db.execute('select status,charge,http_status from calls where id=?',(rid,)).fetchone()==('failed_reserved',.3,429)
    assert ledger.status()['stopped']


def _reserve_process(path, name, barrier, results):
    ledger=Ledger(path,1,name,1)
    barrier.wait(timeout=10)
    try:
        ledger.reserve('text',name,.6)
        results.put('reserved')
    except Stop:
        results.put('stopped')
    finally:
        ledger.db.close()


def test_cross_process_reservations_cannot_overspend(tmp_path):
    import multiprocessing as mp
    import sqlite3
    ctx=mp.get_context('fork')
    path=str(tmp_path/'parallel.sqlite')
    Ledger(path,1,'init',1).db.close()
    barrier=ctx.Barrier(2); results=ctx.Queue()
    children=[ctx.Process(target=_reserve_process,args=(path,str(i),barrier,results)) for i in range(2)]
    for child in children:child.start()
    for child in children:
        child.join(15)
        assert child.exitcode==0
    assert sorted(results.get(timeout=2) for _ in children)==['reserved','stopped']
    with sqlite3.connect(path) as db:
        assert db.execute('select sum(charge) from calls').fetchone()[0]==pytest.approx(.6)
        assert db.execute('select stopped from settings').fetchone()[0]==1


def test_new_match_accepts_other_live_match_pending_but_not_its_own(tmp_path):
    path=tmp_path/'live.sqlite'
    first=Ledger(path,2,'first',2)
    first.reserve('text','first',.6)
    second=Ledger(path,2,'second',2)
    second.reserve('text','second',.6)
    assert second.status()['conservative_usd']==pytest.approx(1.2)
    first.db.close();second.db.close()
    with pytest.raises(Stop,match='pending'):Ledger(path,2,'first',2)


def test_parallel_scheduler_limits_overlap_and_runs_every_job_once():
    import threading,time
    from launch import run_jobs
    lock=threading.Lock();running=0;peak=0;seen=[]
    def run(job):
        nonlocal running,peak
        with lock:
            running+=1;peak=max(peak,running);seen.append(job)
        time.sleep(.03)
        with lock:running-=1
    run_jobs(list(range(7)),3,run,lambda:False)
    assert peak==3 and sorted(seen)==list(range(7)) and running==0


def test_parallel_scheduler_does_not_dispatch_after_failure():
    import threading,time
    from launch import run_jobs
    stopped=threading.Event();seen=[]
    def run(job):
        seen.append(job)
        if job==0:
            stopped.set()
            raise RuntimeError('failed')
        time.sleep(.03)
    with pytest.raises(RuntimeError,match='failed'):
        run_jobs(list(range(8)),3,run,stopped.is_set)
    assert set(seen)<=set(range(3))
