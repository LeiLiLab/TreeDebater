"""Corrected comparison: retain the native adaptive TTS pipeline in BOTH arms.

Whole-stream evidence revision vs evidence-aware native paragraph length edits.
No production source mutations. No paid request bypasses the existing ledger.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from datetime import datetime, timezone
from io import BytesIO
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import argparse

import compare_revision_pipelines_v56 as base
from streaming.config import OutputConfig
from streaming.experiment_client import BudgetedClient, BudgetExceeded
from streaming.experiment_accounting import reconcile_success, exposure_query
from audio_probe import AudioGuard

ROOT,D,LEDGER=base.ROOT,base.D,base.LEDGER
RUN=base.PARENT+'-native-revision-v57'
OUT=LEDGER/RUN
RECORD=D/'prepared_native_revision_v57.json'
POLICIES=('whole_stream_native','paragraph_evidence_native')
CONFIG_KEYS={f.name for f in fields(OutputConfig)}


def config():
    original=json.loads((D/'prepared_full_debate_v53_evidence.json').read_text())['output_config']
    return OutputConfig(**{k:v for k,v in original.items() if k in CONFIG_KEYS})


def paragraph_prompt(case,text,n_words,preceding,following,source):
    data=copy.deepcopy(case['material'])
    data['current_paragraph']=dict(source=source,current_proposal=text,target_words=n_words,
        committed_preceding=preceding,next_paragraph_readonly=following)
    task=case['writing_task']+'''\nNATIVE PARAGRAPH REVISION TASK:
Revise ONLY current_paragraph.current_proposal to approximately current_paragraph.target_words words. This call combines the supplied global feedback, relevant evidence and the existing paragraph's length target. Return only this paragraph, not the full body.
Use the full unpublished draft to understand coverage and argument ownership. The current paragraph's immutable source determines its role; relevant supplied evidence may support that role. Earlier proposals are revisable and never establish facts. Preserve qualifications, speaker position, claim ownership and negation. Do not fill time with invented facts. If expansion is needed, clarify this paragraph's existing reasoning and relevant sourced support.
committed_preceding is immutable delivered or queued text. Do not repeat its arguments. next_paragraph_readonly will still be spoken: do not copy, anticipate or move its distinct claims into this paragraph. Keep the transition natural. Apply the specific global feedback that concerns this paragraph.\n'''
    request=copy.deepcopy(case['request'])
    request['messages'][-1]['content']=task+'\nContext and material (data):\n'+json.dumps(data,ensure_ascii=False)
    return request['messages']


def run_one(record,policy,case,repeat):
    import httpx
    from openai import OpenAI
    from pydub import AudioSegment
    import tts_streaming as tts
    cfg=config()
    name=f'{policy}_case{case["original_call_id"]}_r{repeat}'
    path=OUT/name
    path.mkdir(exist_ok=False)
    label=RUN+'/'+policy+'/'+name
    guard=AudioGuard(LEDGER,label+'/tts',allowance=2.,approved_cap=base.GLOBAL_CAP,max_requests=64)
    clients=[]
    requests=[]
    events=[]
    stop=threading.Event()
    lock=threading.Lock()
    prefix_ready=threading.Event()
    prefix_at=[None]
    result=dict(status='running',policy=policy,case=case['original_call_id'],repeat=repeat,
        output_config=asdict(cfg),chunks=[],text_requests=requests,speculative_audio=events,
        feedback_delay_seconds=case['feedback_delay_seconds'],evidence_policy='full-stream then native length-only' if policy==POLICIES[0] else 'native length edits with full feedback/evidence; raw candidate and fallback unchanged')
    base.save(path/'result.json',result)
    original_factory=tts.OpenAI
    original_text=tts._text_request
    original_revise=tts._revise_to_n_words
    original_query=tts._query_time_profiled
    original_retry=tts._tts_with_retry
    pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='v57-preparation')
    audio_pool=ThreadPoolExecutor(max_workers=2,thread_name_prefix='v57-stream-audio')
    cached={}
    t0=time.perf_counter()

    def fail():
        stop.set()
        with guard.lock:
            guard.failed=True

    def factory(*a,**kw):
        client=OpenAI(http_client=httpx.Client(transport=guard,timeout=60),max_retries=0,
            base_url='https://api.openai.com/v1',timeout=60)
        with lock:clients.append(client)
        return client

    def complete(messages,kind,max_tokens=4096):
        if stop.is_set():raise BudgetExceeded('Stopped; no further text request')
        with lock:
            if len(requests)>=48:
                fail();raise BudgetExceeded('Per-speech model request cap')
            index=len(requests)
            event=dict(index=index,kind=kind,start_seconds=time.perf_counter()-t0)
            requests.append(event)
        meter=BudgetedClient(LEDGER,cap=base.GLOBAL_CAP,label=label+f'/text_{index}_{kind}')
        try:
            text=meter.complete(messages,model=base.MODEL,temperature=.3,max_tokens=max_tokens,request_timeout=60)
            event['done_seconds']=time.perf_counter()-t0
            event['ledger_id']=meter.db.execute('select max(id) from calls where label=?',(meter.label,)).fetchone()[0]
            artifact=json.loads((LEDGER/f'call_{event["ledger_id"]:06}.json').read_text())
            if artifact.get('truncated'):raise ValueError('Truncated native paragraph revision')
            return text
        except BaseException:
            fail();raise
        finally:meter.db.close()

    def text_request(client,model,messages,max_tokens,**kw):
        text=complete(messages,'length_only',max_tokens)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def evidence_revise(client,text,n_words,prev_texts,next_chunk_text='',model=None,motion='',side='',*,source_text=None):
        return complete(paragraph_prompt(case,text,n_words,prev_texts,next_chunk_text,source_text or text),'evidence_and_length')

    def query(client,text,voice='echo',speed=1.,model='tts-1'):
        if stop.is_set():raise BudgetExceeded('Stopped; no further TTS request')
        with lock:future=cached.get((text,voice,speed,model))
        if future is not None:
            value=future.result()
            with lock:events.append(dict(kind='reused_stream_audio',text=text,seconds=time.perf_counter()-t0))
            return value
        try:return original_query(client,text,voice=voice,speed=speed,model=model)
        except BaseException:
            fail();raise

    def no_retry(client,text,voice='echo',speed=1.,max_attempts=1,model='tts-1'):
        return query(client,text,voice,speed,model)

    def pre_synthesize(text):
        client=factory()
        start=time.perf_counter()
        if stop.is_set():raise BudgetExceeded('Stopped speculative TTS')
        try:
            out=original_query(client,text,voice=cfg.voice,model=cfg.model)
            with lock:events.append(dict(kind='stream_audio_ready',text=text,start_seconds=start-t0,
                ready_seconds=time.perf_counter()-t0,audio_seconds=out['audio_seconds']))
            return out
        except BaseException:
            fail();raise

    def emit(paragraph):
        # Reuse exact text only; native whole-body segmentation remains authoritative.
        for part in tts.split_body_chunks(paragraph,case['body_budget_seconds'],cfg):
            with lock:
                key=(part,cfg.voice,1.,cfg.model)
                if key in cached:continue
                if len(cached)>=8:raise ValueError('Stream emitted too many speculative audio pieces')
                cached[key]=audio_pool.submit(pre_synthesize,part)

    def stream_body():
        if not prefix_ready.wait(10):raise TimeoutError('Prefix publication failed')
        wait=max(0.,prefix_at[0]+case['feedback_delay_seconds']-time.perf_counter())
        if stop.wait(wait):raise RuntimeError('Stopped before feedback ready')
        meter=BudgetedClient(LEDGER,cap=base.GLOBAL_CAP,label=label+'/whole_stream')
        event=dict(kind='whole_stream',start_seconds=time.perf_counter()-t0)
        with lock:requests.append(event)
        try:
            req=copy.deepcopy(case['request'])
            req['messages'][-1]['content']+='\nSeparate complete spoken paragraphs with a blank line. Output them in final delivery order; do not revisit earlier paragraphs.'
            parser=base.ParagraphStream(emit)
            text,ident,ttft=base.stream_complete(meter,req['messages'],parser.feed,max_tokens=4096,temperature=.3)
            parser.finish()
            event.update(ledger_id=ident,first_token_seconds=ttft,done_seconds=time.perf_counter()-t0)
            result['whole_revised_body']=text
            return text
        except BaseException:
            fail();raise
        finally:meter.db.close()

    def publish(index,audio_path,text,duration):
        now=time.perf_counter()
        result['chunks'].append(dict(index=index,text=text,path=str(audio_path),audio_seconds=duration,
            ready_seconds=now-t0))
        if index==0:
            prefix_at[0]=now
            prefix_ready.set()

    try:
        tts.OpenAI=factory
        tts._text_request=text_request
        tts._query_time_profiled=query
        tts._tts_with_retry=no_retry
        if policy==POLICIES[1]:tts._revise_to_n_words=evidence_revise
        future=pool.submit(stream_body) if policy==POLICIES[0] else None
        def tail():
            if future is not None:return future.result()
            if not prefix_ready.wait(10):raise TimeoutError('Prefix missing')
            wait=max(0.,prefix_at[0]+case['feedback_delay_seconds']-time.perf_counter())
            if stop.wait(wait):raise RuntimeError('Stopped before feedback ready')
            return case['material']['unpublished_draft']
        raw=Path(case['prefix_audio']).read_bytes()
        prepared=dict(text=case['material']['already_spoken_prefix'],voice=cfg.voice,model=cfg.model,
            tts_out=dict(audio_seconds=case['prefix_seconds'],tts_api_s=0.,mp3_parse_s=0.,mp3_bytes=raw))
        text,_,duration=tts.convert_text_to_speech_streaming(case['material']['already_spoken_prefix'],
            str(path/'speech.mp3'),case['speech_budget_seconds'],config=cfg,on_chunk=publish,
            motion=case['material']['motion'],side=case['side'],tail_supplier=tail,prepared_first_audio=prepared)
        if stop.is_set():raise RuntimeError('A guarded request failed during native processing')
        cursor=result['chunks'][0]['ready_seconds']
        gaps=[]
        for row in result['chunks']:
            gap=max(0.,row['ready_seconds']-cursor)
            row['gap_seconds']=gap
            row['playback_start_seconds']=max(cursor,row['ready_seconds'])
            cursor=row['playback_start_seconds']+row['audio_seconds']
            gaps.append(gap)
        seconds=sum(r['audio_seconds'] for r in result['chunks'])
        result.update(status='completed',text=text,metrics=dict(
            first_body_audio_seconds=result['chunks'][1]['ready_seconds']-result['chunks'][0]['ready_seconds'],
            first_body_after_feedback_seconds=result['chunks'][1]['ready_seconds']-result['chunks'][0]['ready_seconds']-case['feedback_delay_seconds'],
            final_audio_ready_seconds=result['chunks'][-1]['ready_seconds']-result['chunks'][0]['ready_seconds'],
            max_playback_gap_seconds=max(gaps),total_playback_gap_seconds=sum(gaps),audio_seconds=seconds,
            signed_duration_error_seconds=seconds-case['speech_budget_seconds'],duration_error_pct=100*(seconds-case['speech_budget_seconds'])/case['speech_budget_seconds'],
            chunks=len(result['chunks'])-1,text_calls=len(requests)))
        (path/'speech.txt').write_text(text+'\n')
        combined=AudioSegment.empty()
        for row in result['chunks']:
            combined+=AudioSegment.silent(duration=round(row['gap_seconds']*1000))
            combined+=AudioSegment.from_file(row['path'])
        combined.export(path/'playback.wav',format='wav')
    except BaseException as exc:
        fail();result.update(status='stopped',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        audio_pool.shutdown(wait=True,cancel_futures=True)
        tts.OpenAI=original_factory;tts._text_request=original_text;tts._revise_to_n_words=original_revise
        tts._query_time_profiled=original_query;tts._tts_with_retry=original_retry
        for c in clients:c.close()
        guard.finish()
        reconcile_success(guard.db,guard.request_id,guard.path)
        guard.db.close()
        result['wall_seconds']=time.perf_counter()-t0
        base.save(path/'result.json',result)
    print(json.dumps(dict(run=name,**result['metrics']),ensure_ascii=False),flush=True)


def prepare():
    if RECORD.exists():raise ValueError('Preparation already exists')
    old=json.loads(base.RECORD.read_text())
    schedule=[dict(r,policy=POLICIES[0] if r['policy']=='whole_stream' else POLICIES[1]) for r in old['schedule']]
    record=dict(status='approved',run_id=RUN,created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User approved USD50 total for both schemes; clarified BOTH retain the original adaptive paragraph length correction, differing in where evidence is used.',
        parent_cap_usd=50,run_cap_usd=12,policy_caps_usd={p:6 for p in POLICIES},estimate_new_usd=[1,4],
        cases=old['cases'],schedule=schedule,output_config=asdict(config()),
        boundaries='Use actual existing TTS pipeline unchanged. Whole B stream pre-synthesizes exact paragraph candidates; native pipeline still determines final segmentation, budgets and publication after full draft arrives. Paragraph policy adds global evidence/feedback to existing optional length edits, preserving raw candidates and fallback; paragraphs already fitting may remain unedited. No unconditional paragraph rewrite is added.',
        audio_allowance_per_speech=2.,max_audio_requests_per_speech=64,max_text_requests_per_speech=48,
        automatic_transport_retries=False,source_digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tts_source_sha256=hashlib.sha256((ROOT/'src/tts_streaming.py').read_bytes()).hexdigest())
    base.save(RECORD,record)
    meter=BudgetedClient(LEDGER,cap=base.GLOBAL_CAP)
    base.arm(meter.db,base.PARENT,50.)
    base.arm(meter.db,RUN+'/',12.)
    for p in POLICIES:base.arm(meter.db,RUN+'/'+p+'/',6.)
    meter.db.close()
    print(json.dumps({k:v for k,v in record.items() if k not in ('cases','schedule','output_config')},ensure_ascii=False,indent=2))


def execute():
    record=json.loads(RECORD.read_text())
    if record['status']!='approved' or record['source_digest']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():raise ValueError('Source changed')
    if record['tts_source_sha256']!=hashlib.sha256((ROOT/'src/tts_streaming.py').read_bytes()).hexdigest():raise ValueError('Native TTS source changed')
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY']=json.loads((ROOT/'src/configs/api_key.json').read_text())['OPENAI_API_KEY']
    # Warm local estimator before timing, without network calls.
    import tts_streaming
    tts_streaming.duration_estimator(config()).query_time('Warm the local speech duration estimator.')
    OUT.mkdir(exist_ok=False)
    base.save(OUT/'manifest.json',record)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'tts_source_snapshot.py').write_bytes((ROOT/'src/tts_streaming.py').read_bytes())
    count=0
    try:
        for row in record['schedule']:
            case=next(c for c in record['cases'] if c['original_call_id']==row['case'])
            run_one(record,row['policy'],case,row['repeat']);count+=1
            base.save(OUT/'progress.json',dict(status='running',completed=count,total=len(record['schedule'])))
        base.save(OUT/'progress.json',dict(status='completed',completed=count,total=len(record['schedule'])))
    except BaseException as exc:
        base.save(OUT/'progress.json',dict(status='stopped',completed=count,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.execute:execute()
    else:prepare()
