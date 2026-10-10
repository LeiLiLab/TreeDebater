"""Paired text/TTS experiment; actual API timings, simulated playback cursor.

No production authoring changes. All paid requests use durable reservations.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from streaming.experiment_client import BudgetedClient, BudgetExceeded
from streaming.experiment_accounting import MODEL_RATES, exposure_query, accounted_exposure, reconcile_success
from audio_probe import AudioGuard

D = ROOT / 'experiments/incremental_planning'
LEDGER = D / 'run'
PARENT = 'listening-motion-live-v52-refactor'
RUN = PARENT + '-revision-pipelines-v56'
OUT = LEDGER / RUN
RECORD = D / 'prepared_revision_pipelines_v56.json'
GLOBAL_CAP, PARENT_CAP, RUN_CAP = 370., 50., 10.
POLICIES = ('whole_stream', 'paragraph_calls')
MODEL = 'google.gemma-4-26b-a4b'


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def guard_sql(prefix, cap):
    name = 'study_guard_' + hashlib.sha256(prefix.encode()).hexdigest()[:16]
    literal = "'" + prefix.replace("'", "''") + "'"
    exposure = exposure_query(f' WHERE instr(c.label,{literal})=1')
    return name, (f'CREATE TRIGGER {name} BEFORE INSERT ON calls '
        f'WHEN instr(NEW.label,{literal})=1 AND NEW.reserved + ({exposure}) > {float(cap)!r} '
        "BEGIN SELECT RAISE(ABORT, 'Study budget exceeded; no dispatch'); END")


def arm(db, prefix, cap):
    name, sql = guard_sql(prefix, cap)
    old = db.execute("select sql from sqlite_master where type='trigger' and name=?", (name,)).fetchone()
    if old and old[0] != sql:
        raise ValueError('Unexpected existing stop; no automatic increase')
    if not old:
        db.execute(sql)
    db.commit()


def raise_authorized_parent_cap(db, record):
    name, new = guard_sql(PARENT, PARENT_CAP)
    old = db.execute("select sql from sqlite_master where type='trigger' and name=?", (name,)).fetchone()
    expected = guard_sql(PARENT, 20.)[1]
    if old and old[0] == new:
        return
    if not old or old[0] != expected:
        raise ValueError('Original USD20 parent guard differs; do not replace it')
    if record['cumulative_cap_usd'] != 50 or not record['authorization']:
        raise ValueError('Missing explicit USD50 authorization')
    db.execute('BEGIN IMMEDIATE')
    try:
        db.execute(f'DROP TRIGGER {name}')
        db.execute(new)
        db.commit()
    except BaseException:
        db.rollback()
        raise


class ParagraphStream:
    """Incrementally emit only paragraphs delimited by a blank line or EOF."""
    def __init__(self, emit):
        self.pending = ''
        self.emit = emit

    def feed(self, delta):
        self.pending += delta
        while True:
            match = re.search(r'\r?\n[ \t]*\r?\n', self.pending)
            if match is None:
                return
            part, self.pending = self.pending[:match.start()].strip(), self.pending[match.end():]
            if part:
                self.emit(part)

    def finish(self):
        if self.pending.strip():
            self.emit(self.pending.strip())
        self.pending = ''


def stream_complete(client, messages, on_delta, *, max_tokens=4096, temperature=.3):
    """Meter SSE including a final usage chunk; keep partial failures reserved."""
    body = dict(model=MODEL, messages=messages, temperature=temperature, max_tokens=max_tokens,
                stream=True, stream_options={'include_usage': True}, num_retries=0)
    raw = json.dumps(body, ensure_ascii=False).encode()
    reservation = 4 * (len(raw) + 8192 + max_tokens) / 1e6
    db = client.db
    db.execute('BEGIN IMMEDIATE')
    try:
        if accounted_exposure(db) + reservation > GLOBAL_CAP:
            raise BudgetExceeded('Global streaming reservation would exceed cap')
        ident = db.execute("INSERT INTO calls(label,created,reserved,state) VALUES(?,?,?,'pending')",
            (client.label, datetime.now(timezone.utc).isoformat(), reservation)).lastrowid
        db.commit()
    except BaseException:
        db.rollback()
        raise
    started = time.perf_counter()
    artifact = dict(request=body, label=client.label, reservation_usd=reservation,
        rates_per_million=MODEL_RATES[MODEL], request_timeout_seconds=60, total_deadline_seconds=120,
        stream_events=[])
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DEBATE_LLM_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['DEBATE_LLM_API_KEY']
    pieces, usage, finish, response_model = [], None, None, MODEL
    it = ot = cost = None
    try:
        with urlopen(Request(client.base_url+'/chat/completions', data=raw, headers=headers), timeout=60) as response:
            if 'text/event-stream' not in response.headers.get('Content-Type', ''):
                raise ValueError('Endpoint did not return SSE; no buffered fallback allowed')
            event = []
            for raw_line in response:
                if time.perf_counter() - started > 120:
                    raise TimeoutError('Streaming request exceeded total deadline')
                line = raw_line.decode('utf-8').rstrip('\r\n')
                if line.startswith('data:'):
                    event.append(line[5:].lstrip())
                if line or not event:
                    continue
                data = '\n'.join(event)
                event = []
                if data == '[DONE]':
                    break
                packet = json.loads(data)
                if 'error' in packet:
                    raise RuntimeError('SSE provider returned an error')
                response_model = packet.get('model', response_model)
                if packet.get('usage'):
                    usage = packet['usage']
                for choice in packet.get('choices', []):
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
                    delta = choice.get('delta', {}).get('content') or ''
                    if delta:
                        if not isinstance(delta, str):
                            raise ValueError('Non-text streaming delta')
                        pieces.append(delta)
                        artifact['stream_events'].append(dict(seconds=time.perf_counter()-started, delta=delta))
                        on_delta(delta)
        text = ''.join(pieces)
        artifact['response'] = dict(model=response_model, choices=[dict(message=dict(content=text),finish_reason=finish)],usage=usage or {})
        if usage:
            it, ot = usage.get('prompt_tokens'), usage.get('completion_tokens')
            if type(it) is int and type(ot) is int and min(it,ot) >= 0:
                cost = (it*.13 + ot*.4)/1e6
        if not text.strip() or finish != 'stop':
            raise ValueError('Empty, truncated, or unfinished streamed completion')
        if cost is None:
            raise ValueError('Stream usage missing; retain reservation and stop')
        db.execute("UPDATE calls SET state='ok',input_tokens=?,output_tokens=?,estimated_usd=?,seconds=? WHERE id=?",
            (it,ot,cost,time.perf_counter()-started,ident))
        db.commit()
        return text, ident, artifact['stream_events'][0]['seconds']
    except BaseException as exc:
        artifact['error'] = f'{type(exc).__name__}: {exc}'
        artifact['partial_text'] = ''.join(pieces)
        db.execute("UPDATE calls SET state='error',input_tokens=?,output_tokens=?,estimated_usd=?,seconds=? WHERE id=?",
            (it,ot,cost,time.perf_counter()-started,ident))
        db.commit()
        raise
    finally:
        artifact['seconds'] = time.perf_counter()-started
        path = LEDGER/f'call_{ident:06}.json'
        save(path,artifact)
        if 'error' not in artifact:
            reconcile_success(db,ident,path)


def speech_pieces(text, limit=3500):
    """Same sentence-boundary TTS size guard for both policies, no rewriting."""
    if len(text) <= limit:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+',text)
    result, current = [], ''
    for sentence in sentences:
        if len(sentence) > limit:
            raise ValueError('A sentence exceeds TTS size bound')
        if current and len(current)+len(sentence)+1 > limit:
            result.append(current)
            current = ''
        current = (current+' '+sentence).strip()
    if current:
        result.append(current)
    return result


def paragraph_request(case, index, committed, actual_seconds):
    data = copy.deepcopy(case['material'])
    originals = case['original_paragraphs']
    remaining = case['body_budget_seconds'] - actual_seconds
    if remaining < 2:
        raise ValueError('No time remains for an unspoken argument; stop without dropping it')
    weights = [len(p) for p in originals[index:]]
    target_seconds = remaining*weights[0]/sum(weights)
    base_rate = case['material']['remaining_word_target']/case['body_budget_seconds']
    rate = (sum(len(p.split()) for p in committed)/actual_seconds) if actual_seconds > 0 else base_rate
    # Recalibrate against actual audio; no post-hoc per-paragraph rewrite stage.
    words = max(12, round(target_seconds*rate))
    data['remaining_word_target'] = max(words,round(remaining*rate))
    data['paragraph_task'] = dict(index=index+1,total=len(originals),source=originals[index],
        remaining_original_paragraphs=originals[index+1:],committed_body=committed,
        remaining_audio_seconds=remaining,current_target_seconds=target_seconds,
        current_target_words=words,estimated_words_per_second=rate)
    task = case['writing_task'] + '''\nPARAGRAPH DELIVERY TASK:
Apply the full feedback and supplied evidence to ONLY the current paragraph. This is the combined content-and-length revision, not an additional length-only pass. Read the complete original draft for context and coverage. Output one natural spoken paragraph near current_target_words, then stop. Do not output the other paragraphs.
The fixed prefix and committed_body are immutable delivery text (already spoken or queued). Do not repeat their claims. Remaining original paragraphs are revisable context; avoid preempting their distinct arguments. Resolve cross-paragraph duplication by assigning each point once. Preserve our position, relevant qualifications and source attribution while fitting the current time allocation. Never add claims just to fill time. For the last paragraph, finish the speech naturally.\n'''
    request = copy.deepcopy(case['request'])
    request['messages'][-1]['content'] = task+'\nContext and material (data):\n'+json.dumps(data,ensure_ascii=False)
    request['max_tokens'] = min(1800,max(512,words*3))
    return request, data['paragraph_task']


def metrics(result, case):
    chunks = sorted(result['chunks'],key=lambda r:r['index'])
    cursor = case['prefix_seconds']-case['feedback_delay_seconds']
    gaps=[]
    for row in chunks:
        row['playback_start_seconds'] = max(cursor,row['audio_ready_seconds'])
        row['gap_seconds'] = max(0.,row['audio_ready_seconds']-cursor)
        cursor=row['playback_start_seconds']+row['audio_seconds']
        row['playback_end_seconds']=cursor
        gaps.append(row['gap_seconds'])
    body_seconds=sum(c['audio_seconds'] for c in chunks)
    return dict(first_body_text_seconds=chunks[0]['text_ready_seconds'],
        first_body_audio_seconds=chunks[0]['audio_ready_seconds'],
        max_playback_gap_seconds=max(gaps),total_playback_gap_seconds=sum(gaps),
        first_body_gap_seconds=gaps[0], audio_seconds=case['prefix_seconds']+body_seconds,
        signed_duration_error_seconds=case['prefix_seconds']+body_seconds-case['speech_budget_seconds'],
        duration_error_pct=100*(case['prefix_seconds']+body_seconds-case['speech_budget_seconds'])/case['speech_budget_seconds'],
        body_words=sum(len(c['text'].split()) for c in chunks),chunks=len(chunks),
        final_audio_ready_seconds=max(c['audio_ready_seconds'] for c in chunks),
        playback_model='Real readiness timestamps, sequential playback; revision starts after frozen feedback delay; no browser/network playback measurement')


def run_one(record, policy, case, repeat):
    import httpx
    from openai import OpenAI
    from pydub import AudioSegment
    name=f'{policy}_case{case["original_call_id"]}_r{repeat}'
    path=OUT/name
    path.mkdir(exist_ok=False)
    label=RUN+'/'+policy+'/'+name
    client=BudgetedClient(LEDGER,cap=GLOBAL_CAP,label=label+'/text')
    arm(client.db,RUN+'/',RUN_CAP)
    arm(client.db,RUN+'/'+policy+'/',5.)
    guard=AudioGuard(LEDGER,label+'/tts',allowance=1.6,approved_cap=GLOBAL_CAP,max_requests=8)
    tts=OpenAI(http_client=httpx.Client(transport=guard,timeout=60),max_retries=0,
        base_url='https://api.openai.com/v1',timeout=60)
    result=dict(status='running',policy=policy,case=case['original_call_id'],repeat=repeat,chunks=[],text_calls=[],
        original_prefix=case['material']['already_spoken_prefix'],case_sha256=digest(case),
        prefix_seconds=case['prefix_seconds'],feedback_delay_seconds=case['feedback_delay_seconds'])
    save(path/'result.json',result)
    cancel=threading.Event()
    started=time.perf_counter()
    futures=[]
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='v56-tts')
    lock=threading.Lock()

    def synthesize(index,text,text_ready):
        if cancel.is_set():
            raise RuntimeError('Experiment stopped before queued TTS dispatch')
        t0=time.perf_counter()
        raw=tts.audio.speech.create(model='tts-1',voice='echo',input=text,response_format='mp3').content
        audio=AudioSegment.from_file(BytesIO(raw),format='mp3')
        seconds=len(audio)/1000
        if seconds <= 0:
            raise ValueError('Empty TTS audio')
        (path/f'chunk_{index:02}.mp3').write_bytes(raw)
        (path/f'chunk_{index:02}.txt').write_text(text+'\n')
        row=dict(index=index,text=text,text_ready_seconds=text_ready,tts_start_seconds=t0-started,
            tts_seconds=time.perf_counter()-t0,audio_ready_seconds=time.perf_counter()-started,
            audio_seconds=seconds,path=str(path/f'chunk_{index:02}.mp3'))
        with lock:
            result['chunks'].append(row)
        return row

    def emit(paragraph):
        for part in speech_pieces(paragraph):
            if len(futures)>=8:
                raise ValueError('More than eight TTS blocks; stop instead of unbounded calls')
            futures.append(pool.submit(synthesize,len(futures),part,time.perf_counter()-started))

    try:
        if policy=='whole_stream':
            request=copy.deepcopy(case['request'])
            request['messages'][-1]['content'] += '\nSeparate complete spoken paragraphs with a blank line. Output them in final delivery order; do not revisit earlier paragraphs.'
            save(path/'request_00.json',request)
            parser=ParagraphStream(emit)
            text,ident,ttft=stream_complete(client,request['messages'],parser.feed,
                max_tokens=request['max_tokens'],temperature=request['temperature'])
            parser.finish()
            result['text_calls'].append(dict(ledger_id=ident,first_token_seconds=ttft,
                done_seconds=time.perf_counter()-started))
            result['text']=text
            for f in futures:
                f.result()
        else:
            committed=[]
            actual_seconds=0.
            for index in range(len(case['original_paragraphs'])):
                request,task=paragraph_request(case,index,committed,actual_seconds)
                save(path/f'request_{index:02}.json',dict(request=request,paragraph_task=task))
                t0=time.perf_counter()
                client.label=label+f'/text_{index}'
                text=client.complete(request['messages'],model=MODEL,temperature=request['temperature'],
                    max_tokens=request['max_tokens'],request_timeout=60)
                ident=client.db.execute('select max(id) from calls where label=?',(client.label,)).fetchone()[0]
                receipt=json.loads((LEDGER/f'call_{ident:06}.json').read_text())
                if receipt.get('truncated'):
                    raise ValueError('Paragraph output truncated; do not publish')
                result['text_calls'].append(dict(ledger_id=ident,seconds=time.perf_counter()-t0,
                    done_seconds=time.perf_counter()-started,target=task))
                if case['material']['already_spoken_prefix'] in text:
                    raise ValueError('Paragraph repeated immutable prefix')
                count=len(futures)
                emit(text.strip())
                for f in futures[count:]:
                    actual_seconds+=f.result()['audio_seconds']
                committed.append(text.strip())
            result['text']='\n\n'.join(committed)
        result['metrics']=metrics(result,case)
        result['status']='completed'
        (path/'body.txt').write_text(result['text']+'\n')
        audio=AudioSegment.from_file(case['prefix_audio'])
        for row in sorted(result['chunks'],key=lambda r:r['index']):
            audio+=AudioSegment.silent(duration=round(row['gap_seconds']*1000))
            audio+=AudioSegment.from_file(row['path'])
        audio.export(path/'playback.wav',format='wav')
    except BaseException as exc:
        cancel.set()
        result.update(status='stopped',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        cancel.set()
        pool.shutdown(wait=True,cancel_futures=True)
        guard.finish()
        tts.close()
        reconcile_success(client.db,guard.request_id,guard.path)
        guard.db.close()
        result['wall_seconds']=time.perf_counter()-started
        save(path/'result.json',result)
        client.db.close()
    print(json.dumps(dict(run=name,**result['metrics']),ensure_ascii=False),flush=True)
    return result


def prepare():
    if RECORD.exists():
        raise ValueError('Prepared record exists')
    source=json.loads((D/'prepared_evidence_iteration_v55.json').read_text())
    turn_dirs={22936:'01_opening_against',22963:'02_rebuttal_for',22990:'03_rebuttal_against'}
    cases=[]
    for c in source['cases']:
        if c['variant']!='B_evidence_rewrite':continue
        req=copy.deepcopy(c['request'])
        prompt=req['messages'][-1]['content']
        task,data=prompt.split('\nContext and material (data):\n',1)
        material=json.loads(data)
        directory=LEDGER/(PARENT+'-evidence-v53')/turn_dirs[c['original_call_id']]
        turn=json.loads((directory/'result.json').read_text())
        trace=json.loads(next(directory.glob('*_chunks/listening_prefix.json')).read_text())
        prefix=turn['chunks'][0]
        assert prefix['text']==material['already_spoken_prefix']
        prefix_path=ROOT/prefix['path']
        assert prefix_path.is_file()
        cases.append(dict(original_call_id=c['original_call_id'],stage=c['stage'],side=c['side'],request=req,
            writing_task=task,material=material,original_paragraphs=re.split(r'\n\s*\n',material['unpublished_draft'].strip()),
            prefix_audio=str(prefix_path),prefix_seconds=prefix['duration_seconds'],
            speech_budget_seconds=turn['target_seconds'],body_budget_seconds=turn['target_seconds']-prefix['duration_seconds'],
            feedback_delay_seconds=trace['parallel_body_feedback']['end_seconds']-trace['parallel_body_feedback']['start_seconds']))
    # Alternate order in each paired repeat to reduce simple time-of-run confounding.
    schedule=[]
    for repeat in (1,2):
        for ci,case in enumerate(cases):
            order=POLICIES if (ci+repeat)%2 else POLICIES[::-1]
            for policy in order:
                schedule.append(dict(policy=policy,case=case['original_call_id'],repeat=repeat))
    client=BudgetedClient(LEDGER,cap=GLOBAL_CAP)
    before=client.db.execute(exposure_query(' WHERE instr(c.label,?)=1'),(PARENT,)).fetchone()[0]
    record=dict(status='approved',created_utc=datetime.now(timezone.utc).isoformat(),run_id=RUN,
        authorization='User: 你针对方案一和方案二分别进行实验验证。在我之前给你批过的预算再加上三十刀，也就是总共五十刀。',
        cumulative_cap_usd=50,previous_parent_cap_usd=20,parent_prefix=PARENT,
        run_cap_usd=10,policy_caps_usd={p:5 for p in POLICIES},prior_conservative_exposure_usd=before,
        estimate_new_usd=[1,3],model=MODEL,temperature=.3,tts_model='tts-1',voice='echo',
        rates_per_million=dict(input=.13,output=.4,tts_characters=15),
        price_sources=['https://aws.amazon.com/bedrock/pricing/','https://developers.openai.com/api/docs/models/tts-1'],
        maximum_text_requests=sum(1 if r['policy']=='whole_stream' else len(next(c for c in cases if c['original_call_id']==r['case'])['original_paragraphs']) for r in schedule),
        maximum_audio_requests=96,automatic_retries=False,new_asr=False,new_retrieval=False,
        controls='Same original drafts/feedback/evidence/system/history/temp/TTS; no post-hoc length rewrites in either policy; one TTS worker each; same fixed prefix and historical feedback delay.',
        boundaries='Three cases from one motion, two repeats each. Real text/TTS requests; playback reconstructed from readiness and decoded durations, no live opponent or browser. Policy2 can use prior actual TTS durations; policy1 cannot feed them into an in-flight request. Policy2 original paragraph assignment is fixed.',
        schedule=schedule,cases=cases,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(RECORD,record)
    raise_authorized_parent_cap(client.db,record)
    arm(client.db,RUN+'/',RUN_CAP)
    for p in POLICIES:arm(client.db,RUN+'/'+p+'/',5.)
    arm(client.db,'listening-motion-live-',280.)
    client.db.close()
    print(json.dumps({k:v for k,v in record.items() if k not in ('cases','schedule')},ensure_ascii=False,indent=2))


def execute():
    record=json.loads(RECORD.read_text())
    if record['status']!='approved' or record['script_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('Prepared configuration changed')
    # Load the existing authorized audio credential without logging its value.
    if not os.environ.get('OPENAI_API_KEY'):
        keys = json.loads((ROOT/'src/configs/api_key.json').read_text())
        os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY']
    if not os.environ['OPENAI_API_KEY']:
        raise ValueError('Missing audio credential; no reservations or dispatch')
    OUT.mkdir(exist_ok=False)
    save(OUT/'manifest.json',record)
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    try:
        for row in record['schedule']:
            case=next(c for c in record['cases'] if c['original_call_id']==row['case'])
            results.append(run_one(record,row['policy'],case,row['repeat']))
            save(OUT/'progress.json',dict(status='running',completed=len(results),total=len(record['schedule'])))
        save(OUT/'progress.json',dict(status='completed',completed=len(results),total=len(record['schedule'])))
    except BaseException as exc:
        save(OUT/'progress.json',dict(status='stopped',completed=len(results),error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:execute()
    else:prepare()
