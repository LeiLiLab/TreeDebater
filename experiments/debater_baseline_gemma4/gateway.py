"""Experiment-only, serial, fail-closed API gateway with durable reservations."""
import json
import os
from pathlib import Path
import sqlite3
import time
import threading
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
import uvicorn

ROOT = Path(__file__).resolve().parent
MODEL = 'google.gemma-4-26b-a4b'
CAP = 200.0
app = FastAPI()
lock = threading.Lock()
keys = json.loads((ROOT.parents[1] / 'src/configs/api_key.json').read_text())
db = sqlite3.connect(ROOT / 'cost.sqlite', check_same_thread=False)
db.execute('PRAGMA journal_mode=WAL')
db.execute('CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, kind TEXT, label TEXT, bound REAL, charged REAL, estimate REAL, status TEXT, seconds REAL, input_tokens INTEGER, output_tokens INTEGER)')
db.commit()
label = 'setup'

@app.get('/status')
def status():
    with lock:
        rows = db.execute('SELECT kind,count(*),sum(charged),sum(estimate) FROM calls GROUP BY kind').fetchall()
        return {'cap_usd': CAP, 'exposure_usd': sum(x[2] for x in rows), 'by_kind': rows,
                'stopped': (ROOT / 'STOP').exists()}

@app.post('/label')
async def set_label(request: Request):
    global label
    label = (await request.json())['label']
    return {'label': label}

@app.post('/{path:path}')
async def forward(path: str, request: Request):
    body = await request.json()
    kind = path.rsplit('/', 1)[-1]
    headers = {}
    if path == 'v1/chat/completions':
        if body.get('model') != MODEL or body.get('stream'):
            raise HTTPException(400, 'Only the approved nonstreaming Gemma model is allowed')
        body['max_tokens'] = min(body.get('max_tokens') or 4096, 4096)
        body['num_retries'] = 0
        body.pop('seed', None)
        raw = json.dumps(body).encode()
        bound = 4 * ((len(raw) + 8192) + body['max_tokens']) / 1e6
        url = 'http://127.0.0.1:4000/v1/chat/completions'
    elif path == 'v1/audio/speech':
        if body.get('model') != 'tts-1':
            raise HTTPException(400, 'Unpriced audio model')
        bound = 4 * len(body['input'].encode()) * 15 / 1e6
        url = 'https://api.openai.com/v1/audio/speech'
        headers['Authorization'] = 'Bearer ' + keys['OPENAI_API_KEY']
    elif path == 'search':
        bound = 4 * .016
        url = 'https://api.tavily.com/search'
        body['api_key'] = keys.get('TVLY_API_KEY') or keys.get('TAVILY_API_KEY')
    else:
        raise HTTPException(400, 'Unpriced endpoint blocked')
    with lock:
        used = db.execute('SELECT coalesce(sum(charged),0) FROM calls').fetchone()[0]
        if (ROOT / 'STOP').exists() or used + bound > CAP:
            (ROOT / 'STOP').touch()
            raise HTTPException(402, 'Experiment stopped; budget unavailable')
        rid = db.execute('INSERT INTO calls(kind,label,bound,charged,status) VALUES(?,?,?,?,?)',
                         (kind,label,bound,bound,'pending')).lastrowid
        db.commit()
    t0 = time.perf_counter()
    # No retry here. Every caller retry must acquire its own durable reservation.
    try:
        r = requests.post(url, json=body, headers=headers, timeout=600)
        r.raise_for_status()
        i = o = None
        if kind == 'completions':
            result = r.json()
            usage = result['usage']
            i, o = usage['prompt_tokens'], usage['completion_tokens']
            estimate = (i * .13 + o * .4) / 1e6
            audit = {'id':rid, 'label':label, 'request':body, 'response':result}
            (ROOT / 'requests').mkdir(exist_ok=True)
            (ROOT / 'requests' / f'{rid:06d}.json').write_text(json.dumps(audit))
        elif kind == 'speech':
            estimate = len(body['input']) * 15 / 1e6
        else:
            estimate = .016
        with lock:
            db.execute('UPDATE calls SET charged=?,estimate=?,status=?,seconds=?,input_tokens=?,output_tokens=? WHERE id=?',
                       (estimate,estimate,'ok',time.perf_counter()-t0,i,o,rid))
            db.commit()
        return Response(r.content, media_type=r.headers.get('Content-Type'))
    except Exception as exc:
        with lock:
            db.execute('UPDATE calls SET status=?,seconds=? WHERE id=?', ('error',time.perf_counter()-t0,rid))
            db.commit()
        raise HTTPException(502, f'{type(exc).__name__}: upstream request failed; reservation retained')

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4087)
