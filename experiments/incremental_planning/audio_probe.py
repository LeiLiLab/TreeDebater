"""Bounded server audio probe: paced synthetic recording -> real ASR -> engine -> TTS.

This measures the server's first playable chunk, NOT browser playback or microphone
transport. Both policies receive exactly the same real ASR results at the same time.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default as email_policy
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time

import httpx
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "debate-app/backend"))
from streaming.experiment_client import BudgetedClient, BudgetExceeded
from streaming.experiment_accounting import accounted_exposure, initialize_accounting
from scripts.benchmark_incremental_planning import make_player, atomic_json, code_digest


class AudioGuard(httpx.BaseTransport):
    """Reserve a bounded sub-budget in the shared ledger before any external audio call.

    Per-request 4x upper bounds are durably recorded inside that bundle before HTTP
    dispatch. A provider error latches the transport closed; built-in TTS retries
    cannot cause further requests. Credentials/headers/audio bytes are not logged.
    """
    def __init__(self, directory, label, inner=None, allowance=1.0, *, approved_cap=200.0, max_requests=24):
        if not 0 < allowance <= approved_cap:
            raise ValueError("Audio allowance must be positive and within the approved cumulative cap")
        if type(max_requests) is not int or not 1 <= max_requests <= 128:
            raise ValueError("Audio request limit must be an integer within [1, 128]")
        self.allowance = allowance
        self.max_requests = max_requests
        self.directory = Path(directory)
        self.label = label
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.directory / "cost.sqlite", timeout=30, check_same_thread=False)
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.failed = False
        self.finished = False
        initialize_accounting(self.db)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cap = self.db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            used = accounted_exposure(self.db)
            if cap != approved_cap or used + allowance > cap:
                raise BudgetExceeded("No room for the audio sub-budget")
            self.request_id = self.db.execute(
                "INSERT INTO calls(label,created,reserved,state) VALUES(?,?,?,'pending')",
                (label, datetime.now(timezone.utc).isoformat(), allowance)).lastrowid
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.started = time.perf_counter()
        self.artifact = {"label": label, "request": {"model": "bounded-audio-bundle"},
                         "reservation_usd": allowance, "max_requests": max_requests,
                         "external_calls": [], "blocked_dispatches": []}
        self.path = self.directory / f"call_{self.request_id:06}.json"
        self.persist()

    def persist(self):
        atomic_json(self.path, self.artifact)

    def handle_request(self, request):
        if request.url.host != "api.openai.com":
            raise ValueError("Audio probe only permits the fixed OpenAI audio endpoint")
        body = request.read()
        if request.url.path == "/v1/audio/speech":
            data = json.loads(body)
            if data["model"] != "tts-1" or not 0 < len(data["input"]) <= 4096:
                raise ValueError("Unexpected TTS model/size")
            estimate = len(data["input"]) * 15 / 1e6
            bound = len(data["input"].encode()) * 15 / 1e6
            detail = {"model": "tts-1", "characters": len(data["input"])}
        elif request.url.path == "/v1/audio/transcriptions":
            mime = BytesParser(policy=email_policy).parsebytes(
                ("Content-Type: " + request.headers["content-type"] + "\r\nMIME-Version: 1.0\r\n\r\n").encode() + body)
            parts = {p.get_param("name", header="content-disposition"): p.get_payload(decode=True)
                     for p in mime.iter_parts()}
            if parts.get("model") != b"whisper-1":
                raise ValueError("Unexpected ASR model")
            seconds = len(AudioSegment.from_file(BytesIO(parts["file"]))) / 1000
            if not 0 < seconds <= 120:
                raise ValueError("ASR input outside the prepared duration bound")
            estimate = math.ceil(seconds) / 60 * .006
            # Reserve the complete decoded upload, rounded up to a second.
            # The same 4x margin is applied below, including unknown responses.
            bound = estimate
            detail = {"model": "whisper-1", "seconds": seconds}
        else:
            raise ValueError("Unbudgeted endpoint blocked: " + request.url.path)
        with self.lock:
            if self.finished:
                raise BudgetExceeded('Audio bundle already finished; no dispatch')
            used = sum(c["reserved_usd"] for c in self.artifact["external_calls"])
            if self.failed or len(self.artifact["external_calls"]) >= self.max_requests or used + 4*bound > self.allowance:
                self.artifact['blocked_dispatches'].append(dict(
                    failed_latch=self.failed, requests_used=len(self.artifact['external_calls']),
                    requested_bound_usd=4*bound, already_reserved_usd=used))
                self.persist()
                raise BudgetExceeded("Audio transport closed or its bounded/request-count sub-budget exhausted")
            entry = dict(detail, reserved_usd=4*bound, estimated_usd=estimate, state="pending")
            self.artifact["external_calls"].append(entry)
            self.persist()
        t0 = time.perf_counter()
        try:
            response = self.inner.handle_request(request)
            response.read()
            if response.status_code >= 400:
                # Preserve the diagnostic status without logging credentials or
                # provider response bodies. The SDK may wrap this exception.
                with self.lock:
                    entry['http_status'] = response.status_code
                    try:
                        details = response.json().get('error', {})
                        for field in ('code', 'type'):
                            value = details.get(field)
                            if (isinstance(value, str) and len(value) <= 80
                                    and value.replace('_', '').isalnum()):
                                entry['provider_error_' + field] = value
                    except (ValueError, AttributeError, TypeError):
                        pass
                raise RuntimeError("Audio HTTP status " + str(response.status_code))
            with self.lock:
                entry.update(state="ok", elapsed_seconds=time.perf_counter()-t0,
                             response_bytes=len(response.content))
                self.persist()
            return response
        except BaseException as exc:
            with self.lock:
                self.failed = True
                entry.update(state="error", error=type(exc).__name__, elapsed_seconds=time.perf_counter()-t0)
                self.persist()
            raise

    def finish(self):
        with self.lock:
            if self.finished:
                return
            calls = self.artifact["external_calls"]
            if any(c['state'] == 'pending' for c in calls):
                raise RuntimeError('Cannot finish an audio bundle with in-flight requests')
            self.artifact["seconds"] = time.perf_counter()-self.started
            estimate = sum(c["estimated_usd"] for c in calls if c["state"] == "ok")
            self.db.execute("UPDATE calls SET state=?,estimated_usd=?,seconds=? WHERE id=?",
                            ("error" if any(c["state"] != "ok" for c in calls) else "ok",
                             estimate, self.artifact["seconds"], self.request_id))
            self.db.commit()
            self.persist()
            self.finished = True
        self.inner.close()

    def close(self):
        # SDK client contexts can close independently; the bundle owns the transport.
        pass


def main():
    raise SystemExit('Archived comparison includes a retired mode; use its saved source snapshot for reproduction.')


if __name__ == "__main__":
    main()
