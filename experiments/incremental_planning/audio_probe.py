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
from scripts.benchmark_incremental_planning import make_player, atomic_json, code_digest


class AudioGuard(httpx.BaseTransport):
    """Reserve a $1 sub-budget in the shared ledger before any external audio call.

    Per-request 4x upper bounds are durably recorded inside that bundle before HTTP
    dispatch. A provider error latches the transport closed; built-in TTS retries
    cannot cause further requests. Credentials/headers/audio bytes are not logged.
    """
    def __init__(self, directory, label, inner=None):
        self.directory = Path(directory)
        self.label = label
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.directory / "cost.sqlite", timeout=30, check_same_thread=False)
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.failed = False
        self.db.execute("BEGIN IMMEDIATE")
        try:
            cap = self.db.execute("SELECT cap FROM budget WHERE id=1").fetchone()[0]
            used = self.db.execute("SELECT coalesce(sum(reserved),0) FROM calls").fetchone()[0]
            if cap != 200 or used + 1 > cap:
                raise BudgetExceeded("No room for the fixed $1 audio sub-budget")
            self.request_id = self.db.execute(
                "INSERT INTO calls(label,created,reserved,state) VALUES(?,?,1,'pending')",
                (label, datetime.now(timezone.utc).isoformat())).lastrowid
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        self.started = time.perf_counter()
        self.artifact = {"label": label, "request": {"model": "bounded-audio-bundle"},
                         "reservation_usd": 1, "external_calls": []}
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
            bound = .012  # reserve the entire permitted 120 seconds before the 4x margin
            detail = {"model": "whisper-1", "seconds": seconds}
        else:
            raise ValueError("Unbudgeted endpoint blocked: " + request.url.path)
        with self.lock:
            used = sum(c["reserved_usd"] for c in self.artifact["external_calls"])
            if self.failed or len(self.artifact["external_calls"]) >= 24 or used + 4*bound > 1:
                raise BudgetExceeded("Audio transport closed or its $1/24-request sub-budget exhausted")
            entry = dict(detail, reserved_usd=4*bound, estimated_usd=estimate, state="pending")
            self.artifact["external_calls"].append(entry)
            self.persist()
        t0 = time.perf_counter()
        try:
            response = self.inner.handle_request(request)
            response.read()
            if response.status_code >= 400:
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
            calls = self.artifact["external_calls"]
            self.artifact["seconds"] = time.perf_counter()-self.started
            estimate = sum(c["estimated_usd"] for c in calls if c["state"] == "ok")
            self.db.execute("UPDATE calls SET state=?,estimated_usd=?,seconds=? WHERE id=?",
                            ("error" if any(c["state"] != "ok" for c in calls) else "ok",
                             estimate, self.artifact["seconds"], self.request_id))
            self.db.commit()
            self.persist()
        self.inner.close()

    def close(self):
        # SDK client contexts can close independently; the bundle owns the transport.
        pass


def main():
    import openai
    from debate_app.engine_adapter import TreeDebaterEngine
    from streaming.config import OutputConfig

    run_id = "grounded-audio-v1"
    directory = ROOT / "experiments/incremental_planning"
    output = directory / "run" / run_id
    output.mkdir(exist_ok=False)
    keys = ROOT / "src/configs/api_key.json"
    if keys.exists():
        for key, value in json.loads(keys.read_text()).items():
            os.environ.setdefault(key, value)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Existing OpenAI audio credential unavailable")
    os.environ["DEBATE_LLM_API_BASE"] = "http://127.0.0.1:4000/v1"
    os.environ["DEBATE_LOG_PROMPTS"] = "0"
    BudgetedClient(directory / "run")  # validate existing shared budget
    guard = AudioGuard(directory / "run", run_id + "/external-audio")
    real_openai = openai.OpenAI
    def client_factory(**kwargs):
        kwargs.update(http_client=httpx.Client(transport=guard, timeout=60), max_retries=0,
                      base_url="https://api.openai.com/v1", timeout=60)
        return real_openai(**kwargs)
    openai.OpenAI = client_factory
    import tts_streaming
    tts_streaming.OpenAI = client_factory
    import litellm
    from utils.tool import logger
    import logging
    logger.setLevel(logging.WARNING)
    def blocked(*args, **kwargs):
        raise RuntimeError("Unmetered model/refinement call blocked by audio probe")
    litellm.completion = blocked
    # All text preparation/generation remains on Gemma; separate TTS rewrites are off.
    tts_streaming._revise_to_n_words = blocked
    cfg = OutputConfig(budget_mode="audio_duration", adaptive_delivery=True,
                       first_chunk_seconds=6, later_chunk_seconds=15, min_chunk_words=1,
                       max_refinements=0, early_max_refinements=0, max_parallel_tts=1,
                       speed_adjust_min=1, speed_adjust_max=1)
    case = next(c for c in json.loads((directory / "cases_v2.json").read_text())
                if c["id"] == "dev_v2_plastics_split")
    modes = ("linear", "light_linear")
    executors = {mode: ThreadPoolExecutor(max_workers=1) for mode in modes}
    engines, events = {}, {mode: [] for mode in modes}
    def prepare(mode):
        client = BudgetedClient(directory / "run", label=run_id + "/" + mode)
        player = make_player(case, mode, client)
        player.config.streaming_tts = True
        player.streaming_output_config = cfg
        engine = TreeDebaterEngine({"model_timeout_seconds": 60, "budgets": {"rebuttal": 60}}, output / mode)
        engine.players = {player.side: player}
        return engine
    try:
        for mode in modes:
            engines[mode] = executors[mode].submit(prepare, mode).result()
        source = []
        with client_factory() as client:
            for i, chunk in enumerate(case["chunks"]):
                result = client.audio.speech.create(model="tts-1", voice="echo", input=chunk)
                path = output / f"input_{i}.mp3"
                path.write_bytes(result.content)
                duration = len(AudioSegment.from_file(path)) / 1000
                source.append((path, duration))
        atomic_json(output / "metadata.json", {"source_digest": code_digest(), "case": case,
                    "modes": modes, "input": "paced synthesized recording, shared real Whisper transcripts",
                    "metric": "server first playable TTS chunk after actual recorded-audio endpoint; no browser transport/playback",
                    "output_settings": vars(cfg), "shared_cap_usd": 200, "audio_subbudget_usd": 1})
        start = time.perf_counter()
        endpoint = start + sum(seconds for _, seconds in source)
        transcripts, arrivals = [], []
        arrival = start
        analysis_futures = []
        for path, seconds in source:
            arrival += seconds
            time.sleep(max(0, arrival-time.perf_counter()))
            text = engines["linear"].transcribe(path)
            if not text:
                raise ValueError("Empty ASR transcript")
            transcripts.append(text)
            arrivals.append({"input_end_seconds": arrival-start, "asr_ready_seconds": time.perf_counter()-start,
                             "transcript": text})
            for mode in modes:
                analysis_futures.append(executors[mode].submit(engines[mode].analyze, text, "for", "opening"))
        history = [{"stage": "opening", "side": "against", "content": case["own_opening"]},
                   {"stage": "opening", "side": "for", "content": " ".join(transcripts), "tree_via_streaming": True}]
        def emit(mode, chunk):
            at = time.perf_counter()
            decoded = AudioSegment.from_file(chunk["path"])
            if len(decoded) == 0:
                raise ValueError("Emitted audio is not playable")
            events[mode].append(dict(chunk, endpoint_to_chunk_seconds=at-endpoint, decoded_ms=len(decoded)))
        futures = {mode: executors[mode].submit(engines[mode].generate, "against", "rebuttal", history,
                                               output / mode, lambda chunk, m=mode: emit(m, chunk)) for mode in modes}
        for future in analysis_futures:
            future.result()
        results = {}
        for mode, future in futures.items():
            generated = future.result(timeout=240)
            if not events[mode]:
                raise ValueError("No first audio chunk emitted")
            results[mode] = {"first_playable_chunk_seconds": events[mode][0]["endpoint_to_chunk_seconds"],
                             "chunks": events[mode], "answer": generated["text"]}
        atomic_json(output / "result.json", {"input_seconds": endpoint-start, "asr_arrivals": arrivals, "results": results})
        print(json.dumps({mode: result["first_playable_chunk_seconds"] for mode, result in results.items()}), flush=True)
    finally:
        for ex in executors.values():
            ex.shutdown(wait=True, cancel_futures=True)
        guard.finish()


if __name__ == "__main__":
    main()
