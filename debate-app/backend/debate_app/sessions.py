import asyncio
import base64
import hashlib
import json
import math
from pathlib import Path
import secrets
import time
import uuid
import wave

from .storage import Storage, atomic_write
from .worker import Worker

TERMINAL = {"completed", "cancelled", "failed"}


def now_ms():
    return time.monotonic() * 1000


def finish_pause(state, at):
    """Account for a pause without charging it to speaking or waiting time."""
    if not state.get("paused"):
        return
    started = state["paused_at_ms"]
    elapsed = max(0, at - started)
    turn = state["turns"][max(0, state["current"])]
    turn["paused_ms"] = turn.get("paused_ms", 0) + elapsed
    if state.pop("pause_extends_deadline", False):
        turn["deadline_ms"] += elapsed
        if "timer_stopped_ms" in turn:
            turn["timer_stopped_ms"] += elapsed
    if "finish_by_ms" in turn:
        turn["finish_by_ms"] += elapsed
    if "waiting_started_ms" in turn:
        end = min(at, turn.get("waiting_ended_ms", at))
        turn["waiting_paused_ms"] = turn.get("waiting_paused_ms", 0) + max(
            0, end - max(started, turn["waiting_started_ms"])
        )
    state["paused"] = False
    state.pop("paused_at_ms", None)


class Conflict(ValueError):
    pass


class CapacityFull(Conflict):
    pass


class Session:
    def __init__(self, settings, storage, worker_factory=Worker, *, session_id=None):
        self.storage, self.worker_factory = storage, worker_factory
        self.worker = None
        self.ai_workers = {}
        self.asr_worker = None
        self.analysis_task = None
        self.lock = asyncio.Lock()
        self.processing = None
        self.deadline_task = None
        self.runner = None
        self.resumed = asyncio.Event()
        self.resumed.set()
        self.controller = None
        self.last_seen = time.monotonic()
        self.created = self.last_seen
        self.tasks = set()
        self.worker_stops = {}
        self.finalizer = None
        self.settings = settings
        self.state = {
            "id": session_id or uuid.uuid4().hex,
            "token": secrets.token_urlsafe(32),
            "config": settings.model_dump(),
            "status": "created",
            "finalized": False,
            "paused": False,
            "preparation_status": "pending",
            "preparation_error": None,
            "turns": [],
            "current": -1,
            "history": [],
            "trees": {},
            "evaluation": None,
            "error": None,
            "revision": 0,
            "created_at": time.time(),
        }
        order = [
            settings.first_side,
            "against" if settings.first_side == "for" else "for",
        ]
        for stage in ["opening", "rebuttal", "closing"]:
            for side in order:
                human = settings.mode == "human_ai" and side == settings.human_side
                self.state["turns"].append(
                    {
                        "id": f"{stage}_{side}",
                        "stage": stage,
                        "side": side,
                        "human": human,
                        "status": "pending",
                        "attempt_id": None,
                        "attempts": [],
                        "rerecords": 0,
                        "epochs": [],
                        "transcripts": [],
                        "chunks": [],
                        "played_ms": 0,
                        "generated_ms": 0,
                        "generation_done": False,
                        "analysis_done": 0,
                        "omissions": False,
                    }
                )
        self.root = storage.root / "sessions" / self.state["id"]
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write(
            self.root / "config.json",
            json.dumps(self.state["config"], indent=2).encode(),
        )
        self.event("session_status")

    def snapshot(self):
        return json.loads(json.dumps({k: v for k, v in self.state.items() if k != "token"}))

    def task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def finalize(self):
        # Background jobs can call fail(), so they must never await this task:
        # it drains those jobs after stopping their model executors.
        if self.finalizer is None:
            self.finalizer = asyncio.create_task(self._finalize())
        return self.finalizer

    async def _finalize(self):
        if self.deadline_task and not self.deadline_task.done():
            self.deadline_task.cancel()
        await self.stop_workers()
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        # In-flight capture/recovery commands may still hold this lock.
        async with self.lock:
            ws = getattr(self, "microphone_socket", None)
            if ws is not None:
                try:
                    await ws.close(code=1000)
                except (RuntimeError, OSError):
                    pass
            self.state["finalized"] = True
            self.event("session_finalized")

    def event(self, kind, data=None):
        # Persist the clock endpoint so finished/stopped timers also survive refresh.
        for turn in self.state["turns"]:
            if self.state["status"] in TERMINAL and "waiting_started_ms" in turn:
                turn.setdefault("waiting_ended_ms", now_ms())
            if turn.get("deadline_ms") and turn["status"] not in ("capturing", "interrupted", "waiting_for_human"):
                turn.setdefault("timer_stopped_ms", min(now_ms(), self.state.get("paused_at_ms", math.inf), turn["deadline_ms"]))
        self.storage.persist(self.state, kind, data or {"status": self.state["status"]})

    def current(self, turn_id=None, attempt_id=None, *, allow_paused=False):
        if self.state["status"] != "running" or self.state["current"] < 0:
            raise Conflict("There is no active turn")
        if self.state.get("paused") and not allow_paused:
            raise Conflict("Debate is paused. Resume the debate first.")
        turn = self.state["turns"][self.state["current"]]
        if turn_id is not None and turn["id"] != turn_id:
            raise Conflict("This command belongs to a different turn")
        if attempt_id is not None and turn["attempt_id"] != attempt_id:
            raise Conflict("This recording attempt is no longer active")
        return turn

    def save_command(self, key, operation, response):
        self.storage.command(self.state["id"], key, operation, response)
        return response

    async def start(self, key):
        async with self.lock:
            old = self.storage.command(self.state["id"], key, "start")
            if old is not None:
                return old
            if self.state["status"] != "created":
                raise Conflict("Session has already started")
            self.state["status"] = "preparing"
            # Include initial AI preparation in the first speaker's waiting time.
            self.state["turns"][0]["waiting_started_ms"] = now_ms()
            self.event("session_status")
            self.state["preparation_status"] = "preparing"
            if self.state["turns"][0]["human"]:
                self.state["status"] = "running"
                await self.advance()
            self.runner = self.task(self.prepare())
            return self.save_command(key, "start", {"status": self.state["status"]})

    async def prepare(self):
        if self.state["status"] in TERMINAL:
            return
        try:
            if self.settings.mode == "ai_ai":
                self.ai_workers = {}
                for side in ("for", "against"):
                    self.ai_workers[side] = self.worker_factory(
                        {**self.settings.model_dump(), "worker_side": side},
                        self.root / "players" / side,
                    )
                self.worker = self.ai_workers[self.settings.first_side]
                prepared = await asyncio.gather(*(w.call("prepare") for w in self.ai_workers.values()))
                trees = {side: tree for result in prepared for side, tree in result.items()}
            else:
                self.worker = self.worker_factory(self.settings.model_dump(), self.root)
                trees = await self.worker.call("prepare")
            if self.state["status"] in TERMINAL:
                await self.stop_workers()
                return
            self.state["trees"] = trees
            self.state["preparation_status"] = "ready"
            self.state["preparation_error"] = None
            self.event("preparation_status")
            if self.state["current"] < 0:
                self.state["status"] = "running"
                await self.advance()
        except Exception as e:
            if self.state["status"] in TERMINAL:
                return
            await self.stop_workers(include_asr=False)
            if self.state["status"] in TERMINAL:
                return
            self.state["preparation_status"] = "failed"
            self.state["preparation_error"] = str(e)
            self.event("preparation_status")

    async def retry_preparation(self, key):
        async with self.lock:
            old = self.storage.command(self.state["id"], key, "retry_preparation")
            if old is not None:
                return old
            if self.state["status"] in TERMINAL or self.state["preparation_status"] != "failed":
                raise Conflict("AI preparation is not awaiting retry")
            if self.state.get("paused"):
                raise Conflict("Resume the debate before retrying preparation")
            self.state["preparation_status"] = "preparing"
            self.state["preparation_error"] = None
            self.event("preparation_status")
            self.runner = self.task(self.prepare())
            return self.save_command(key, "retry_preparation", {"status": "preparing"})

    async def stop_workers(self, include_asr=True):
        workers = set(self.ai_workers.values())
        if self.worker:
            workers.add(self.worker)
        if include_asr and self.asr_worker:
            workers.add(self.asr_worker)
        for worker in workers:
            if worker not in self.worker_stops:
                self.worker_stops[worker] = asyncio.create_task(worker.stop())
        results = await asyncio.gather(
            *(asyncio.shield(self.worker_stops[w]) for w in workers), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def wait_until_resumed(self):
        await self.resumed.wait()
        return self.state["status"] not in TERMINAL

    async def pause(self, key):
        async with self.lock:
            old = self.storage.command(self.state["id"], key, "pause")
            if old is not None:
                return old
            if self.state["status"] not in ("running", "preparing"):
                raise Conflict("Only an active debate can be paused")
            if not self.state.get("paused"):
                at = now_ms()
                turn = self.state["turns"][max(0, self.state["current"])]
                self.state["pause_extends_deadline"] = bool(
                    turn["human"] and turn["status"] in ("capturing", "interrupted")
                    and not turn.get("capture_closed") and turn.get("deadline_ms", 0) > at
                )
                for epoch in turn["epochs"]:
                    if not epoch["closed"]:
                        epoch["closed"] = True
                        epoch["paused"] = turn["status"] == "capturing"
                self.state.update(paused=True, paused_at_ms=at)
                self.resumed.clear()
                self.event("debate_paused")
            return self.save_command(key, "pause", {"paused": True})

    async def resume(self, key):
        async with self.lock:
            old = self.storage.command(self.state["id"], key, "resume")
            if old is not None:
                return old
            if self.state["status"] not in ("running", "preparing"):
                raise Conflict("Only an active debate can be resumed")
            if self.state.get("paused"):
                finish_pause(self.state, now_ms())
                turn = self.state["turns"][max(0, self.state["current"])]
                if turn["human"] and turn["status"] == "capturing" and not turn.get("capture_closed"):
                    # A new browser capture epoch resumes the same attempt.
                    turn["status"] = "interrupted"
                self.resumed.set()
                self.event("debate_resumed")
            return self.save_command(key, "resume", {"paused": False})

    async def fail(self, message):
        if self.state["status"] in TERMINAL:
            return
        finish_pause(self.state, now_ms())
        self.resumed.set()
        self.state["status"], self.state["error"] = "failed", message
        if self.state["current"] >= 0:
            self.state["turns"][self.state["current"]]["status"] = "failed"
        self.event("session_failed", {"message": message})
        self.finalize()
        await self.stop_workers()

    async def stop(self, key, reason=None):
        # Control bypasses the capture/recovery lock, which may be waiting on a model.
        old = self.storage.command(self.state["id"], key, "stop")
        if old is not None:
            await asyncio.shield(self.finalize())
            return old
        finish_pause(self.state, now_ms())
        self.resumed.set()
        if self.state["status"] not in TERMINAL:
            self.state["status"] = "cancelled"
            if reason:
                self.state["error"] = reason
            if self.state["current"] >= 0:
                self.state["turns"][self.state["current"]]["status"] = "cancelled"
            self.event("session_status")
        result = self.save_command(key, "stop", {"status": self.state["status"]})
        await asyncio.shield(self.finalize())
        return result

    async def advance(self):
        if not await self.wait_until_resumed():
            return
        if self.state["status"] in TERMINAL:
            return
        self.state["current"] += 1
        if self.state["current"] == len(self.state["turns"]):
            self.state["current"] -= 1
            self.state["status"] = "completed"
            self.event("session_completed")
            try:
                if self.settings.evaluation:
                    try:
                        self.state["evaluation"] = await self.worker.call("evaluate", history=self.state["history"])
                    except Exception as e:
                        self.state["evaluation"] = "Evaluation unavailable: " + str(e)
                    self.event("evaluation_ready")
            finally:
                self.finalize()
                await self.stop_workers()
            return
        t = self.current()
        t.setdefault("waiting_started_ms", now_ms())
        t["status"] = "waiting_for_human" if t["human"] else "generating"
        self.event("turn_started", {"turn_id": t["id"], "side": t["side"], "stage": t["stage"]})
        if not t["human"]:
            self.processing = self.task(self.generate(t))

    async def begin(self, turn_id, key):
        async with self.lock:
            op = "begin:" + turn_id
            old = self.storage.command(self.state["id"], key, op)
            if old is not None:
                return old
            t = self.current(turn_id)
            if not t["human"] or t["status"] != "waiting_for_human":
                raise Conflict("Microphone capture cannot start in this state")
            if self.state["status"] != "running":
                raise Conflict("Session stopped while preparing capture")
            if self.asr_worker is None or self.asr_worker.closed:
                self.asr_worker = self.worker_factory(self.settings.model_dump(), self.root)
            t.update(
                attempt_id=uuid.uuid4().hex,
                epochs=[],
                transcripts=[],
                omissions=False,
                status="capturing",
                start_ms=now_ms(),
                error=None,
                checkpoint_ready=False,
            )
            t["deadline_ms"] = t["start_ms"] + self.settings.budgets[t["stage"]] * 1000
            t.setdefault("waiting_ended_ms", t["start_ms"])
            t["capture_closed"] = False
            t.pop("timer_stopped_ms", None)
            t.pop("finish_by_ms", None)
            self.event("microphone_status", {"turn_id": t["id"], "status": "capturing"})
            self.processing = self.task(self.listen(t, t["attempt_id"]))
            self.deadline_task = self.task(self.enforce_deadline(t, t["attempt_id"]))
            response = {
                "attempt_id": t["attempt_id"],
                "start_ms": t["start_ms"],
                "deadline_ms": t["deadline_ms"],
            }
            return self.save_command(key, op, response)

    async def enforce_deadline(self, turn, attempt):
        """The speaking clock stays responsive even during a blocked model call."""
        while self.state["status"] == "running" and turn["attempt_id"] == attempt:
            if not await self.wait_until_resumed():
                return
            if turn["status"] not in ("capturing", "interrupted"):
                return
            if now_ms() >= turn["deadline_ms"]:
                turn["status"] = "draining"
                self.event("turn_draining", {"turn_id": turn["id"], "reason": "deadline"})
                return
            await asyncio.sleep(0.1)

    async def epoch(self, turn_id, data):
        async with self.lock:
            t = self.current(turn_id, data.attempt_id)
            if t["status"] != "capturing" or t.get("capture_closed"):
                raise Conflict("Capture is not active")
            old = next((e for e in t["epochs"] if e["id"] == data.epoch_id), None)
            if old:
                if old["rate"] != data.sample_rate or old["start_ms"] != data.capture_server_ms:
                    raise Conflict("An epoch cannot change its clock or sample rate")
                return {
                    "epoch_id": old["id"],
                    "next_sequence": old["next"],
                    "samples": old["samples"],
                }
            if data.capture_server_ms < t["start_ms"] - 250 or data.capture_server_ms > now_ms() + 250:
                raise Conflict("Capture clock is outside the authorized range; recalibrate")
            if data.capture_server_ms >= t["deadline_ms"]:
                raise Conflict("Speaking time has expired")
            if t["epochs"]:
                previous = t["epochs"][-1]
                if data.capture_server_ms < previous["start_ms"] + previous["samples"] / previous["rate"] * 1000:
                    raise Conflict("Capture epochs overlap")
                previous["closed"] = True
                if not previous.get("paused"):
                    t["omissions"] = True
            epoch = {
                "id": data.epoch_id,
                "rate": data.sample_rate,
                "start_ms": data.capture_server_ms,
                "uncertainty_ms": data.uncertainty_ms,
                "next": 0,
                "samples": 0,
                "processed": 0,
                "closed": False,
                "finish": None,
                "partial": False,
                "file": f"{t['id']}/{t['attempt_id']}/{data.epoch_id}.pcm",
            }
            t["epochs"].append(epoch)
            self.event("microphone_status", {"epoch_id": data.epoch_id})
            return {"epoch_id": epoch["id"], "next_sequence": 0, "samples": 0}

    async def frame(self, turn_id, frame):
        raw = base64.b64decode(frame.pcm, validate=True)
        if len(raw) != frame.samples * 2:
            raise Conflict("PCM must contain exactly samples × 2 bytes of mono signed 16-bit audio")
        digest = hashlib.sha256(raw).hexdigest()
        async with self.lock:
            t = self.current(turn_id, frame.attempt_id, allow_paused=True)
            epoch = next((e for e in t["epochs"] if e["id"] == frame.epoch_id), None)
            if not epoch:
                raise Conflict("Handshake for this capture epoch is missing")
            old = self.storage.frame(
                self.state["id"],
                frame.attempt_id,
                frame.epoch_id,
                frame.sequence,
                digest,
            )
            if old is not None:
                return {
                    "next_sequence": epoch["next"],
                    "samples": epoch["samples"],
                    "accepted": old,
                }
            if self.state.get("paused"):
                raise Conflict("Debate is paused. Resume the debate first.")
            if (
                t["status"] not in ("capturing", "interrupted", "draining")
                or t.get("capture_closed")
                or epoch["closed"]
            ):
                raise Conflict("This capture epoch is closed")
            if frame.sequence != epoch["next"] or frame.sample_start != epoch["samples"]:
                return {
                    "next_sequence": epoch["next"],
                    "samples": epoch["samples"],
                    "retry": True,
                }
            if frame.samples > math.ceil(epoch["rate"] * 0.2):
                raise Conflict("Microphone frames may contain at most 200 ms of audio")
            if now_ms() > t["deadline_ms"] + self.settings.transport_grace_seconds * 1000:
                raise Conflict("Transport grace period has expired")
            allowed = max(
                0,
                math.floor((t["deadline_ms"] - epoch["start_ms"]) * epoch["rate"] / 1000) - frame.sample_start,
            )
            accepted = min(frame.samples, allowed)
            if accepted == 0:
                raise Conflict("Frame was captured after the deadline")
            total_bytes = sum(e["samples"] * 2 for e in t["epochs"]) + accepted * 2
            if total_bytes > math.ceil(self.settings.budgets[t["stage"]] + 2) * 96000 * 2:
                raise Conflict("Recording storage limit reached; finish with received audio")
            # Durable file write is off the event loop; acknowledgements do not await ASR.
            path = self.root / epoch["file"]

            def spool():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("r+b" if path.exists() else "wb") as f:
                    f.seek(frame.sample_start * 2)
                    f.write(raw[: accepted * 2])
                    f.truncate()
                    f.flush()
                    import os

                    os.fsync(f.fileno())

            await asyncio.to_thread(spool)
            if self.state["status"] != "running":
                raise Conflict("Session stopped before audio was acknowledged")
            epoch["next"] += 1
            epoch["samples"] += accepted
            epoch["partial"] = accepted != frame.samples
            self.storage.frame(
                self.state["id"],
                frame.attempt_id,
                frame.epoch_id,
                frame.sequence,
                digest,
                accepted,
            )
            self.event(
                "audio_ingest_ack",
                {
                    "turn_id": turn_id,
                    "epoch_id": epoch["id"],
                    "next_sequence": epoch["next"],
                    "samples": epoch["samples"],
                },
            )
            return {
                "next_sequence": epoch["next"],
                "samples": epoch["samples"],
                "accepted": accepted,
            }

    async def finish(self, turn_id, data):
        async with self.lock:
            op = "finish:" + turn_id + ":" + data.attempt_id
            old = self.storage.command(self.state["id"], data.key, op)
            if old is not None:
                return old
            t = self.current(turn_id, data.attempt_id)
            if t["status"] not in ("capturing", "interrupted", "draining"):
                raise Conflict("This turn cannot finish recording now")
            epoch = next((e for e in t["epochs"] if e["id"] == data.epoch_id), None)
            if not epoch:
                raise Conflict("Capture epoch does not exist")
            if data.last_sequence < epoch["next"] - 1 or data.total_samples < epoch["samples"]:
                raise Conflict("Finish marker excludes acknowledged audio")
            epoch["finish"] = {
                "sequence": data.last_sequence,
                "samples": data.total_samples,
            }
            t["finish_by_ms"] = min(
                now_ms() + self.settings.transport_grace_seconds * 1000,
                t["deadline_ms"] + self.settings.transport_grace_seconds * 1000,
            )
            t["status"] = "draining"
            self.event("turn_draining", {"turn_id": turn_id})
            return self.save_command(data.key, op, {"status": "draining"})

    async def disconnect(self):
        async with self.lock:
            if self.state["status"] == "running" and not self.state.get("paused"):
                t = self.current()
                if t["human"] and t["status"] == "capturing":
                    t["status"] = "interrupted"
                    self.event("microphone_status", {"status": "interrupted"})

    def audio_file(self, t, e, start, end):
        path = self.root / t["id"] / t["attempt_id"] / f"{e['id']}_{start}_{end}.wav"
        with (self.root / e["file"]).open("rb") as f:
            f.seek(start * 2)
            pcm = f.read((end - start) * 2)
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(e["rate"])
            f.writeframes(pcm)
        return path

    def recoverable(self, t, message):
        t["status"], t["error"], t["capture_closed"] = (
            "recovery_required",
            message,
            True,
        )
        for e in t["epochs"]:
            e["closed"] = True
        self.event("microphone_status", {"status": "recovery_required", "message": message})

    async def listen(self, t, attempt):
        try:
            while self.state["status"] == "running" and t["attempt_id"] == attempt:
                if not await self.wait_until_resumed():
                    return
                if t["status"] in ("recovering", "recovery_required", "cancelled", "completed"):
                    return
                now = now_ms()
                if now >= t["deadline_ms"] and t["status"] in (
                    "capturing",
                    "interrupted",
                ):
                    t["status"] = "draining"
                    self.event("turn_draining", {"turn_id": t["id"], "reason": "deadline"})
                final = t["epochs"][-1] if t["epochs"] else None
                marker = final.get("finish") if final else None
                if (
                    marker
                    and final["next"] - 1 == marker["sequence"]
                    and (final["samples"] == marker["samples"] or final["partial"])
                ):
                    t["capture_closed"] = True
                expiry = t.get(
                    "finish_by_ms",
                    t["deadline_ms"] + self.settings.transport_grace_seconds * 1000,
                )
                if now >= expiry:
                    if marker and not t["capture_closed"]:
                        self.recoverable(
                            t,
                            "Final audio frames are missing. Retry or finish with received audio.",
                        )
                        return
                    t["capture_closed"] = True
                    if not marker:
                        t["omissions"] = True
                if t["capture_closed"]:
                    for e in t["epochs"]:
                        e["closed"] = True
                for e in t["epochs"]:
                    if not await self.wait_until_resumed():
                        return
                    available = e["samples"] - e["processed"]
                    threshold = max(
                        1,
                        int(self.settings.streaming["input"]["min_audio_seconds"] * e["rate"]),
                    )
                    if available <= 0 or (available < threshold and not e["closed"]):
                        continue
                    start = e["processed"]
                    # Keep final drains bounded too (WAV uploads must fit provider limits).
                    end = start + min(available, threshold, 10_000_000)
                    path = await asyncio.to_thread(self.audio_file, t, e, start, end)
                    if (self.state["status"] != "running" or t["attempt_id"] != attempt
                            or t["status"] in ("recovering", "recovery_required")):
                        return
                    text = await self.asr_worker.call("transcribe", path=str(path))
                    if (self.state["status"] != "running" or t["attempt_id"] != attempt
                            or t["status"] in ("recovering", "recovery_required")):
                        return
                    segment = {
                        "id": f"{attempt}:{e['id']}:{start}:{end}",
                        "epoch_id": e["id"],
                        "start": start,
                        "end": end,
                        "start_ms": e["start_ms"] - t["start_ms"] + start / e["rate"] * 1000,
                        "end_ms": e["start_ms"] - t["start_ms"] + end / e["rate"] * 1000,
                        "text": text,
                        "analyzed": False,
                        "audio": str(path.relative_to(self.root)),
                    }
                    t["transcripts"].append(segment)
                    e["processed"] = end
                    self.event("transcript_ready", {"turn_id": t["id"], "segment": segment})
                    segment["received_ms"] = now_ms()
                pending = [
                    segment for segment in t["transcripts"] if not segment["analyzed"] and segment["text"].strip()
                ]
                words = sum(len(segment["text"].split()) for segment in pending)
                idle = self.settings.streaming["input"]["max_text_wait_seconds"]
                flush = pending and (
                    t["capture_closed"]
                    or words >= self.settings.streaming["input"]["min_text_words"]
                    or (idle > 0 and now_ms() - pending[0]["received_ms"] >= idle * 1000)
                )
                if (
                    flush
                    and not self.state.get("paused")
                    and self.state["preparation_status"] == "ready"
                    and (self.analysis_task is None or self.analysis_task.done())
                ):
                    self.analysis_task = self.task(self.analyze_segments(t, attempt, pending))
                if self.state["preparation_status"] != "ready" or pending:
                    await asyncio.sleep(0.1)
                    continue
                if t["capture_closed"] and all(e["processed"] == e["samples"] for e in t["epochs"]):
                    text = " ".join(s["text"] for s in t["transcripts"]).strip()
                    if not text:
                        self.recoverable(
                            t,
                            "No usable speech was transcribed. Re-record or explicitly skip this turn.",
                        )
                        return
                    await self.complete(t, text)
                    return
                await asyncio.sleep(0.1)
        except Exception as e:
            if (self.state["status"] == "running" and t["attempt_id"] == attempt
                    and t["status"] != "recovering"):
                if self.asr_worker.closed:
                    await self.fail(str(e))
                else:
                    self.recoverable(t, str(e))

    async def analyze_segments(self, t, attempt, segments):
        try:
            if not await self.wait_until_resumed():
                return
            if not t.get("checkpoint_ready"):
                await self.worker.call("checkpoint")
                if self.state["status"] != "running" or t["attempt_id"] != attempt:
                    return
                t["checkpoint_ready"] = True
            if t["status"] in ("recovering", "recovery_required"):
                return
            trees = await self.worker.call(
                "analyze",
                text=" ".join(s["text"] for s in segments),
                side=t["side"],
                stage=t["stage"],
            )
            if (self.state["status"] != "running" or t["attempt_id"] != attempt
                    or t["status"] in ("recovering", "recovery_required")):
                return
            for segment in segments:
                segment["analyzed"] = True
            self.state["trees"] = trees
            self.state["revision"] += 1
            self.event("tree_updated", {"revision": self.state["revision"], "segment_ids": [s["id"] for s in segments]})
        except Exception as e:
            if (self.state["status"] == "running" and t["attempt_id"] == attempt
                    and t["status"] != "recovering"):
                if self.worker.closed:
                    await self.fail(str(e))
                else:
                    self.recoverable(t, str(e))

    async def complete(self, t, text):
        if not await self.wait_until_resumed():
            return
        if t["status"] == "completed" or self.state["status"] != "running":
            return
        t["status"], t["text"] = "completed", text
        record = {"stage": t["stage"], "side": t["side"], "content": text}
        if t["transcripts"] and all(s["analyzed"] or not s["text"].strip() for s in t["transcripts"]):
            record["tree_via_streaming"] = True
        self.state["history"].append(record)
        self.event("turn_completed", {"turn_id": t["id"]})
        await self.advance()

    async def recover(self, turn_id, data):
        async with self.lock:
            op = "recover:" + turn_id + ":" + str(data.attempt_id) + ":" + str(data.action)
            old = self.storage.command(self.state["id"], data.key, op)
            if old is not None:
                return old
            t = self.current(turn_id, data.attempt_id)
            if t["status"] not in ("interrupted", "recovery_required") and not (
                data.action == "resume" and t["status"] == "capturing"
            ):
                raise Conflict("This turn does not need recovery")
            action = data.action
            if action == "resume":
                if now_ms() >= t["deadline_ms"] or t.get("capture_closed"):
                    raise Conflict("Recording cannot resume after the deadline or a closed capture")
                t["status"] = "capturing"
            elif action in (
                "rerecord",
                "retry_processing",
                "finish_received",
                "skip_empty",
            ):
                if action == "rerecord" and t["rerecords"] >= self.settings.max_rerecords:
                    raise Conflict("The re-record limit for this turn has been reached")
                if action == "skip_empty" and any(s["text"].strip() for s in t["transcripts"]):
                    raise Conflict("A turn containing speech cannot be skipped as empty")
                if action == "skip_empty" and any(e["processed"] < e["samples"] for e in t["epochs"]):
                    raise Conflict("Transcribe the received audio before deciding whether this turn is empty")
                if self.state["preparation_status"] != "ready":
                    raise Conflict("Retry AI preparation before processing recovery")
                # Suppress outstanding results without losing the recording identity.
                # Stop bypasses this lock, so recheck it after every wait.
                old_attempt = t["attempt_id"]
                t["status"] = "recovering"
                self.event("microphone_status", {"status": "recovering"})
                try:
                    if self.processing:
                        await self.processing
                    self.current(turn_id, old_attempt)
                    if self.analysis_task:
                        await self.analysis_task
                    self.current(turn_id, old_attempt)
                    if self.worker.closed or (self.asr_worker and self.asr_worker.closed):
                        raise RuntimeError("Debate worker exited during recovery")
                    trees = self.state["trees"]
                    if t.get("checkpoint_ready"):
                        trees = await self.worker.call("restore")
                    self.current(turn_id, old_attempt)
                except (Exception, asyncio.CancelledError) as e:
                    if self.state["status"] == "running":
                        message = "Recovery failed: " + (str(e) or "operation interrupted; retry recovery")
                        if (isinstance(e, asyncio.CancelledError) or self.worker.closed
                                or (self.asr_worker and self.asr_worker.closed)):
                            await self.fail(message)
                        else:
                            self.recoverable(t, message)
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    raise Conflict((self.state["error"] or "Session stopped during recovery")
                                   if self.state["status"] in TERMINAL else t["error"]) from e
                self.state["trees"] = trees
                self.state["revision"] += 1
                self.event(
                    "analysis_reset",
                    {"revision": self.state["revision"], "attempt_id": old_attempt},
                )
                t["transcripts"] = []
                for e in t["epochs"]:
                    e["processed"] = 0
                if action == "rerecord":
                    t["attempts"].append(
                        {
                            "attempt_id": old_attempt,
                            "epochs": t["epochs"],
                            "excluded": True,
                        }
                    )
                    t["rerecords"] += 1
                    t.update(status="waiting_for_human", attempt_id=None, epochs=[], error=None)
                elif action == "skip_empty":
                    await self.complete(t, "")
                else:
                    t["attempt_id"] = old_attempt
                    t.update(status="draining", error=None, capture_closed=True)
                    if action == "finish_received":
                        t["omissions"] = True
                        for e in t["epochs"]:
                            e["finish"] = None
                    for e in t["epochs"]:
                        e["closed"] = True
                    self.processing = self.task(self.listen(t, old_attempt))
            else:
                raise Conflict("Select a recovery action")
            self.event("microphone_status", {"status": t["status"]})
            return self.save_command(data.key, op, {"status": t["status"]})

    async def generate(self, t):
        speaker = self.ai_workers.get(t["side"], self.worker)
        opponent = "against" if t["side"] == "for" else "for"
        listener = self.ai_workers.get(opponent)

        async def chunk(data):
            if self.state["status"] != "running":
                return
            if data["index"] != len(t["chunks"]):
                raise RuntimeError("TTS chunk order is not contiguous")
            path = Path(data.pop("path")).resolve()
            data["file"] = str(path.relative_to(self.root))
            data["id"] = f"{t['id']}-{data['index']}"
            data["end_ms"] = t["generated_ms"] + data["duration_ms"]
            t["generated_ms"] = data["end_ms"]
            t["chunks"].append(data)
            # The first published TTS chunk can be fetched and played immediately;
            # subsequent generation and browser playback pauses are not wait time.
            t.setdefault("waiting_ended_ms", now_ms())
            t["status"] = "playing"
            self.event("audio_chunk_ready", {"turn_id": t["id"], **data})

        async def produce():
            output = self.root / t["id"] / "ai"
            output.mkdir(parents=True, exist_ok=True)
            result = await speaker.call(
                "generate", on_chunk=chunk, side=t["side"], stage=t["stage"],
                history=self.state["history"], output=str(output),
            )
            if self.state["status"] != "running":
                return
            if not t["chunks"]:
                raise RuntimeError("Generation finished without playable audio")
            t["generation_done"] = True
            t["text"] = result["text"]
            # Each AI worker owns one player's trees; never replace the listener's
            # newer tree snapshot with the speaker's generation result.
            self.state["trees"].update(result["trees"])
            self.event("speech_text_ready", {"turn_id": t["id"], "text": t["text"]})

        async def consume():
            while self.state["status"] == "running":
                if not await self.wait_until_resumed():
                    return
                for c in t["chunks"][t["analysis_done"] :]:
                    if t["played_ms"] + 1 < c["end_ms"]:
                        break
                    if listener:
                        text = await listener.call("transcribe", path=str(self.root / c["file"]))
                        if self.state["status"] != "running":
                            return
                        segment = {"id": c["id"], "text": text, "analyzed": False}
                        t["transcripts"].append(segment)
                        self.event("transcript_ready", {"turn_id": t["id"], "segment": segment})
                        trees = await listener.call("analyze", text=text, side=t["side"], stage=t["stage"])
                        if self.state["status"] != "running":
                            return
                        self.state["trees"].update(trees)
                        segment["analyzed"] = True
                        self.state["revision"] += 1
                        self.event("tree_updated", {"revision": self.state["revision"]})
                    t["analysis_done"] += 1
                if (t["generation_done"] and t["played_ms"] + 1 >= t["generated_ms"]
                        and t["analysis_done"] == len(t["chunks"])):
                    return
                await asyncio.sleep(0.1)

        tasks = [asyncio.create_task(produce()), asyncio.create_task(consume())]
        try:
            await asyncio.gather(*tasks)
            if self.state["status"] == "running":
                await self.complete(t, t["text"])
        except Exception as e:
            await self.fail(str(e))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def playback(self, data):
        async with self.lock:
            # Acknowledgements for audio played just before Pause can arrive late.
            t = self.current(data.turn_id, allow_paused=True)
            if t["human"] or data.played_ms > t["generated_ms"] + 1:
                raise Conflict("Playback position exceeds available AI audio")
            if data.played_ms < t["played_ms"]:
                raise Conflict("Playback position cannot go backwards")
            t["played_ms"] = min(data.played_ms, t["generated_ms"])
            self.event("playback_progress", {"turn_id": t["id"], "played_ms": t["played_ms"]})
            return {"played_ms": t["played_ms"]}


class Sessions:
    def __init__(self, root, worker_factory=Worker, *, max_active=2, idle_seconds=120,
                 unstarted_seconds=120, max_lifetime_seconds=None):
        if min(max_active, idle_seconds, unstarted_seconds) < 1:
            raise ValueError("Session capacity and timeouts must be positive")
        self.max_active = max_active
        self.idle_seconds = idle_seconds
        self.unstarted_seconds = unstarted_seconds
        self.max_lifetime_seconds = max_lifetime_seconds
        self.storage = Storage(root)
        self.worker_factory = worker_factory
        self.live = {}
        # Browser reconnect is supported; process-crash resumption is not.
        for state in self.storage.snapshots():
            if state["status"] not in TERMINAL or state.get("finalized") is False:
                finish_pause(state, now_ms())
                if state["status"] not in TERMINAL:
                    state["status"], state["error"] = (
                        "failed",
                        "Service restarted; audio retained. Create a new session.",
                    )
                elif state["status"] == "completed" and state["config"].get("evaluation") and state.get("evaluation") is None:
                    state["evaluation"] = "Evaluation unavailable: service restarted."
                state["finalized"] = True
                for turn in state["turns"]:
                    if "waiting_started_ms" in turn:
                        turn.setdefault("waiting_ended_ms", now_ms())
                kind = "session_failed" if state["status"] == "failed" else "session_finalized"
                self.storage.persist(state, kind, {"message": state["error"]})

    def archive_finished(self):
        for sid, session in list(self.live.items()):
            task = session.finalizer
            if task and task.done() and not task.cancelled() and task.exception() is None:
                self.live.pop(sid)

    def capacity(self):
        self.archive_finished()
        return {"limit": self.max_active, "active": len(self.live),
                "available": max(0, self.max_active - len(self.live))}

    async def reap(self):
        now = time.monotonic()
        expired = []
        for session in list(self.live.values()):
            reason = None
            if (self.max_lifetime_seconds and now - session.created >= self.max_lifetime_seconds):
                reason = "The public debate time limit was reached."
            elif session.state["status"] == "created" and now - session.created >= self.unstarted_seconds:
                reason = "The debate was never started. Create a new session."
            elif session.state["status"] not in TERMINAL and now - session.last_seen >= self.idle_seconds:
                reason = "The browser disconnected for too long. Create a new session."
            if reason:
                expired.append(session.stop("session-expired", reason))
        if expired:
            await asyncio.gather(*expired)
        self.archive_finished()

    def create(self, settings, *, creation_key=None, controller=None):
        # The random client key is a recovery capability, scoped to its controller.
        # Deriving the ID keeps the mapping durable with the initial snapshot.
        sid = (hashlib.sha256(f"{controller}\0{creation_key}".encode()).hexdigest()[:32]
               if creation_key else None)
        if sid:
            previous = self.snapshot(sid)
            if previous is not None:
                if previous["config"] != settings.model_dump():
                    raise Conflict("Creation key already used with different settings")
                return previous
        # This synchronous admission section never yields within the single API
        # event loop. Reserve at creation, including unstarted sessions.
        self.archive_finished()
        if controller and any(s.controller == controller for s in self.live.values()):
            raise Conflict("This browser already has a debate. Finish or stop it before starting another.")
        if len(self.live) >= self.max_active:
            raise CapacityFull("Server busy—all debate slots are occupied. Please try again shortly.")
        s = Session(settings, self.storage, self.worker_factory, session_id=sid)
        s.controller = controller
        self.live[s.state["id"]] = s
        return s.state

    def snapshot(self, sid):
        if sid in self.live:
            return self.live[sid].state
        return self.storage.snapshot(sid)
