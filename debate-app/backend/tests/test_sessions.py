import asyncio
import base64
from pathlib import Path
import tempfile
import unittest

from debate_app.engine_adapter import DemoEngine
from debate_app.schemas import Command, Epoch, Finish, Frame, Playback, SessionSettings
from debate_app.sessions import Session, Conflict, now_ms
from debate_app.storage import Storage


class InlineWorker:
    def __init__(self, config, root):
        self.engine = DemoEngine(config, root)
        self.closed = False
        self.fail_analysis = False
        self.gate = None

    async def call(self, operation, on_chunk=None, **args):
        if self.gate is not None and operation == "transcribe":
            await self.gate.wait()
        if operation == "generate":
            chunks = []
            result = self.engine.generate(**args, emit=chunks.append)
            for c in chunks:
                await on_chunk(c)
            return result
        if self.fail_analysis and operation == "analyze":
            self.engine.nodes.append({"side": "for", "claim": "partial mutation"})
            raise RuntimeError("Injected partial tree failure")
        return getattr(self.engine, operation)(**args)

    async def stop(self):
        self.closed = True
        if self.gate:
            self.gate.set()


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.tmp.name))
        cfg = SessionSettings(
            motion="Test microphone debate",
            engine="demo",
            evaluation=False,
            budgets={"opening": 10, "rebuttal": 10, "closing": 10},
        )
        self.s = Session(cfg, self.storage, InlineWorker)
        await self.s.start("start")
        await self.s.runner
        self.auth = await self.s.begin("opening_for", "begin")
        self.t = self.s.current()
        self.e = Epoch(
            attempt_id=self.auth["attempt_id"],
            epoch_id="epoch1",
            sample_rate=16000,
            capture_server_ms=now_ms(),
            uncertainty_ms=1,
        )
        await self.s.epoch(self.t["id"], self.e)

    async def asyncTearDown(self):
        await self.s.stop("teardown")
        if self.s.processing:
            await self.s.processing
        self.storage.db.close()
        self.tmp.cleanup()

    def frame(self, seq=0, start=0, samples=1600):
        return Frame(
            attempt_id=self.auth["attempt_id"],
            epoch_id="epoch1",
            sequence=seq,
            sample_start=start,
            samples=samples,
            pcm=base64.b64encode(b"\0\x01" * samples).decode(),
        )

    async def finish(self, last=0, samples=1600):
        await self.s.finish(
            self.t["id"],
            Finish(
                attempt_id=self.auth["attempt_id"],
                epoch_id="epoch1",
                last_sequence=last,
                total_samples=samples,
                key="finish",
            ),
        )

    async def wait_status(self, status, timeout=2):
        end = asyncio.get_running_loop().time() + timeout
        while self.t["status"] != status:
            if asyncio.get_running_loop().time() > end:
                self.fail(f"Expected {status}, got {self.t}")
            await asyncio.sleep(0.02)

    async def test_short_speech_tail_drains_once(self):
        await self.s.frame(self.t["id"], self.frame())
        await self.finish()
        await self.wait_status("completed")
        self.assertEqual(self.t["epochs"][0]["processed"], 1600)
        self.assertEqual(len(self.t["transcripts"]), 1)
        self.assertTrue(self.s.state["history"][0]["tree_via_streaming"])

    async def test_duplicate_audio_is_durable_and_idempotent(self):
        first = await self.s.frame(self.t["id"], self.frame())
        second = await self.s.frame(self.t["id"], self.frame())
        self.assertEqual(first["samples"], second["samples"])
        self.assertEqual((self.s.root / self.t["epochs"][0]["file"]).stat().st_size, 3200)
        other = self.frame()
        other.pcm = base64.b64encode(b"\x02\x01" * 1600).decode()
        with self.assertRaises(ValueError):
            await self.s.frame(self.t["id"], other)

    async def test_gap_requests_retry_without_consuming_samples(self):
        ack = await self.s.frame(self.t["id"], self.frame(1, 1600))
        self.assertTrue(ack["retry"])
        self.assertEqual(self.t["epochs"][0]["samples"], 0)

    async def test_finish_waits_for_last_frame(self):
        await self.finish()
        await asyncio.sleep(0.1)
        self.assertEqual(self.t["status"], "draining")
        await self.s.frame(self.t["id"], self.frame())
        await self.wait_status("completed")

    async def test_deadline_truncates_boundary_frame(self):
        self.t["deadline_ms"] = self.e.capture_server_ms + 50
        ack = await self.s.frame(self.t["id"], self.frame())
        self.assertEqual(ack["accepted"], 800)
        with self.assertRaises(Conflict):
            await self.s.frame(self.t["id"], self.frame(1, 800))

    async def test_rejected_overlapping_epoch_leaves_current_capture_open(self):
        await self.s.frame(self.t["id"], self.frame())
        overlap = self.e.model_copy(update={"epoch_id": "overlap"})
        with self.assertRaises(Conflict):
            await self.s.epoch(self.t["id"], overlap)
        self.assertFalse(self.t["epochs"][0]["closed"])
        ack = await self.s.frame(self.t["id"], self.frame(seq=1, start=1600))
        self.assertEqual(ack["samples"], 3200)

    async def test_skip_empty_does_not_discard_untranscribed_audio(self):
        await self.s.frame(self.t["id"], self.frame())
        await self.s.disconnect()
        with self.assertRaises(Conflict):
            await self.s.recover(self.t["id"], Command(
                key="skip-unprocessed", attempt_id=self.auth["attempt_id"], action="skip_empty",
            ))
        self.assertFalse(self.s.state["history"])
        self.assertEqual(self.t["attempt_id"], self.auth["attempt_id"])

    async def test_post_deadline_epoch_is_rejected(self):
        self.t["deadline_ms"] = now_ms() - 1
        data = self.e.model_copy(update={"epoch_id": "new", "capture_server_ms": now_ms()})
        with self.assertRaises(Conflict):
            await self.s.epoch(self.t["id"], data)

    async def test_same_epoch_cannot_change_rate(self):
        with self.assertRaises(Conflict):
            await self.s.epoch(self.t["id"], self.e.model_copy(update={"sample_rate": 48000}))

    async def test_asr_hang_does_not_block_ingestion_or_stop(self):
        self.s.settings.streaming["input"]["min_audio_seconds"] = 0.1
        self.s.asr_worker.gate = asyncio.Event()
        await self.s.frame(self.t["id"], self.frame())
        await asyncio.sleep(0.15)
        ack = await asyncio.wait_for(self.s.frame(self.t["id"], self.frame(1, 1600)), 0.5)
        self.assertEqual(ack["samples"], 3200)
        await asyncio.wait_for(self.s.stop("stop"), 0.5)
        self.assertEqual(self.s.state["status"], "cancelled")

    async def test_partial_analysis_failure_restores_before_retry(self):
        self.s.worker.fail_analysis = True
        await self.s.frame(self.t["id"], self.frame())
        await self.finish()
        await self.wait_status("recovery_required")
        self.s.worker.fail_analysis = False
        await self.s.recover(
            self.t["id"],
            Command(
                key="retry",
                attempt_id=self.auth["attempt_id"],
                action="retry_processing",
            ),
        )
        await self.wait_status("completed")
        self.assertFalse(any(n["claim"] == "partial mutation" for n in self.s.worker.engine.nodes))
        self.assertEqual(len(self.t["transcripts"]), 1)
        self.assertTrue(self.s.state["history"][0]["tree_via_streaming"])

    async def test_rerecord_invalidates_attempt_and_restores_tree(self):
        await self.s.disconnect()
        await self.s.recover(
            self.t["id"],
            Command(key="again", attempt_id=self.auth["attempt_id"], action="rerecord"),
        )
        self.assertEqual(self.t["status"], "waiting_for_human")
        with self.assertRaises(Conflict):
            await self.s.frame(self.t["id"], self.frame())
        auth = await self.s.begin(self.t["id"], "begin2")
        self.assertNotEqual(auth["attempt_id"], self.auth["attempt_id"])

    async def fail_analysis_for_recovery(self):
        self.s.worker.fail_analysis = True
        await self.s.frame(self.t["id"], self.frame())
        await self.finish()
        await self.wait_status("recovery_required")
        self.s.worker.fail_analysis = False
        self.assertTrue(self.t["checkpoint_ready"])

    async def test_restore_failure_preserves_attempt_and_can_retry_same_command(self):
        await self.fail_analysis_for_recovery()
        before = self.s.snapshot()["turns"][0]
        original = self.s.worker.call

        async def fail_once(operation, **args):
            if operation == "restore":
                self.s.worker.call = original
                raise RuntimeError("Injected restore failure")
            return await original(operation, **args)

        self.s.worker.call = fail_once
        command = Command(key="restore-retry", attempt_id=self.auth["attempt_id"], action="retry_processing")
        with self.assertRaisesRegex(Conflict, "Injected restore failure"):
            await self.s.recover(self.t["id"], command)
        self.assertEqual(self.t["attempt_id"], self.auth["attempt_id"])
        self.assertEqual(self.t["status"], "recovery_required")
        self.assertEqual(self.t["epochs"], before["epochs"])
        self.assertEqual(self.t["transcripts"], before["transcripts"])
        self.assertEqual(self.storage.snapshots()[0]["turns"][0]["attempt_id"], self.auth["attempt_id"])
        await self.s.recover(self.t["id"], command)
        await self.wait_status("completed")
        self.assertEqual(len(self.t["transcripts"]), 1)
        self.assertFalse(any(n["claim"] == "partial mutation" for n in self.s.worker.engine.nodes))

    async def stop_during_recovery(self, action):
        self.s.settings.streaming["input"]["min_audio_seconds"] = 0.1
        entered, release = asyncio.Event(), asyncio.Event()
        self.s.asr_worker.gate = release
        original = self.s.asr_worker.call

        async def delayed(operation, **args):
            if operation == "transcribe":
                entered.set()
            return await original(operation, **args)

        self.s.asr_worker.call = delayed
        await self.s.frame(self.t["id"], self.frame())
        await asyncio.wait_for(entered.wait(), 1)
        await self.s.disconnect()
        recovery = asyncio.create_task(self.s.recover(self.t["id"], Command(
            key="recover-then-stop", attempt_id=self.auth["attempt_id"], action=action,
        )))
        try:
            await self.wait_status("recovering")
            await asyncio.wait_for(self.s.stop("stop-during-recovery"), 1)
            stopped = self.s.snapshot()
            with self.assertRaisesRegex(Conflict, "stopped during recovery"):
                await asyncio.wait_for(recovery, 1)
            self.assertEqual(self.s.snapshot(), stopped)
            self.assertEqual(self.t["status"], "cancelled")
            self.assertEqual(self.t["attempt_id"], self.auth["attempt_id"])
            self.assertEqual(self.t["rerecords"], 0)
            self.assertFalse(self.t["transcripts"])
        finally:
            release.set()
            await asyncio.gather(recovery, return_exceptions=True)

    async def test_stop_during_rerecord_keeps_cancelled_recording(self):
        await self.stop_during_recovery("rerecord")

    async def test_stop_during_processing_retry_keeps_cancelled_recording(self):
        await self.stop_during_recovery("retry_processing")

    async def test_stop_during_restore_does_not_publish_restored_trees(self):
        await self.fail_analysis_for_recovery()
        entered, release = asyncio.Event(), asyncio.Event()
        self.s.worker.gate = release
        original = self.s.worker.call

        async def delayed(operation, **args):
            if operation == "restore":
                entered.set()
                await release.wait()
            return await original(operation, **args)

        self.s.worker.call = delayed
        recovery = asyncio.create_task(self.s.recover(self.t["id"], Command(
            key="stop-restore", attempt_id=self.auth["attempt_id"], action="rerecord",
        )))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.wait_for(self.s.stop("stop-during-restore"), 1)
            stopped = self.s.snapshot()
            with self.assertRaisesRegex(Conflict, "stopped during recovery"):
                await asyncio.wait_for(recovery, 1)
            self.assertEqual(self.s.snapshot(), stopped)
            self.assertEqual(self.t["status"], "cancelled")
            self.assertEqual(self.t["attempt_id"], self.auth["attempt_id"])
            self.assertEqual(self.t["rerecords"], 0)
        finally:
            release.set()
            await asyncio.gather(recovery, return_exceptions=True)

    async def test_fatal_restore_failure_fails_session_with_recording_retained(self):
        await self.fail_analysis_for_recovery()
        original = self.s.worker.call

        async def fatal(operation, **args):
            if operation == "restore":
                self.s.worker.closed = True
                raise RuntimeError("Debate worker exited")
            return await original(operation, **args)

        self.s.worker.call = fatal
        with self.assertRaisesRegex(Conflict, "worker exited"):
            await self.s.recover(self.t["id"], Command(
                key="fatal-restore", attempt_id=self.auth["attempt_id"], action="retry_processing",
            ))
        self.assertEqual(self.s.state["status"], "failed")
        self.assertEqual(self.t["status"], "failed")
        self.assertEqual(self.t["attempt_id"], self.auth["attempt_id"])
        self.assertEqual(self.t["epochs"][0]["samples"], 1600)
        self.assertTrue(self.s.asr_worker.closed)

    async def test_playback_cannot_exceed_generated_audio(self):
        await self.s.frame(self.t["id"], self.frame())
        await self.finish()
        await self.wait_status("completed")
        await asyncio.sleep(0.1)
        t = self.s.current()
        with self.assertRaises(Conflict):
            await self.s.playback(Playback(turn_id=t["id"], played_ms=999999))
        await self.s.playback(Playback(turn_id=t["id"], played_ms=t["generated_ms"]))
        await asyncio.sleep(0.15)
        self.assertEqual(t["status"], "completed")

    async def test_idempotency_rejects_reused_operation(self):
        result = await self.s.begin(self.t["id"], "begin")
        self.assertEqual(result, self.auth)
        with self.assertRaises(ValueError):
            await self.s.stop("begin")

    async def test_resume_keeps_deadline_and_records_new_epoch_gap(self):
        deadline = self.t["deadline_ms"]
        await self.s.frame(self.t["id"], self.frame())
        await self.s.disconnect()
        await self.s.recover(
            self.t["id"],
            Command(key="resume", attempt_id=self.auth["attempt_id"], action="resume"),
        )
        self.assertEqual(self.t["deadline_ms"], deadline)
        await asyncio.sleep(0.11)
        new = self.e.model_copy(
            update={
                "epoch_id": "epoch2",
                "sample_rate": 48000,
                "capture_server_ms": now_ms(),
            }
        )
        await self.s.epoch(self.t["id"], new)
        self.assertTrue(self.t["omissions"])
        self.assertTrue(self.t["epochs"][0]["closed"])

    async def test_missing_final_frame_requires_recovery_after_grace(self):
        await self.finish()
        self.t["finish_by_ms"] = now_ms() - 1
        await self.wait_status("recovery_required")
        self.assertIn("missing", self.t["error"])

    async def test_empty_turn_can_be_skipped_explicitly(self):
        await self.finish(last=-1, samples=0)
        await self.wait_status("recovery_required")
        await self.s.recover(
            self.t["id"],
            Command(key="skip", attempt_id=self.auth["attempt_id"], action="skip_empty"),
        )
        self.assertEqual(self.t["status"], "completed")
        self.assertEqual(self.s.state["history"][0]["content"], "")

    async def test_deadline_status_updates_while_asr_is_blocked(self):
        self.s.settings.streaming["input"]["min_audio_seconds"] = 0.1
        self.s.asr_worker.gate = asyncio.Event()
        await self.s.frame(self.t["id"], self.frame())
        await asyncio.sleep(0.12)
        self.t["deadline_ms"] = now_ms() + 50
        await self.wait_status("draining", timeout=0.5)
        await self.s.stop("stop-deadline")

    async def start_with_blocked_preparation(self, fail=False):
        await self.s.stop("replace")
        await self.s.processing
        gate = asyncio.Event()

        class SlowWorker(InlineWorker):
            async def call(self, operation, **args):
                if operation == "prepare":
                    await gate.wait()
                    if fail:
                        raise RuntimeError("Preparation rejected")
                return await super().call(operation, **args)

            async def stop(self):
                gate.set()
                await super().stop()

        self.s = Session(self.s.settings, self.storage, SlowWorker)
        await self.s.start("slow-start")
        self.assertEqual(self.s.current()["id"], "opening_for")
        self.auth = await asyncio.wait_for(self.s.begin("opening_for", "begin"), 0.5)
        self.t = self.s.current()
        self.e = Epoch(
            attempt_id=self.auth["attempt_id"],
            epoch_id="epoch1",
            sample_rate=16000,
            capture_server_ms=now_ms(),
            uncertainty_ms=1,
        )
        await self.s.epoch(self.t["id"], self.e)
        return gate

    async def test_capture_and_finish_before_preparation_completes(self):
        gate = await self.start_with_blocked_preparation()
        deadline = self.t["deadline_ms"]
        ack = await asyncio.wait_for(self.s.frame(self.t["id"], self.frame()), 0.5)
        self.assertEqual(ack["accepted"], 1600)
        await self.finish()
        await asyncio.sleep(0.15)
        self.assertTrue(self.t["capture_closed"])
        self.assertEqual(self.t["epochs"][0]["processed"], 1600)
        self.assertEqual(len(self.t["transcripts"]), 1)
        self.assertFalse(self.t["transcripts"][0]["analyzed"])
        self.assertEqual(self.s.state["current"], 0)
        gate.set()
        await self.s.runner
        await self.wait_status("completed")
        self.assertEqual(self.t["deadline_ms"], deadline)
        self.assertEqual(self.t["epochs"][0]["processed"], 1600)
        self.assertEqual(len(self.s.state["history"]), 1)
        self.assertTrue(self.s.state["history"][0]["tree_via_streaming"])

    async def test_preparation_failure_retains_audio_and_retry_drains_it(self):
        gate = await self.start_with_blocked_preparation(fail=True)
        await self.s.frame(self.t["id"], self.frame())
        gate.set()
        await self.s.runner
        self.assertEqual(self.s.state["preparation_status"], "failed")
        self.assertEqual(self.t["status"], "capturing")
        await self.s.frame(self.t["id"], self.frame(seq=1, start=1600))
        await self.finish(last=1, samples=3200)
        await asyncio.sleep(0.15)
        before = (self.t["attempt_id"], self.t["deadline_ms"])
        self.s.worker_factory = InlineWorker
        await self.s.retry_preparation("retry-prep")
        await self.s.runner
        await self.wait_status("completed")
        self.assertEqual(before, (self.t["attempt_id"], self.t["deadline_ms"]))
        self.assertEqual(self.t["epochs"][0]["processed"], 3200)
        self.assertEqual((self.s.root / self.t["epochs"][0]["file"]).stat().st_size, 6400)
        self.assertEqual(self.s.state["preparation_status"], "ready")

    async def test_deadline_and_stop_remain_responsive_during_preparation(self):
        await self.start_with_blocked_preparation()
        self.t["deadline_ms"] = now_ms() + 50
        await self.wait_status("draining", timeout=0.5)
        await asyncio.wait_for(self.s.stop("stop-preparing"), 0.5)
        await self.s.runner
        self.assertEqual(self.s.state["status"], "cancelled")
        self.assertEqual(self.s.state["current"], 0)

    async def test_live_transcript_before_preparation_and_before_finish(self):
        await self.start_with_blocked_preparation()
        self.s.settings.streaming["input"]["min_audio_seconds"] = 0.1
        await self.s.frame(self.t["id"], self.frame())
        for _ in range(50):
            if self.t["transcripts"]:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(len(self.t["transcripts"]), 1)
        self.assertEqual(self.t["status"], "capturing")
        self.assertEqual(self.s.state["preparation_status"], "preparing")
        self.assertFalse(self.t["transcripts"][0]["analyzed"])

    async def test_live_transcript_continues_while_analysis_is_blocked(self):
        self.s.settings.streaming["input"].update(min_audio_seconds=0.1, min_text_words=1)
        gate, entered = asyncio.Event(), asyncio.Event()
        original = self.s.worker.call

        async def delayed(operation, **args):
            if operation == "analyze":
                entered.set()
                await gate.wait()
            return await original(operation, **args)

        self.s.worker.call = delayed
        try:
            await self.s.frame(self.t["id"], self.frame())
            await asyncio.wait_for(entered.wait(), 1)
            await self.s.frame(self.t["id"], self.frame(seq=1, start=1600))
            for _ in range(50):
                if len(self.t["transcripts"]) == 2:
                    break
                await asyncio.sleep(0.02)
            self.assertEqual(len(self.t["transcripts"]), 2)
            self.assertFalse(any(s["analyzed"] for s in self.t["transcripts"]))
        finally:
            gate.set()
            if self.s.analysis_task:
                await self.s.analysis_task

    async def test_analysis_failure_does_not_schedule_more_analysis_from_inflight_asr(self):
        self.s.settings.streaming["input"].update(min_audio_seconds=0.1, min_text_words=1)
        analysis_entered, fail_analysis = asyncio.Event(), asyncio.Event()
        asr_entered, finish_asr = asyncio.Event(), asyncio.Event()
        original_model, original_asr = self.s.worker.call, self.s.asr_worker.call
        analysis_calls, asr_calls = 0, 0

        async def model(operation, **args):
            nonlocal analysis_calls
            if operation == "analyze":
                analysis_calls += 1
                if analysis_calls == 1:
                    analysis_entered.set()
                    await fail_analysis.wait()
                    raise RuntimeError("Analysis failed while the next ASR call was running")
            return await original_model(operation, **args)

        async def asr(operation, **args):
            nonlocal asr_calls
            if operation == "transcribe":
                asr_calls += 1
                if asr_calls == 2:
                    asr_entered.set()
                    await finish_asr.wait()
            return await original_asr(operation, **args)

        self.s.worker.call, self.s.asr_worker.call = model, asr
        try:
            await self.s.frame(self.t["id"], self.frame())
            await asyncio.wait_for(analysis_entered.wait(), 1)
            await self.s.frame(self.t["id"], self.frame(seq=1, start=1600))
            await asyncio.wait_for(asr_entered.wait(), 1)
            fail_analysis.set()
            await self.wait_status("recovery_required")
            finish_asr.set()
            await self.s.processing
            await self.s.analysis_task
            self.assertEqual(analysis_calls, 1)
        finally:
            fail_analysis.set()
            finish_asr.set()

    async def test_timer_freezes_at_finish_and_survives_later_events(self):
        await self.s.frame(self.t["id"], self.frame())
        await self.finish()
        stopped = self.t["timer_stopped_ms"]
        self.assertLess(stopped, self.t["deadline_ms"])
        await asyncio.sleep(0.1)
        self.s.event("test_snapshot")
        self.assertEqual(self.t["timer_stopped_ms"], stopped)

    async def test_stop_freezes_timer_and_resume_keeps_original_deadline(self):
        deadline = self.t["deadline_ms"]
        await self.s.disconnect()
        await self.s.recover(
            self.t["id"], Command(key="resume-clock", attempt_id=self.t["attempt_id"], action="resume")
        )
        self.assertEqual(self.t["deadline_ms"], deadline)
        self.assertNotIn("timer_stopped_ms", self.t)
        await self.s.stop("stop-clock")
        stopped = self.t["timer_stopped_ms"]
        await asyncio.sleep(0.1)
        self.s.event("test_snapshot")
        self.assertEqual(self.t["timer_stopped_ms"], stopped)


class OverlapTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.tmp.name))
        self.release = asyncio.Event()
        self.published = asyncio.Event()
        self.analyzed = asyncio.Event()
        self.fail_generation = False
        self.fail_listener = False
        self.analysis_gate = asyncio.Event()
        self.analysis_gate.set()
        owner = self

        class SerialWorker(InlineWorker):
            def __init__(self, config, root):
                super().__init__(config, root)
                self.lock = asyncio.Lock()

            async def call(self, operation, on_chunk=None, **args):
                async with self.lock:
                    if operation == "generate" and args["stage"] == "opening" and args["side"] == "for":
                        chunks = []
                        result = self.engine.generate(**args, emit=chunks.append)
                        await on_chunk(chunks[0])
                        owner.published.set()
                        await owner.release.wait()
                        if owner.fail_generation:
                            raise RuntimeError("producer failed")
                        await on_chunk(chunks[1])
                        return result
                    if operation == "analyze":
                        await owner.analysis_gate.wait()
                        if owner.fail_listener:
                            raise RuntimeError("listener failed")
                        result = await super().call(operation, **args)
                        owner.analyzed.set()
                        return result
                    return await super().call(operation, on_chunk=on_chunk, **args)

            async def stop(self):
                owner.release.set()
                owner.analysis_gate.set()
                await super().stop()

        self.s = Session(SessionSettings(motion="Overlap", engine="demo", mode="ai_ai", evaluation=False),
                         self.storage, SerialWorker)
        await self.s.start("start")
        await self.s.runner
        self.t = self.s.current()
        self.task = self.s.processing
        await asyncio.wait_for(self.published.wait(), 1)

    async def asyncTearDown(self):
        await self.s.stop("teardown")
        await asyncio.wait_for(self.task, 1)
        if self.s.processing is not self.task:
            await asyncio.wait_for(self.s.processing, 1)
        self.storage.db.close()
        self.tmp.cleanup()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(wait(), 2)

    async def test_delivered_chunk_is_analyzed_before_generation_finishes(self):
        await asyncio.sleep(0.12)
        self.assertFalse(self.t["transcripts"])
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=250))
        await asyncio.sleep(0.12)
        self.assertFalse(self.t["transcripts"])
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=500))
        await asyncio.wait_for(self.analyzed.wait(), 1)
        self.assertFalse(self.t["generation_done"])
        self.assertEqual(self.t["analysis_done"], 1)
        self.assertEqual(self.s.state["current"], 0)
        listener_trees = self.s.state["trees"]["against"]
        self.release.set()
        await self.until(lambda: self.t["generation_done"])
        self.assertEqual(self.s.state["trees"]["against"], listener_trees)
        self.assertEqual(self.t["status"], "playing")
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=1000))
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.t["analysis_done"], 2)
        self.assertEqual(len(self.t["transcripts"]), 2)
        self.assertTrue(self.s.state["history"][0]["tree_via_streaming"])
        self.assertEqual(self.s.state["current"], 1)
        # Roles reverse using the same prepared workers and accumulated trees.
        await self.until(lambda: self.s.current()["generation_done"])
        other = self.s.current()
        await self.s.playback(Playback(turn_id=other["id"], played_ms=other["generated_ms"]))
        await self.until(lambda: other["status"] == "completed")
        self.assertTrue(self.s.state["trees"]["for"]["against"])
        self.assertTrue(self.s.state["trees"]["against"]["for"])

    async def test_turn_waits_for_listener_tail_after_generation_and_playback(self):
        self.analysis_gate.clear()
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=500))
        await self.until(lambda: bool(self.t["transcripts"]))
        self.release.set()
        await self.until(lambda: self.t["generation_done"])
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=1000))
        await asyncio.sleep(0.12)
        self.assertEqual(self.s.state["current"], 0)
        self.assertFalse(self.s.state["history"])
        self.analysis_gate.set()
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.t["analysis_done"], 2)
        self.assertTrue(self.s.state["history"][0]["tree_via_streaming"])

    async def test_generation_failure_stops_both_workers_without_advancing(self):
        self.fail_generation = True
        self.release.set()
        await asyncio.wait_for(self.task, 1)
        self.assertEqual(self.s.state["status"], "failed")
        self.assertEqual(self.s.state["current"], 0)
        self.assertTrue(all(w.closed for w in self.s.ai_workers.values()))

    async def test_listener_failure_stops_blocked_producer(self):
        self.fail_listener = True
        await self.s.playback(Playback(turn_id=self.t["id"], played_ms=500))
        await asyncio.wait_for(self.task, 1)
        self.assertEqual(self.s.state["status"], "failed")
        self.assertIn("listener failed", self.s.state["error"])
        self.assertFalse(self.s.state["history"])
        self.assertTrue(all(w.closed for w in self.s.ai_workers.values()))

    async def test_stop_during_generation_drains_tasks_without_advancing(self):
        await self.s.stop("stop-blocked")
        await asyncio.wait_for(self.task, 1)
        self.assertEqual(self.s.state["status"], "cancelled")
        self.assertEqual(self.s.state["current"], 0)
        self.assertTrue(all(w.closed for w in self.s.ai_workers.values()))
