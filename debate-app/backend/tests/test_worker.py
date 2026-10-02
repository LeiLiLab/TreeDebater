import asyncio
import base64
from pathlib import Path
import tempfile
import unittest
from debate_app.schemas import Epoch, Finish, Frame, Playback, SessionSettings
from debate_app.sessions import Session, Sessions, now_ms
from debate_app.storage import Storage
from debate_app.worker import Worker


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_crash_does_not_stop_a_second_debate(self):
        with tempfile.TemporaryDirectory() as root:
            sessions = Sessions(Path(root))
            cfg = SessionSettings(motion='Concurrent process isolation', engine='demo',
                                  mode='ai_ai', evaluation=True)
            first, second = [sessions.live[sessions.create(cfg)['id']] for _ in range(2)]
            try:
                await asyncio.gather(first.start('start'), second.start('start'))
                await asyncio.wait_for(asyncio.gather(first.runner, second.runner), 10)
                workers = [w for s in (first, second) for w in s.ai_workers.values()]
                self.assertEqual(len({w.process.pid for w in workers}), 4)
                first.worker.process.terminate()
                await asyncio.to_thread(first.worker.process.join, 2)

                async def drive(s):
                    while s.state['status'] == 'running':
                        t = s.current()
                        if t['generated_ms'] > t['played_ms']:
                            await s.playback(Playback(turn_id=t['id'], played_ms=t['generated_ms']))
                        await asyncio.sleep(0.02)
                    if s.processing:
                        await s.processing
                    await s.finalizer

                await asyncio.wait_for(asyncio.gather(drive(first), drive(second)), 15)
                self.assertEqual(first.state['status'], 'failed')
                self.assertEqual(second.state['status'], 'completed')
                self.assertEqual(len(second.state['history']), 6)
                self.assertTrue(second.state['evaluation'])
                self.assertEqual(sessions.capacity()['available'], 2)
                self.assertTrue(all(not w.process.is_alive() for w in workers))
            finally:
                await asyncio.gather(first.stop('cleanup'), second.stop('cleanup'))
                sessions.storage.db.close()

    async def dead_worker_fails_session(self, role):
        with tempfile.TemporaryDirectory() as root:
            storage = Storage(Path(root))
            session = Session(SessionSettings(
                motion="Unexpected worker exit", engine="demo", evaluation=False,
            ), storage)
            try:
                await session.start("start")
                await asyncio.wait_for(session.runner, 5)
                auth = await session.begin("opening_for", "begin")
                await session.asr_worker.call("prepare")
                dead = session.asr_worker if role == "asr" else session.worker
                dead.process.terminate()
                await asyncio.to_thread(dead.process.join, 2)
                self.assertFalse(dead.process.is_alive())
                turn = session.current()
                await session.epoch(turn["id"], Epoch(
                    attempt_id=auth["attempt_id"], epoch_id="epoch", sample_rate=16000,
                    capture_server_ms=now_ms(), uncertainty_ms=1,
                ))
                pcm = b"\0\x01" * 1600
                await session.frame(turn["id"], Frame(
                    attempt_id=auth["attempt_id"], epoch_id="epoch", sequence=0,
                    sample_start=0, samples=1600, pcm=base64.b64encode(pcm).decode(),
                ))
                await session.finish(turn["id"], Finish(
                    attempt_id=auth["attempt_id"], epoch_id="epoch", last_sequence=0,
                    total_samples=1600, key="finish",
                ))
                await asyncio.wait_for(session.processing, 5)
                if session.analysis_task:
                    await asyncio.wait_for(session.analysis_task, 5)
                self.assertTrue(dead.closed)
                self.assertEqual(session.state["status"], "failed")
                self.assertEqual(turn["status"], "failed")
                self.assertIn("worker exited", session.state["error"])
                self.assertTrue(session.worker.closed)
                self.assertTrue(session.asr_worker.closed)
                self.assertEqual((session.root / turn["epochs"][0]["file"]).read_bytes(), pcm)
                with self.assertRaisesRegex(RuntimeError, "worker has stopped"):
                    await dead.call("prepare")
            finally:
                await session.stop("cleanup")
                tasks = [task for task in (session.processing, session.analysis_task) if task]
                await asyncio.gather(*tasks, return_exceptions=True)
                storage.db.close()

    async def test_dead_asr_worker_fails_session_and_preserves_audio(self):
        await self.dead_worker_fails_session("asr")

    async def test_dead_analysis_worker_fails_session_and_preserves_audio(self):
        await self.dead_worker_fails_session("analysis")

    async def test_two_process_ai_debate_preserves_trees_and_completes(self):
        with tempfile.TemporaryDirectory() as root:
            storage = Storage(Path(root))
            session = Session(SessionSettings(
                motion="Two workers", engine="demo", mode="ai_ai", evaluation=False,
            ), storage)
            try:
                await session.start("start")
                await asyncio.wait_for(session.runner, 5)

                async def play_debate():
                    while session.state["status"] == "running":
                        turn = session.current()
                        if turn["generated_ms"] > turn["played_ms"]:
                            await session.playback(Playback(
                                turn_id=turn["id"], played_ms=turn["generated_ms"],
                            ))
                        await asyncio.sleep(0.02)
                    await session.processing

                await asyncio.wait_for(play_debate(), 10)
                self.assertEqual(session.state["status"], "completed")
                self.assertEqual(len(session.state["history"]), 6)
                self.assertTrue(all(t["analysis_done"] == 2 for t in session.state["turns"]))
                self.assertEqual(set(session.state["trees"]), {"for", "against"})
                self.assertTrue(all(not w.process.is_alive() for w in session.ai_workers.values()))
            finally:
                await session.stop("cleanup")
                if session.processing:
                    await session.processing
                storage.db.close()

    async def test_real_process_demo_and_shutdown(self):
        with tempfile.TemporaryDirectory() as root:
            config = SessionSettings(
                motion="Test worker process", engine="demo", evaluation=False
            ).model_dump()
            worker = Worker(config, Path(root))
            try:
                trees = await worker.call("prepare")
                self.assertIn("demo", trees)
                chunks = []

                async def chunk(value):
                    chunks.append(value)

                result = await worker.call(
                    "generate",
                    on_chunk=chunk,
                    side="for",
                    stage="opening",
                    history=[],
                    output=str(Path(root) / "audio"),
                )
                self.assertEqual(len(chunks), 2)
                self.assertTrue(Path(chunks[0]["path"]).is_file())
                self.assertIn("Demo", result["text"])
            finally:
                await worker.stop()
            self.assertFalse(worker.process.is_alive())
