import asyncio
import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from debate_app.schemas import Command, Epoch, Frame, Playback, SessionSettings
from debate_app.sessions import Conflict, Session, Sessions, now_ms
from debate_app.storage import Storage
from test_sessions import InlineWorker


class PauseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.tmp.name))
        self.base = now_ms()
        self.patch = patch('debate_app.sessions.now_ms', return_value=self.base)
        self.clock = self.patch.start()
        self.tasks = []
        self.s = None

    async def asyncTearDown(self):
        if self.s:
            await self.s.stop('cleanup')
            self.tasks.extend(t for t in (self.s.runner, self.s.processing, self.s.analysis_task, self.s.deadline_task) if t)
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), 3)
        self.patch.stop()
        self.storage.db.close()
        self.tmp.cleanup()

    async def start(self, first='for', factory=InlineWorker, wait=True):
        cfg = SessionSettings(motion='Pause regression', engine='demo', evaluation=False,
                              first_side=first, budgets={'opening': 10, 'rebuttal': 10, 'closing': 10})
        self.s = Session(cfg, self.storage, factory)
        await self.s.start('start')
        if wait:
            await self.s.runner
        return self.s.state['turns'][0]

    async def capture(self):
        t = await self.start()
        auth = await self.s.begin(t['id'], 'begin')
        await self.s.epoch(t['id'], Epoch(attempt_id=auth['attempt_id'], epoch_id='before',
                                         sample_rate=16000, capture_server_ms=self.base, uncertainty_ms=1))
        frame = Frame(type='frame', attempt_id=auth['attempt_id'], epoch_id='before', sequence=0,
                      sample_start=0, samples=1600, pcm=base64.b64encode(b'\0\x01' * 1600).decode())
        await self.s.frame(t['id'], frame)
        return t, frame

    async def test_pause_preserves_samples_attempt_and_remaining_speaking_time(self):
        t, frame = await self.capture()
        deadline, attempt = t['deadline_ms'], t['attempt_id']
        self.clock.return_value = self.base + 2000
        await self.s.pause('pause')
        self.assertTrue(t['epochs'][0]['closed'])
        self.assertEqual(t['epochs'][0]['samples'], 1600)
        self.assertEqual((await self.s.frame(t['id'], frame))['next_sequence'], 1)
        with self.assertRaisesRegex(Conflict, 'paused'):
            await self.s.frame(t['id'], frame.model_copy(update={'sequence': 1, 'sample_start': 1600}))
        self.clock.return_value = self.base + 62000
        await asyncio.sleep(0.15)
        self.assertEqual(t['status'], 'capturing')
        self.assertFalse(t['capture_closed'])
        await self.s.resume('resume')
        self.assertEqual(t['deadline_ms'], deadline + 60000)
        self.assertEqual(t['attempt_id'], attempt)
        await self.s.recover(t['id'], Command(key='capture-resume', attempt_id=attempt, action='resume'))
        await self.s.epoch(t['id'], Epoch(attempt_id=attempt, epoch_id='after', sample_rate=16000,
                                         capture_server_ms=self.clock.return_value, uncertainty_ms=1))
        await self.s.frame(t['id'], frame.model_copy(update={'epoch_id': 'after'}))
        self.assertFalse(t['omissions'], 'Intentional pauses must not mark recordings incomplete')
        self.assertEqual(sum(e['samples'] for e in t['epochs']), 3200)
        self.assertEqual(t['deadline_ms'] - self.clock.return_value, 8000)

    async def test_waiting_pause_is_idempotent_and_blocks_capture(self):
        t = await self.start()
        self.clock.return_value = self.base + 2000
        await self.s.pause('pause')
        self.clock.return_value = self.base + 5000
        await self.s.pause('pause')
        self.assertEqual(self.s.state['paused_at_ms'], self.base + 2000)
        with self.assertRaisesRegex(Conflict, 'paused'):
            await self.s.begin(t['id'], 'begin')
        self.clock.return_value = self.base + 10000
        await self.s.resume('resume')
        self.clock.return_value = self.base + 12000
        await self.s.resume('resume')
        self.assertEqual(t['waiting_paused_ms'], 8000)
        self.assertNotIn('deadline_ms', t)
        await self.s.begin(t['id'], 'begin')
        self.assertEqual(t['waiting_ended_ms'] - t['waiting_started_ms'] - t['waiting_paused_ms'], 4000)

    async def test_completion_waits_for_resume_and_stop_wakes_it(self):
        t = await self.start()
        await self.s.pause('pause')
        task = asyncio.create_task(self.s.complete(t, 'A completed claim.'))
        self.tasks.append(task)
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.assertEqual(self.s.state['current'], 0)
        await self.s.stop('stop')
        await asyncio.wait_for(task, 1)
        self.assertEqual(self.s.state['status'], 'cancelled')
        self.assertEqual(self.s.state['current'], 0)
        with self.assertRaisesRegex(Conflict, 'active'):
            await self.s.resume('resume')

    async def test_ai_preparation_and_generated_audio_are_retained_while_paused(self):
        prepare_gate, generate_gate = asyncio.Event(), asyncio.Event()

        class GatedWorker(InlineWorker):
            async def call(self, operation, **kwargs):
                if operation == 'prepare':
                    await prepare_gate.wait()
                if operation == 'generate':
                    await generate_gate.wait()
                return await super().call(operation, **kwargs)

            async def stop(self):
                prepare_gate.set()
                generate_gate.set()
                await super().stop()

        t = await self.start(first='against', factory=GatedWorker, wait=False)
        self.clock.return_value = self.base + 1000
        await self.s.pause('pause-prepare')
        prepare_gate.set()
        for _ in range(100):
            if self.s.state['preparation_status'] == 'ready':
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.s.state['current'], -1)
        self.clock.return_value = self.base + 3000
        await self.s.resume('resume-prepare')
        await self.s.runner
        self.assertEqual(self.s.state['current'], 0)
        await self.s.pause('pause-generate')
        self.clock.return_value = self.base + 5000
        generate_gate.set()
        for _ in range(100):
            if t['generation_done']:
                break
            await asyncio.sleep(0.01)
        self.assertTrue(t['generation_done'])
        self.assertGreater(len(t['chunks']), 0)
        await self.s.playback(Playback(turn_id=t['id'], played_ms=t['generated_ms']))
        await asyncio.sleep(0.15)
        self.assertEqual(self.s.state['current'], 0)
        self.clock.return_value = self.base + 9000
        processing = self.s.processing
        await self.s.resume('resume-playback')
        await asyncio.wait_for(processing, 2)
        self.assertEqual(self.s.state['current'], 1)
        self.assertEqual(t['waiting_paused_ms'], 4000)
        self.assertEqual(t['waiting_ended_ms'] - t['waiting_started_ms'] - t['waiting_paused_ms'], 1000)

    async def test_restart_archives_paused_session_and_retains_remaining_time(self):
        t, _ = await self.capture()
        self.clock.return_value = self.base + 2000
        await self.s.pause('pause')
        self.assertTrue(self.storage.snapshots()[0]['paused'])
        self.clock.return_value = self.base + 62000
        restarted = Sessions(Path(self.tmp.name))
        try:
            saved = restarted.storage.snapshots()[0]
            self.assertFalse(saved['paused'])
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['turns'][0]['deadline_ms'] - self.clock.return_value, 8000)
        finally:
            restarted.storage.db.close()
