import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from debate_app.schemas import SessionSettings
from debate_app.sessions import Session, Sessions
from debate_app.storage import Storage


class WaitingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self.temp.name))
        self.prepare_gate = asyncio.Event()
        self.chunk_gate = asyncio.Event()
        self.later_chunk_gate = asyncio.Event()
        test = self

        class Worker:
            closed = False

            async def call(self, operation, on_chunk=None, **args):
                if operation == 'prepare':
                    await test.prepare_gate.wait()
                if operation == 'generate':
                    for index, gate in enumerate((test.chunk_gate, test.later_chunk_gate)):
                        await gate.wait()
                        await on_chunk({'index': index, 'path': str(Path(args['output']) / f'{index}.mp3'),
                                        'text': 'A spoken claim.', 'duration_ms': 1000})
                    return {'text': 'A spoken claim.', 'trees': {}}
                return {}

            async def stop(self):
                self.closed = True
                for gate in (test.prepare_gate, test.chunk_gate, test.later_chunk_gate):
                    gate.set()

        self.worker = Worker()
        self.clock = patch('debate_app.sessions.now_ms', return_value=1000).start()

    async def asyncTearDown(self):
        await self.session.stop('cleanup')
        await asyncio.gather(*(task for task in (self.session.runner, self.session.processing) if task),
                             return_exceptions=True)
        patch.stopall()
        self.storage.db.close()
        self.temp.cleanup()

    async def start(self, human=True):
        settings = SessionSettings(motion='Waiting test', engine='demo', evaluation=False,
                                   first_side='for' if human else 'against')
        self.session = Session(settings, self.storage, lambda *a: self.worker)
        await self.session.start('start')
        return self.session.state['turns'][0]

    async def test_human_wait_stops_on_capture_and_survives_stop_and_storage(self):
        turn = await self.start()
        self.assertEqual(turn['waiting_started_ms'], 1000)
        self.assertNotIn('waiting_ended_ms', turn)
        self.clock.return_value = 6500
        await self.session.begin(turn['id'], 'begin')
        self.assertEqual(turn['waiting_ended_ms'], 6500)
        self.clock.return_value = 9000
        await self.session.stop('stop')
        saved = self.storage.snapshots()[0]['turns'][0]
        self.assertEqual(saved['waiting_ended_ms'] - saved['waiting_started_ms'], 5500)

    async def test_ai_wait_includes_preparation_and_stops_on_first_chunk_only(self):
        turn = await self.start(human=False)
        self.assertEqual(turn['waiting_started_ms'], 1000)
        self.clock.return_value = 4000
        self.prepare_gate.set()
        await self.session.runner
        self.assertEqual(turn['waiting_started_ms'], 1000)
        self.assertNotIn('waiting_ended_ms', turn)
        self.clock.return_value = 8000
        self.chunk_gate.set()
        for _ in range(30):
            if turn['chunks']:
                break
            await asyncio.sleep(0)
        self.assertEqual(turn['waiting_ended_ms'], 8000)
        self.clock.return_value = 12000
        self.later_chunk_gate.set()
        for _ in range(30):
            if len(turn['chunks']) == 2:
                break
            await asyncio.sleep(0)
        self.assertEqual(len(turn['chunks']), 2)
        self.assertEqual(turn['waiting_ended_ms'], 8000)

    async def test_stop_freezes_unfinished_wait(self):
        turn = await self.start()
        self.clock.return_value = 7000
        await self.session.stop('stop')
        self.assertEqual(turn['waiting_ended_ms'], 7000)
        self.clock.return_value = 9000
        await self.session.stop('stop-again')
        self.assertEqual(turn['waiting_ended_ms'], 7000)

    async def test_restart_freezes_persisted_wait(self):
        await self.start()
        self.clock.return_value = 7000
        restarted = Sessions(Path(self.temp.name))
        try:
            turn = restarted.storage.snapshots()[0]['turns'][0]
            self.assertEqual(turn['waiting_ended_ms'], 7000)
        finally:
            restarted.storage.db.close()
