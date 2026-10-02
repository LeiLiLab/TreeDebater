import asyncio
from pathlib import Path
import tempfile
import time
import unittest

from debate_app.schemas import SessionSettings
from debate_app.sessions import CapacityFull, Conflict, Sessions
from test_sessions import InlineWorker


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sessions = Sessions(Path(self.tmp.name), InlineWorker)
        self.config = SessionSettings(motion='Independent debates', engine='demo', evaluation=False)

    async def asyncTearDown(self):
        await asyncio.gather(*(s.stop('cleanup') for s in list(self.sessions.live.values())))
        self.sessions.storage.db.close()
        self.tmp.cleanup()

    def create(self, controller, **settings):
        state = self.sessions.create(self.config.model_copy(update=settings),
                                     controller=controller, creation_key=controller * 32)
        return self.sessions.live[state['id']]

    async def test_created_sessions_reserve_capacity_and_retries_do_not(self):
        a, b = self.create('a'), self.create('b')
        self.assertEqual(self.create('a'), a)
        self.assertNotEqual(a.state['token'], b.state['token'])
        self.assertNotEqual(a.root, b.root)
        with self.assertRaises(CapacityFull):
            self.create('c')
        with self.assertRaisesRegex(Conflict, 'already has a debate'):
            self.sessions.create(self.config, controller='a', creation_key='different' * 8)
        await a.stop('stop')
        c = self.create('c')
        self.assertNotIn(a.state['id'], self.sessions.live)
        archived = self.sessions.snapshot(a.state['id'])
        self.assertTrue(archived['finalized'])
        self.assertEqual(archived['status'], 'cancelled')
        self.assertEqual(self.sessions.capacity()['active'], 2)
        self.assertEqual(c.state['status'], 'created')

    async def test_evaluation_and_overlapping_stops_hold_capacity_until_cleanup(self):
        evaluate_entered, evaluate_release = asyncio.Event(), asyncio.Event()
        stop_entered, stop_release = asyncio.Event(), asyncio.Event()

        class SlowWorker(InlineWorker):
            stops = 0

            async def call(self, operation, **kwargs):
                if operation == 'evaluate':
                    evaluate_entered.set()
                    await evaluate_release.wait()
                return await super().call(operation, **kwargs)

            async def stop(self):
                self.stops += 1
                stop_entered.set()
                await stop_release.wait()
                await super().stop()

        self.sessions.worker_factory = SlowWorker
        self.sessions.max_active = 1
        s = self.create('a', evaluation=True)
        await s.start('start')
        await s.runner
        s.state['current'] = 5
        finish = s.task(s.advance())
        await asyncio.wait_for(evaluate_entered.wait(), 1)
        self.assertEqual(s.state['status'], 'completed')
        with self.assertRaises(CapacityFull):
            self.create('b')
        evaluate_release.set()
        await asyncio.wait_for(stop_entered.wait(), 1)
        first = asyncio.create_task(s.stop('same-stop'))
        second = asyncio.create_task(s.stop('same-stop'))
        await asyncio.sleep(0)
        try:
            self.assertFalse(first.done())
            self.assertFalse(second.done())
            with self.assertRaises(CapacityFull):
                self.create('b')
        finally:
            stop_release.set()
        await asyncio.wait_for(asyncio.gather(first, second, finish), 2)
        self.assertEqual(s.worker.stops, 1)
        self.assertTrue(s.state['evaluation'])
        self.assertEqual(self.sessions.capacity()['active'], 0)
        self.assertEqual(self.sessions.storage.events(s.state['id'], 0)[-1]['type'], 'session_finalized')

    async def test_stopping_preparation_drains_jobs_before_reusing_slot(self):
        prepare_entered, release = asyncio.Event(), asyncio.Event()

        class SlowWorker(InlineWorker):
            async def call(self, operation, **kwargs):
                if operation == 'prepare':
                    prepare_entered.set()
                    await release.wait()
                return await super().call(operation, **kwargs)

            async def stop(self):
                release.set()
                await super().stop()

        self.sessions.worker_factory = SlowWorker
        s = self.create('a')
        await s.start('start')
        await asyncio.wait_for(prepare_entered.wait(), 1)
        await asyncio.wait_for(s.stop('stop'), 2)
        self.assertTrue(s.runner.done())
        self.assertTrue(s.worker.closed)
        self.assertTrue(s.state['finalized'])
        self.assertEqual(s.state['status'], 'cancelled')
        self.assertEqual(s.state['trees'], {})

    async def test_cancelled_stop_request_does_not_cancel_resource_cleanup(self):
        entered, release = asyncio.Event(), asyncio.Event()

        class SlowStop(InlineWorker):
            async def stop(self):
                entered.set()
                await release.wait()
                await super().stop()

        self.sessions.worker_factory = SlowStop
        self.sessions.max_active = 1
        s = self.create('a')
        await s.start('start')
        await s.runner
        request = asyncio.create_task(s.stop('stop'))
        await asyncio.wait_for(entered.wait(), 1)
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        try:
            self.assertFalse(s.finalizer.cancelled())
            with self.assertRaises(CapacityFull):
                self.create('b')
        finally:
            release.set()
        await asyncio.wait_for(s.finalizer, 2)
        self.assertTrue(s.worker.closed)
        self.assertEqual(self.sessions.capacity()['available'], 1)

    async def test_abandonment_expires_one_session_and_reconnect_keeps_other(self):
        a, b = self.create('a'), self.create('b')
        await asyncio.gather(a.start('start'), b.start('start'))
        await asyncio.gather(a.runner, b.runner)
        await b.pause('pause')
        a.last_seen = time.monotonic() - 121
        b.last_seen = time.monotonic() - 119
        await self.sessions.reap()
        self.assertEqual(a.state['status'], 'cancelled')
        self.assertIn('disconnected', a.state['error'])
        self.assertEqual(b.state['status'], 'running')
        self.assertFalse(b.worker.closed)
        b.last_seen = time.monotonic()  # authenticated reconnect heartbeat
        await self.sessions.reap()
        self.assertEqual(self.sessions.capacity()['active'], 1)

    async def test_unstarted_timeout_cannot_be_extended_with_heartbeats(self):
        s = self.create('a')
        s.created = time.monotonic() - 121
        s.last_seen = time.monotonic()
        await self.sessions.reap()
        self.assertEqual(s.state['status'], 'cancelled')
        self.assertIn('never started', s.state['error'])
        self.assertEqual(self.sessions.capacity()['available'], 2)

    async def test_public_lifetime_applies_even_with_active_heartbeats(self):
        self.sessions.max_lifetime_seconds = 2400
        s = self.create('a')
        await s.start('start')
        await s.runner
        s.created = time.monotonic() - 2401
        s.last_seen = time.monotonic()
        await self.sessions.reap()
        self.assertTrue(s.state['finalized'])
        self.assertIn('time limit', s.state['error'])
