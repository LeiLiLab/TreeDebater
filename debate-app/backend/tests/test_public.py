from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from debate_app.api import create_app
from test_sessions import InlineWorker


class PublicTests(unittest.TestCase):
    def test_four_slot_public_profile_uses_mini_and_rejects_a_fifth_session(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {
            'DEBATE_APP_PUBLIC': '1', 'DEBATE_APP_PUBLIC_DAILY_SESSIONS': '20',
            'DEBATE_APP_MAX_ACTIVE_SESSIONS': '4',
        }):
            with TestClient(create_app(Path(root), InlineWorker)) as client:
                defaults = client.get('/api/config/defaults').json()
                settings = defaults['settings']
                self.assertEqual(defaults['capacity'], {'limit': 4, 'active': 0, 'available': 4})
                self.assertEqual(settings['ai_model'], 'gpt-4o-mini')
                self.assertEqual(settings['helper_model'], 'gpt-4o-mini')
                self.assertEqual(settings['streaming']['output']['refinement_model'], 'gpt-4o-mini')
                settings.update(engine='demo', evaluation=False)
                for i in range(4):
                    response = client.post('/api/sessions', json=settings,
                                           headers={'X-Controller-ID': f'visitor-{i}'})
                    self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(client.get('/api/health').json()['capacity'],
                                 {'limit': 4, 'active': 4, 'available': 0})
                fifth = client.post('/api/sessions', json=settings,
                                    headers={'X-Controller-ID': 'visitor-4'})
                self.assertEqual(fifth.status_code, 503)
                self.assertEqual(fifth.json()['code'], 'server_busy')
                self.assertEqual(fifth.headers['retry-after'], '5')

    def test_busy_rejection_does_not_consume_daily_allowance(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {
            'DEBATE_APP_PUBLIC': '1', 'DEBATE_APP_PUBLIC_DAILY_SESSIONS': '3',
            'DEBATE_APP_MAX_ACTIVE_SESSIONS': '2',
        }):
            with TestClient(create_app(Path(root), InlineWorker)) as client:
                settings = client.get('/api/config/defaults').json()['settings']
                settings.update(engine='demo', evaluation=False)
                first = client.post('/api/sessions', json=settings).json()
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 200)
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 503)
                client.post(f'/api/sessions/{first["session"]["id"]}/stop',
                            json={'key': 'stop'}, headers={
                                'Authorization': 'Bearer ' + first['token'],
                                'X-Controller-ID': 'public-test',
                            })
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 200)
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 429)

    def test_public_cap_is_durable_and_retries_keep_their_slot(self):
        with tempfile.TemporaryDirectory() as root, patch.dict('os.environ', {
            'DEBATE_APP_PUBLIC': '1', 'DEBATE_APP_PUBLIC_DAILY_SESSIONS': '1',
        }):
            headers = {'X-Controller-ID': 'public-test', 'Idempotency-Key': 'x' * 32}
            with TestClient(create_app(Path(root), InlineWorker)) as client:
                settings = client.get('/api/config/defaults').json()['settings']
                settings.update(engine='demo', evaluation=False)
                self.assertEqual(client.post('/api/sessions', json={**settings, 'ai_model': 'unlimited-model'}).status_code, 422)
                self.assertEqual(client.post('/api/sessions', json={**settings, 'budgets': {'opening': 300, 'rebuttal': 60, 'closing': 30}}).status_code, 422)
                first = client.post('/api/sessions', json=settings, headers=headers)
                self.assertEqual(first.status_code, 200, first.text)
                self.assertEqual(client.post('/api/sessions', json=settings, headers=headers).json(), first.json())
                sid = first.json()['session']['id']
                self.assertEqual(client.get(f'/api/sessions/{sid}').status_code, 403)
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 429)
            with TestClient(create_app(Path(root), InlineWorker)) as client:
                self.assertEqual(client.post('/api/sessions', json=settings).status_code, 429)
                self.assertEqual(client.post('/api/sessions', json=settings, headers=headers).status_code, 200)
