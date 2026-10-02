from pathlib import Path
import base64
import tempfile
import unittest
from fastapi.testclient import TestClient
from debate_app.api import create_app
from test_sessions import InlineWorker


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = TestClient(create_app(Path(self.tmp.name), InlineWorker))
        self.client.__enter__()
        data = self.client.post(
            "/api/sessions",
            json={"motion": "Test API debate", "engine": "demo", "evaluation": False},
        ).json()
        self.sid = data["session"]["id"]
        self.token = data["token"]
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "X-Controller-ID": "browser-1",
        }

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def test_api_requires_session_token(self):
        self.assertEqual(self.client.get("/api/sessions/" + self.sid).status_code, 403)
        self.assertEqual(
            self.client.get(
                "/api/sessions/" + self.sid, headers=self.headers
            ).status_code,
            200,
        )

    def test_defaults_and_validation(self):
        self.assertEqual(self.client.get("/api/config/defaults").status_code, 200)
        r = self.client.post(
            "/api/config/validate",
            json={
                "motion": "Test debate",
                "streaming": {"input": {"min_audio_seconds": 0}},
            },
        )
        self.assertEqual(r.status_code, 422)

    def test_creation_replays_lost_response_without_occupying_another_slot(self):
        self.client.post(f"/api/sessions/{self.sid}/stop", json={"key": "replace"}, headers=self.headers)
        body = {"motion": "Retry session creation", "engine": "demo", "evaluation": False}
        headers = {"X-Controller-ID": "browser-1", "Idempotency-Key": "creation-capability-" + "x" * 32}
        first = self.client.post("/api/sessions", json=body, headers=headers)
        replay = self.client.post("/api/sessions", json=body, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json(), first.json())
        self.assertEqual(len(self.client.app.state.sessions.live), 1)
        self.assertNotIn(headers["Idempotency-Key"], replay.text)
        changed = self.client.post("/api/sessions", json={**body, "motion": "Different motion"}, headers=headers)
        self.assertEqual(changed.status_code, 409)
        another_controller = self.client.post("/api/sessions", json=body,
            headers={**headers, "X-Controller-ID": "browser-2"})
        self.assertEqual(another_controller.status_code, 200)
        self.assertNotEqual(another_controller.json()["session"]["id"], first.json()["session"]["id"])
        self.assertNotIn(first.json()["token"], another_controller.text)
        sid = first.json()["session"]["id"]
        control = {**self.headers, "Authorization": "Bearer " + first.json()["token"]}
        started = self.client.post(f"/api/sessions/{sid}/start", json={"key": "start"}, headers=control)
        self.assertEqual(started.status_code, 200)
        current = self.client.post("/api/sessions", json=body, headers=headers).json()
        self.assertEqual(current["session"]["status"], "running")

    def test_creation_replay_survives_service_restart(self):
        self.client.post(f"/api/sessions/{self.sid}/stop", json={"key": "replace"}, headers=self.headers)
        body = {"motion": "Durable creation retry", "engine": "demo", "evaluation": False}
        headers = {"X-Controller-ID": "browser-1", "Idempotency-Key": "restart-capability-" + "x" * 32}
        first = self.client.post("/api/sessions", json=body, headers=headers).json()
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(Path(self.tmp.name), InlineWorker))
        self.client.__enter__()
        replay = self.client.post("/api/sessions", json=body, headers=headers)
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()["session"]["id"], first["session"]["id"])
        self.assertEqual(replay.json()["token"], first["token"])
        self.assertEqual(replay.json()["session"]["status"], "failed")
        self.assertEqual(len(self.client.app.state.sessions.storage.snapshots()), 2)

    def test_simultaneous_creations_respect_capacity_and_recover_after_stop(self):
        from concurrent.futures import ThreadPoolExecutor
        body = {"motion": "Concurrent creation", "engine": "demo", "evaluation": False}
        with ThreadPoolExecutor(4) as pool:
            responses = list(pool.map(lambda i: self.client.post("/api/sessions", json=body,
                headers={"X-Controller-ID": f"visitor-{i}", "Idempotency-Key": str(i) * 32}), range(1, 5)))
        self.assertEqual(sorted(r.status_code for r in responses), [200, 503, 503, 503])
        rejected = next(r for r in responses if r.status_code == 503)
        self.assertEqual(rejected.json()["code"], "server_busy")
        self.assertEqual(rejected.headers["Retry-After"], "5")
        self.assertEqual(self.client.get('/api/health').json()['capacity'],
                         {'limit': 2, 'active': 2, 'available': 0})
        self.client.post(f'/api/sessions/{self.sid}/stop', json={'key': 'stop'}, headers=self.headers)
        self.assertEqual(self.client.post('/api/sessions', json=body).status_code, 200)
        # Archived access and idempotent stop still work after the slot is reused.
        self.assertEqual(self.client.get(f'/api/sessions/{self.sid}/results', headers=self.headers).status_code, 200)
        self.assertEqual(self.client.post(f'/api/sessions/{self.sid}/stop', json={'key': 'stop'}, headers=self.headers).status_code, 200)

    def test_only_controlling_browser_can_refresh_presence(self):
        self.client.post(f'/api/sessions/{self.sid}/start', json={'key': 'start'}, headers=self.headers)
        session = self.client.app.state.sessions.live[self.sid]
        session.last_seen = 1
        url = f'/api/sessions/{self.sid}/heartbeat'
        self.assertEqual(self.client.post(url).status_code, 403)
        self.assertEqual(self.client.post(url, headers={**self.headers, 'X-Controller-ID': 'other'}).status_code, 409)
        self.client.get(f'/api/sessions/{self.sid}', headers=self.headers)
        self.assertEqual(session.last_seen, 1, 'Result viewing must not keep a debate alive')
        self.assertEqual(self.client.post(url, headers=self.headers).status_code, 200)
        self.assertGreater(session.last_seen, 1)

    def test_creation_key_validation_and_cors(self):
        body = {"motion": "Validate creation key", "engine": "demo"}
        response = self.client.post("/api/sessions", json=body,
            headers={"Idempotency-Key": "too-short", "X-Controller-ID": "browser-1"})
        self.assertEqual(response.status_code, 422)
        response = self.client.post("/api/sessions", json=body,
            headers={"Idempotency-Key": "x" * 32})
        self.assertEqual(response.status_code, 403)
        response = self.client.options("/api/sessions", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-controller-id,idempotency-key",
        })
        self.assertEqual(response.status_code, 200)

    def test_origin_rejected(self):
        self.assertEqual(
            self.client.post(
                "/api/sessions",
                json={"motion": "Test debate"},
                headers={"Origin": "https://attacker.example"},
            ).status_code,
            403,
        )

    def test_controller_ownership(self):
        url = "/api/sessions/" + self.sid
        self.client.post(url + "/start", json={"key": "start"}, headers=self.headers)
        headers = {**self.headers, "X-Controller-ID": "browser-2"}
        self.assertEqual(
            self.client.post(
                url + "/stop", json={"key": "stop"}, headers=headers
            ).status_code,
            409,
        )

    def test_restart_preserves_artifacts_and_marks_failed(self):
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(Path(self.tmp.name), InlineWorker))
        self.client.__enter__()
        data = self.client.get("/api/sessions/" + self.sid, headers=self.headers).json()
        self.assertEqual(data["status"], "failed")

    def test_microphone_clock_handshake(self):
        with self.client.websocket_connect(
            f"/api/sessions/{self.sid}/microphone?token={self.token}&controller=browser-1",
            headers={"Origin": "http://localhost:3000"},
        ) as ws:
            ws.send_json({"type": "clock", "request_id": "clock"})
            ack = ws.receive_json()
            self.assertEqual(ack["request_id"], "clock")
            self.assertGreater(ack["server_ms"], 0)

    def test_pipelined_microphone_frames_preserve_audio_and_replay_cursor(self):
        url = f"/api/sessions/{self.sid}"
        self.client.post(url + "/start", json={"key": "start"}, headers=self.headers).raise_for_status()
        auth = self.client.post(url + "/turns/opening_for/begin", json={"key": "begin"}, headers=self.headers)
        auth.raise_for_status()
        common = {"turn_id": "opening_for", "attempt_id": auth.json()["attempt_id"], "epoch_id": "pipeline"}
        audio = [bytes([i, 1]) * 1600 for i in range(8)]
        packets = [
            {**common, "type": "frame", "request_id": f"frame-{i}", "sequence": i,
             "sample_start": i * 1600, "samples": 1600, "pcm": base64.b64encode(raw).decode()}
            for i, raw in enumerate(audio)
        ]
        with self.client.websocket_connect(
            f"{url}/microphone?token={self.token}&controller=browser-1",
            headers={"Origin": "http://localhost:3000"},
        ) as ws:
            ws.send_json({"type": "clock", "request_id": "clock"})
            clock = ws.receive_json()
            ws.send_json({**common, "type": "epoch", "request_id": "epoch", "sample_rate": 16000,
                          "capture_server_ms": clock["server_ms"], "uncertainty_ms": 1})
            epoch_ack = ws.receive_json()
            self.assertEqual(epoch_ack["type"], "ack", epoch_ack)
            self.assertEqual(epoch_ack["next_sequence"], 0)
            # Queue the whole client window without waiting between frames.
            for packet in packets:
                ws.send_json(packet)
            for i in range(8):
                ack = ws.receive_json()
                self.assertEqual(ack["request_id"], f"frame-{i}")
                self.assertEqual(ack["next_sequence"], i + 1)
                self.assertEqual(ack["accepted"], 1600)
            # A lost reply may cause replay; the cumulative cursor stays at 8.
            for packet in packets[:2]:
                ws.send_json(packet)
            for _ in range(2):
                self.assertEqual(ws.receive_json()["next_sequence"], 8)
            ws.send_json({**common, "type": "finish", "request_id": "finish", "key": "finish",
                          "last_sequence": 7, "total_samples": 12800})
            self.assertEqual(ws.receive_json()["type"], "ack")
            session = self.client.app.state.sessions.live[self.sid]
            epoch = session.state["turns"][0]["epochs"][0]
            self.assertEqual(epoch["samples"], 12800)
            self.assertEqual((session.root / epoch["file"]).read_bytes(), b"".join(audio))

    def test_yaml_import_and_export(self):
        response = self.client.post(
            "/api/config/import",
            json={
                "text": "motion: A YAML debate\nengine: demo\nbudgets:\n  opening: 12\n"
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["budgets"]["opening"], 12)
        exported = self.client.post("/api/config/export", json=response.json())
        self.assertEqual(exported.status_code, 200)
        self.assertIn("budget_mode: audio_duration", exported.text)

    def test_unsupported_streaming_settings_rejected_at_all_entry_points(self):
        import json

        cases = [({section: {}}, section) for section in ("playback", "posthoc")]
        cases += [({"input": {key: value}}, key) for key, value in (
            ("poll_interval", 1), ("audio_format", "mp3"),
            ("max_audio_wait_seconds", 0), ("max_total_audio_seconds", 0),
        )]
        cases.append(({"output": {"budget_mode": "experiment_elapsed"}}, "budget_mode"))
        for streaming, field in cases:
            settings = {"motion": "A valid motion", "engine": "demo", "streaming": streaming}
            for route in ("/api/config/validate", "/api/config/export", "/api/sessions", "/api/config/import"):
                with self.subTest(route=route, field=field):
                    body = {"text": json.dumps(settings)} if route.endswith("import") else settings
                    response = self.client.post(route, json=body)
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn(field, response.text)

    def test_live_defaults_and_examples_roundtrip_without_unused_settings(self):
        defaults = self.client.get("/api/config/defaults").json()["settings"]
        streaming = defaults["streaming"]
        self.assertEqual(set(streaming), {"input", "output"})
        self.assertEqual(streaming["input"], {
            "min_audio_seconds": 3.0, "min_text_words": 60, "max_text_wait_seconds": 15.0,
        })
        defaults["streaming"]["input"]["min_audio_seconds"] = 7
        defaults["streaming"]["output"]["voice"] = "alloy"
        exported = self.client.post("/api/config/export", json=defaults)
        self.assertEqual(exported.status_code, 200, exported.text)
        imported = self.client.post("/api/config/import", json={"text": exported.text})
        self.assertEqual(imported.status_code, 200, imported.text)
        self.assertEqual(imported.json(), defaults)
        for name in ("demo.yml", "human-vs-ai.yml"):
            with self.subTest(name=name):
                path = Path(__file__).resolve().parents[2] / "configs" / name
                response = self.client.post("/api/config/import", json={"text": path.read_text()})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(set(response.json()["streaming"]), {"input", "output"})

    def test_artifact_download_excludes_session_token(self):
        import io, zipfile

        r = self.client.get(
            "/api/sessions/" + self.sid + "/artifacts", headers=self.headers
        )
        self.assertEqual(r.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            self.assertIn("results.json", z.namelist())
            self.assertNotIn(self.token, z.read("results.json").decode())

    def test_invalid_nested_input_is_validation_error(self):
        r = self.client.post(
            "/api/config/validate",
            json={"motion": "A valid motion", "streaming": {"input": None}},
        )
        self.assertEqual(r.status_code, 422)

    def test_malformed_microphone_messages_do_not_crash_the_connection(self):
        with self.client.websocket_connect(
            f"/api/sessions/{self.sid}/microphone?token={self.token}&controller=browser-1",
            headers={"Origin": "http://localhost:3000"},
        ) as ws:
            for raw in ('not-json', 'null', '[]', '{}'):
                ws.send_text(raw)
                self.assertEqual(ws.receive_json()["type"], "error")
            ws.send_json({"type": "clock", "request_id": "still-alive"})
            self.assertEqual(ws.receive_json()["request_id"], "still-alive")

    def test_config_import_requires_an_object(self):
        for body in (None, [], "text"):
            r = self.client.post("/api/config/import", json=body)
            self.assertEqual(r.status_code, 422)

    def test_non_ascii_session_token_is_rejected_without_server_error(self):
        r = self.client.get(f"/api/sessions/{self.sid}", params={"token": "invalid-观点"})
        self.assertEqual(r.status_code, 403)

    def test_binary_microphone_message_is_rejected_cleanly(self):
        from starlette.websockets import WebSocketDisconnect

        with self.client.websocket_connect(
            f"/api/sessions/{self.sid}/microphone?token={self.token}&controller=browser-1",
            headers={"Origin": "http://localhost:3000"},
        ) as ws:
            ws.send_bytes(b"not the microphone protocol")
            with self.assertRaises(WebSocketDisconnect) as closed:
                ws.receive_json()
            self.assertEqual(closed.exception.code, 1003)
