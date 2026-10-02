import asyncio
from contextlib import asynccontextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import secrets
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from .config import DATA_ROOT, positive_env_int
from .schemas import Command, ConfigImport, Epoch, Finish, Frame, Playback, SessionSettings
from .sessions import CapacityFull, Conflict, Sessions, now_ms


def create_app(data_root=DATA_ROOT, worker_factory=None):
    public_demo = os.getenv("DEBATE_APP_PUBLIC", "0") == "1"

    async def maintain_sessions(app):
        while True:
            await asyncio.sleep(5)
            try:
                await app.state.sessions.reap()
            except Exception:
                logging.getLogger(__name__).exception("Session cleanup failed; capacity remains reserved")

    @asynccontextmanager
    async def lifespan(app):
        app.state.sessions = Sessions(
            Path(data_root),
            max_active=positive_env_int("DEBATE_APP_MAX_ACTIVE_SESSIONS", 2),
            idle_seconds=positive_env_int("DEBATE_APP_RECONNECT_GRACE_SECONDS", 120),
            unstarted_seconds=positive_env_int("DEBATE_APP_UNSTARTED_TIMEOUT_SECONDS", 120),
            max_lifetime_seconds=40 * 60 if public_demo else None,
            **({"worker_factory": worker_factory} if worker_factory else {}),
        )
        expiry = asyncio.create_task(maintain_sessions(app))
        yield
        if expiry:
            expiry.cancel()
            with suppress(asyncio.CancelledError):
                await expiry
        async def shutdown(session):
            await session.fail("Service stopped; audio retained. Create a new session.")
            await session.stop("service-shutdown")
        await asyncio.gather(*(shutdown(s) for s in list(app.state.sessions.live.values())))
        app.state.sessions.storage.db.close()

    app = FastAPI(title="TreeDebater microphone app", lifespan=lifespan)
    origins = os.getenv(
        "DEBATE_APP_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Controller-ID", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def local_origins(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse({"detail": "This browser origin is not allowed"}, 403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        if isinstance(exc, CapacityFull):
            return JSONResponse({"detail": str(exc), "code": "server_busy"},
                                status_code=503, headers={"Retry-After": "5"})
        return JSONResponse({"detail": str(exc)}, status_code=409 if isinstance(exc, Conflict) else 422)

    def store():
        return app.state.sessions

    def authorize(sid, token):
        state = store().snapshot(sid)
        if state is None:
            raise HTTPException(404, "Session not found")
        if not token or not secrets.compare_digest(token.encode(), state["token"].encode()):
            raise HTTPException(403, "Session access token is missing or invalid")
        return state

    def access(sid, request):
        token = request.headers.get("authorization", "").removeprefix("Bearer ") or request.query_params.get("token")
        return authorize(sid, token)

    def live(sid, request):
        access(sid, request)
        if sid not in store().live:
            raise HTTPException(409, "This session is archived; create a new session")
        session = store().live[sid]
        controller = request.headers.get("x-controller-id")
        if not controller or len(controller) > 128:
            raise HTTPException(403, "A browser controller ID is required")
        if session.controller and session.controller != controller:
            raise HTTPException(409, "Another browser controls this session")
        session.controller = controller
        session.last_seen = time.monotonic()
        return session

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "protocol": 1, "capacity": store().capacity()}

    @app.get("/api/clock")
    async def clock():
        return {"server_ms": now_ms()}

    @app.get("/api/config/defaults")
    async def defaults():
        from .public import public_settings
        return {
            "settings": (public_settings() if public_demo else SessionSettings(
                motion="Learning to be a good writer still matters in the age of AI"
            )).model_dump(),
            "schema": SessionSettings.model_json_schema(),
            "protocol": 1,
            "capacity": store().capacity(),
        }

    @app.post("/api/config/validate")
    async def validate(settings: SessionSettings):
        return settings.model_dump()

    @app.post("/api/config/import")
    async def import_config(data: ConfigImport):
        import yaml

        try:
            raw = yaml.safe_load(data.text)
            if isinstance(raw, dict) and "env" in raw and "debater" in raw:
                ai = next((d for d in raw["debater"] if d.get("type") == "treedebater"), {})
                raw = {
                    "motion": raw["env"]["motion"],
                    "ai_model": ai.get("model", "gpt-4o-mini"),
                    "helper_model": ai.get("helper_model"),
                    "streaming": raw.get("streaming", {}),
                    "budgets": raw["env"].get("speech_budgets", {}),
                    "first_side": "against" if raw["env"].get("reverse") else "for",
                }
            return SessionSettings.model_validate(raw).model_dump()
        except Exception as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/api/config/export")
    async def export_config(settings: SessionSettings):
        import yaml
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(
            yaml.safe_dump(settings.model_dump(), sort_keys=False),
            media_type="application/yaml",
        )

    @app.post("/api/sessions")
    async def create(
        settings: SessionSettings,
        request: Request,
        creation_key: Optional[str] = Header(default=None, alias="Idempotency-Key", min_length=32, max_length=128),
    ):
        controller = request.headers.get("x-controller-id")
        if (creation_key and not controller) or (controller is not None and (not controller or len(controller) > 128)):
            raise HTTPException(403, "A browser controller ID is required")
        if public_demo:
            from .public import check_quota, record_usage, validate_public
            validate_public(settings)
            sid = hashlib.sha256(f"{controller}\0{creation_key}".encode()).hexdigest()[:32] if creation_key else ""
            check_quota(store().storage, sid)
        state = store().create(settings, creation_key=creation_key, controller=controller)
        if public_demo:
            record_usage(store().storage, state["id"])
        return {"session": {k: v for k, v in state.items() if k != "token"}, "token": state["token"]}

    @app.get("/api/sessions/{sid}")
    async def snapshot(sid: str, request: Request):
        state = access(sid, request)
        return {k: v for k, v in state.items() if k != "token"}

    @app.post("/api/sessions/{sid}/start")
    async def start(sid: str, data: Command, request: Request):
        return await live(sid, request).start(data.key)

    @app.post("/api/sessions/{sid}/retry-preparation")
    async def retry_preparation(sid: str, data: Command, request: Request):
        return await live(sid, request).retry_preparation(data.key)

    @app.post("/api/sessions/{sid}/stop")
    async def stop(sid: str, data: Command, request: Request):
        state = access(sid, request)
        if sid not in store().live:
            return store().storage.command(sid, data.key, "stop") or {"status": state["status"]}
        return await live(sid, request).stop(data.key)

    @app.post("/api/sessions/{sid}/heartbeat")
    async def heartbeat(sid: str, request: Request):
        state = access(sid, request)
        if sid in store().live:
            live(sid, request)
        return {"status": state["status"], "finalized": state.get("finalized", True)}

    @app.post("/api/sessions/{sid}/pause")
    async def pause(sid: str, data: Command, request: Request):
        return await live(sid, request).pause(data.key)

    @app.post("/api/sessions/{sid}/resume")
    async def resume(sid: str, data: Command, request: Request):
        return await live(sid, request).resume(data.key)

    @app.post("/api/sessions/{sid}/turns/{turn_id}/begin")
    async def begin(sid: str, turn_id: str, data: Command, request: Request):
        return await live(sid, request).begin(turn_id, data.key)

    @app.post("/api/sessions/{sid}/turns/{turn_id}/recover")
    async def recover(sid: str, turn_id: str, data: Command, request: Request):
        if not data.attempt_id or not data.action:
            raise HTTPException(422, "Recovery needs attempt_id and action")
        return await live(sid, request).recover(turn_id, data)

    @app.post("/api/sessions/{sid}/playback")
    async def playback(sid: str, data: Playback, request: Request):
        return await live(sid, request).playback(data)

    @app.get("/api/sessions/{sid}/events")
    async def events(sid: str, request: Request, after: int = 0):
        access(sid, request)
        after = max(after, int(request.headers.get("last-event-id", "0")))

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                rows = store().storage.events(sid, cursor)
                for event in rows:
                    cursor = event["seq"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.4)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions/{sid}/audio/{chunk_id}")
    async def audio(sid: str, chunk_id: str, request: Request):
        state = access(sid, request)
        chunk = next(
            (c for t in state["turns"] for c in t["chunks"] if c["id"] == chunk_id),
            None,
        )
        if chunk is None:
            raise HTTPException(404, "Audio chunk not found")
        path = store().storage.root / "sessions" / sid / chunk["file"]
        return FileResponse(path, media_type="audio/wav" if path.suffix == ".wav" else "audio/mpeg")

    @app.get("/api/sessions/{sid}/results")
    async def results(sid: str, request: Request):
        state = access(sid, request)
        clean = {k: v for k, v in state.items() if k != "token"}
        return JSONResponse(
            clean,
            headers={"Content-Disposition": f'attachment; filename="debate-{sid}.json"'},
        )

    @app.get("/api/sessions/{sid}/artifacts")
    async def artifacts(sid: str, request: Request):
        state = access(sid, request)
        # Use disk-backed archives so recordings never have to fit in service RAM.
        import tempfile
        import zipfile
        from starlette.background import BackgroundTask

        clean = json.loads(json.dumps({k: v for k, v in state.items() if k != "token"}))
        root = store().storage.root / "sessions" / sid

        def build():
            fd, name = tempfile.mkstemp(prefix="artifacts-", suffix=".zip", dir=root)
            os.close(fd)
            target = Path(name)
            try:
                with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                    z.writestr("results.json", json.dumps(clean, indent=2))
                    for path in root.rglob("*"):
                        if (
                            path.is_file()
                            and not path.is_symlink()
                            and path.suffix in (".wav", ".mp3", ".pcm", ".json", ".csv", ".txt")
                        ):
                            z.write(path, str(path.relative_to(root)))
                return target
            except Exception:
                target.unlink(missing_ok=True)
                raise

        archive = await asyncio.to_thread(build)
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"debate-{sid}.zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )

    @app.websocket("/api/sessions/{sid}/microphone")
    async def microphone(ws: WebSocket, sid: str):
        if ws.headers.get("origin") not in origins:
            await ws.close(code=1008)
            return
        try:
            authorize(sid, ws.query_params.get("token"))
        except HTTPException:
            await ws.close(code=1008)
            return
        session = store().live.get(sid)
        controller = ws.query_params.get("controller")
        if not session or not controller or len(controller) > 128 or (session.controller and controller != session.controller):
            await ws.close(code=1008)
            return
        if getattr(session, "microphone_socket", None) is not None:
            await ws.close(code=1008)
            return
        session.controller = controller
        session.last_seen = time.monotonic()
        session.microphone_socket = ws
        await ws.accept()
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                raw = message.get("text")
                if raw is None:
                    await ws.close(code=1003, reason="Microphone messages must be JSON text")
                    break
                if len(raw) > 30000:
                    await ws.close(code=1009)
                    break
                request_id = None
                try:
                    packet = json.loads(raw)
                    if not isinstance(packet, dict):
                        raise ValueError("Microphone messages must be JSON objects")
                    kind = packet.pop("type", None)
                    request_id = packet.pop("request_id", None)
                    if request_id is not None and (not isinstance(request_id, str) or len(request_id) > 128):
                        request_id = None
                        raise ValueError("A request ID must be a string of at most 128 characters")
                    turn_id = packet.pop("turn_id", None)
                    if kind != "clock" and not isinstance(turn_id, str):
                        raise ValueError("A microphone turn ID is required")
                    if kind == "clock":
                        result = {"server_ms": now_ms()}
                    elif kind == "epoch":
                        result = await session.epoch(turn_id, Epoch(**packet))
                    elif kind == "frame":
                        result = await session.frame(turn_id, Frame(**packet))
                    elif kind == "finish":
                        result = await session.finish(turn_id, Finish(**packet))
                    else:
                        raise ValueError("Unknown microphone message type")
                    session.last_seen = time.monotonic()
                    await ws.send_json({"type": "ack", "request_id": request_id, **result})
                except (ValueError, ValidationError) as e:
                    await ws.send_json({"type": "error", "request_id": request_id, "message": str(e)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if getattr(session, "microphone_socket", None) is ws:
                session.microphone_socket = None
                await session.disconnect()

    return app


app = create_app()
