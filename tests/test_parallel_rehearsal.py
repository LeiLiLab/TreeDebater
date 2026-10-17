"""Offline concurrency and shared-budget regressions for rehearsal generation."""

from optional_experiments import require_local_experiment
require_local_experiment('experiments/assets/remaining_semantic_trees', module=True)


import asyncio
import importlib.util
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments/assets/remaining_semantic_trees"


def load_script(name):
    spec = importlib.util.spec_from_file_location("parallel_rehearsal_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(root, count=6, cap=60, prior_ledgers=None):
    manifest = {
        "motions": [{"index": index, "motion": f"Test motion {index}"} for index in range(1, count + 1)],
        "aggregate_cap_usd": cap,
        "prior_ledgers": prior_ledgers or [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def response_payload():
    return {
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "choices": [{"message": {"content": "Offline response"}, "finish_reason": "stop"}],
    }


def mock_response():
    payload = response_payload()
    response = Mock(content=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    response.json.return_value = payload
    return response


def completion_body():
    return {"model": "google.gemma-4-26b-a4b", "messages": [{"role": "user", "content": "Offline"}], "max_tokens": 128}


@pytest.fixture
def gateway_factory(tmp_path):
    module = load_script("gateway")
    applications = []

    def create(transport, cap=60, max_inflight=4, prior_ledgers=None):
        write_manifest(tmp_path, cap=cap, prior_ledgers=prior_ledgers)
        app = module.create_app(tmp_path, {"TVLY_API_KEY": "offline"}, transport, max_inflight)
        applications.append(app)
        return app

    yield create
    for app in applications:
        app.state.database.close()


async def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://offline")


async def set_label(client, index, stage="for:tree:1/2"):
    worker = f"motion_{index:02d}"
    response = await client.post(f"/workers/{worker}/label", json={"label": worker + ":" + stage})
    assert response.status_code == 200


async def wait_async(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for mocked concurrent work"
        await asyncio.sleep(0.01)


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, "Timed out waiting for offline worker"
        time.sleep(0.01)


def test_gateway_overlaps_requests_and_keeps_original_labels(gateway_factory, tmp_path):
    release = threading.Event()
    counter_lock = threading.Lock()
    entered = []

    def transport(*args, **kwargs):
        with counter_lock:
            entered.append(kwargs["json"])
        assert release.wait(5)
        return mock_response()

    app = gateway_factory(transport)

    async def scenario():
        async with await client_for(app) as client:
            for index in range(1, 5):
                await set_label(client, index)
            tasks = [asyncio.create_task(client.post(
                f"/workers/motion_{index:02d}/v1/chat/completions", json=completion_body(),
            )) for index in range(1, 5)]
            try:
                await wait_async(lambda: len(entered) == 4)
                await set_label(client, 1, "against:evidence:2/2")
                status = await client.get("/status")
                assert status.json()["exposure_usd"] > 0
            finally:
                release.set()
            assert all(response.status_code == 200 for response in await asyncio.gather(*tasks))

    asyncio.run(scenario())
    rows = app.state.database.execute("SELECT label,charged,status FROM calls ORDER BY label").fetchall()
    assert [row[0] for row in rows] == [f"motion_{index:02d}:for:tree:1/2" for index in range(1, 5)]
    assert all(row[1] == pytest.approx(0.000021) and row[2] == "ok" for row in rows)
    audits = [json.loads(path.read_text()) for path in (tmp_path / "requests").glob("*.json")]
    assert sorted(audit["label"] for audit in audits) == [row[0] for row in rows]


def test_pending_reservations_prevent_concurrent_budget_overspend(gateway_factory, tmp_path):
    release = threading.Event()
    entered = threading.Event()
    transport_calls = []

    def transport(*args, **kwargs):
        transport_calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return mock_response()

    app = gateway_factory(transport, cap=0.1)

    async def scenario():
        async with await client_for(app) as client:
            await set_label(client, 1)
            await set_label(client, 2)
            first = asyncio.create_task(client.post("/workers/motion_01/search", json={"query": "offline"}))
            try:
                await wait_async(entered.is_set)
                second = await client.post("/workers/motion_02/search", json={"query": "offline"})
                assert second.status_code == 402
                assert (tmp_path / "STOP").exists()
                with sqlite3.connect(tmp_path / "cost.sqlite") as connection:
                    assert connection.execute("SELECT charged,status FROM calls").fetchall() == [(0.064, "pending")]
            finally:
                release.set()
            assert (await first).status_code == 200
            assert (await client.post("/workers/motion_01/search", json={"query": "offline"})).status_code == 402

    asyncio.run(scenario())
    assert len(transport_calls) == 1
    assert app.state.database.execute("SELECT charged,status FROM calls").fetchall() == [(0.016, "ok")]


def test_waiting_requests_recheck_stop_after_concurrency_slot_opens(gateway_factory, tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return mock_response()

    app = gateway_factory(transport, max_inflight=1)

    async def scenario():
        async with await client_for(app) as client:
            await set_label(client, 1)
            await set_label(client, 2)
            tasks = [asyncio.create_task(client.post(
                f"/workers/motion_{index:02d}/search", json={"query": "offline"},
            )) for index in (1, 2)]
            try:
                await wait_async(entered.is_set)
                (tmp_path / "STOP").touch()
            finally:
                release.set()
            responses = await asyncio.gather(*tasks)
            assert sorted(response.status_code for response in responses) == [200, 402]

    asyncio.run(scenario())
    assert len(calls) == 1


def test_gateway_keeps_failed_reservations_across_restart(gateway_factory, tmp_path):
    transport = Mock(side_effect=ConnectionError("Offline failure"))
    first_app = gateway_factory(transport, cap=0.1)

    async def fail():
        async with await client_for(first_app) as client:
            await set_label(client, 1)
            response = await client.post("/workers/motion_01/search", json={"query": "offline"})
            assert response.status_code == 502

    asyncio.run(fail())
    second_app = gateway_factory(transport, cap=0.1)

    async def restart():
        async with await client_for(second_app) as client:
            await set_label(client, 2)
            response = await client.post("/workers/motion_02/search", json={"query": "offline"})
            assert response.status_code == 402

    asyncio.run(restart())
    transport.assert_called_once()
    records = first_app.state.database.execute("SELECT bound,charged,status FROM calls").fetchall()
    assert records == [(0.064, 0.064, "error")]


def test_default_accounting_includes_relocated_rehearsal_ledger(tmp_path):
    root = tmp_path / "remaining_semantic_trees"
    root.mkdir()
    manifest = write_manifest(root, cap=1)
    del manifest["prior_ledgers"]
    (root / "manifest.json").write_text(json.dumps(manifest))
    for name, charged in (("debater_baseline_gemma4", 0.2), ("debugging/rehearsal_gemma_motion01", 0.9)):
        directory = tmp_path / name
        directory.mkdir(parents=True)
        with sqlite3.connect(directory / "cost.sqlite") as connection:
            connection.execute("CREATE TABLE calls(estimate REAL, charged REAL)")
            connection.execute("INSERT INTO calls VALUES(?, ?)", (charged / 4, charged))
    transport = Mock(side_effect=AssertionError("No provider request should be dispatched"))
    app = load_script("gateway").create_app(root, {"TVLY_API_KEY": "offline"}, transport)

    async def scenario():
        async with await client_for(app) as client:
            status = await client.get("/status")
            assert status.status_code == 200
            assert status.json()["exposure_usd"] == pytest.approx(1.1)
            await set_label(client, 1)
            response = await client.post("/workers/motion_01/v1/chat/completions", json=completion_body())
            assert response.status_code == 402

    try:
        asyncio.run(scenario())
        result = load_script("monitor").snapshot(root)
        assert result["exposure_usd"] == pytest.approx(1.1)
        assert result["estimated_usd"] == pytest.approx(0.275)
        transport.assert_not_called()
    finally:
        app.state.database.close()
    assert not (tmp_path / "rehearsal_gemma_motion01").exists()


@pytest.mark.parametrize("prior_missing", [False, True])
def test_prior_exposure_or_unreadable_ledger_blocks_dispatch(gateway_factory, tmp_path, prior_missing):
    prior = tmp_path / "prior.sqlite"
    if not prior_missing:
        with sqlite3.connect(prior) as connection:
            connection.execute("CREATE TABLE calls(charged REAL)")
            connection.execute("INSERT INTO calls VALUES(0.05)")
    transport = Mock(side_effect=AssertionError("No transport is allowed"))
    app = gateway_factory(transport, cap=0.1, prior_ledgers=[str(prior)])

    async def scenario():
        async with await client_for(app) as client:
            await set_label(client, 1)
            response = await client.post("/workers/motion_01/search", json={"query": "offline"})
            assert response.status_code == (503 if prior_missing else 402)

    asyncio.run(scenario())
    transport.assert_not_called()
    assert (tmp_path / "STOP").exists()
    assert prior.exists() is not prior_missing


def test_unregistered_worker_and_unpriced_model_cannot_dispatch(gateway_factory):
    transport = Mock(side_effect=AssertionError("No transport is allowed"))
    app = gateway_factory(transport)

    async def scenario():
        async with await client_for(app) as client:
            assert (await client.post("/workers/motion_01/search", json={})).status_code == 400
            assert (await client.post("/workers/motion_99/label", json={"label": "motion_99:for"})).status_code == 400
            await set_label(client, 1)
            body = dict(completion_body(), model="unpriced")
            assert (await client.post("/workers/motion_01/v1/chat/completions", json=body)).status_code == 400

    asyncio.run(scenario())
    transport.assert_not_called()


@pytest.fixture
def controller_run(tmp_path):
    controller = load_script("controller")
    write_manifest(tmp_path)
    worker = tmp_path / "offline_worker.py"
    worker.write_text(
        "import json, os, time\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['TREE_MOTION_DIR'])\n"
        "(root / 'started.json').write_text(json.dumps(dict(pid=os.getpid(), time=time.time())))\n"
        "while not (root / 'release').exists():\n"
        "    time.sleep(0.01)\n"
        "if (root / 'FAIL').exists():\n"
        "    raise SystemExit(2)\n"
        "(root / 'status.json').write_text(json.dumps(dict(state='complete', motion=os.environ['TREE_MOTION'])))\n"
    )
    try:
        yield controller, worker
    finally:
        (tmp_path / "STOP").touch()


def test_controller_refills_four_slots_and_skips_completed_on_resume(controller_run, tmp_path):
    controller, script = controller_run
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller.run, tmp_path, motion_script=script, poll_seconds=0.01)
        try:
            wait_for(lambda: len(list((tmp_path / "motions").glob("*/started.json"))) == 4)
            assert not (tmp_path / "motions/motion_05/started.json").exists()
            (tmp_path / "motions/motion_01/release").touch()
            wait_for(lambda: (tmp_path / "motions/motion_05/started.json").exists())
            assert not (tmp_path / "motions/motion_06/started.json").exists()
            for index in range(1, 7):
                (tmp_path / f"motions/motion_{index:02d}/release").touch()
            status = future.result(timeout=10)
        finally:
            (tmp_path / "STOP").touch()
    assert status["state"] == "complete" and len(status["completed"]) == 6
    assert status["parallelism"] == 4 and not status["active"]
    starts = {path: path.read_text() for path in (tmp_path / "motions").glob("*/started.json")}
    (tmp_path / "STOP").unlink()
    resumed = controller.run(tmp_path, motion_script=script, poll_seconds=0.01)
    assert resumed["state"] == "complete"
    assert starts == {path: path.read_text() for path in starts}


def test_stop_terminates_all_workers_and_preserves_queued_work(controller_run, tmp_path):
    controller, script = controller_run
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(controller.run, tmp_path, motion_script=script, poll_seconds=0.01)
        try:
            wait_for(lambda: len(list((tmp_path / "motions").glob("*/started.json"))) == 4)
        finally:
            (tmp_path / "STOP").touch()
        status = future.result(timeout=10)
    assert status["state"] == "stopped" and len(status["interrupted"]) == 4
    assert status["queued"] == 2 and status["active"] == []
    for path in (tmp_path / "motions").glob("*/started.json"):
        process_id = json.loads(path.read_text())["pid"]
        assert not Path(f"/proc/{process_id}").exists()
    assert not (tmp_path / "motions/motion_05/started.json").exists()


def test_failed_and_timed_out_workers_do_not_block_other_motions(controller_run, tmp_path):
    controller, script = controller_run
    write_manifest(tmp_path, count=3)
    for index in (1, 3):
        directory = tmp_path / f"motions/motion_{index:02d}"
        directory.mkdir(parents=True)
        (directory / "release").touch()
    (tmp_path / "motions/motion_01/FAIL").touch()
    status = controller.run(tmp_path, parallelism=2, motion_script=script, poll_seconds=0.01, motion_timeout=1)
    assert status["state"] == "completed_with_failures"
    assert [item["index"] for item in status["completed"]] == [3]
    assert {item["index"] for item in status["failed"]} == {1, 2}
    assert next(item for item in status["failed"] if item["index"] == 2)["timed_out"]


@pytest.mark.parametrize("relative_script", [False, True])
def test_gateway_shutdown_matches_both_script_and_run_directory(tmp_path, monkeypatch, relative_script):
    controller = load_script("controller")
    process = tmp_path / "process"
    process.mkdir()
    repository = SCRIPTS.parents[2]
    (process / "cwd").symlink_to(repository, target_is_directory=True)
    script = "experiments/assets/remaining_semantic_trees/gateway.py" if relative_script else str(SCRIPTS / "gateway.py")
    arguments = ["python", script, "--run-dir", str(tmp_path)]
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in arguments))
    monkeypatch.setattr(controller, "Path", lambda value: process if value == "/proc/123" else Path(value))
    assert controller.gateway_process_matches(tmp_path, 123)
    assert not controller.gateway_process_matches(tmp_path / "other_run", 123)
    arguments[1] = "unrelated.py"
    (process / "cmdline").write_bytes(b"\0".join(value.encode() for value in arguments))
    assert not controller.gateway_process_matches(tmp_path, 123)


def test_gateway_shutdown_waits_for_reserved_requests(tmp_path, monkeypatch):
    controller = load_script("controller")
    with sqlite3.connect(tmp_path / "cost.sqlite") as connection:
        connection.execute("CREATE TABLE calls(status TEXT)")
        connection.execute("INSERT INTO calls VALUES('pending')")
    kill = Mock()
    monkeypatch.setattr(controller.os, "kill", kill)
    monkeypatch.setattr(controller, "gateway_process_matches", lambda root, process_id: True)
    controller.settle_gateway(tmp_path, 123, timeout=0)
    kill.assert_not_called()
    with sqlite3.connect(tmp_path / "cost.sqlite") as connection:
        connection.execute("UPDATE calls SET status='ok'")
    controller.settle_gateway(tmp_path, 123, timeout=0)
    kill.assert_called_once_with(123, controller.signal.SIGTERM)


def test_monitor_reports_prepared_without_charging_or_stopping(tmp_path):
    monitor = load_script("monitor")
    write_manifest(tmp_path)
    result = monitor.observe(tmp_path)
    assert result["state"] == "prepared" and result["estimated_usd"] is None
    assert not (tmp_path / "STOP").exists()
    assert not (tmp_path / "cost.sqlite").exists()


@pytest.mark.parametrize("failure", ["controller", "gateway", "budget", "ledger"])
def test_monitor_stops_on_process_or_accounting_failure(tmp_path, monkeypatch, failure):
    monitor = load_script("monitor")
    write_manifest(tmp_path, cap=0.1)
    (tmp_path / "status.json").write_text(json.dumps({"state": "running", "active": []}))
    (tmp_path / "controller.pid").write_text("123")
    (tmp_path / "gateway.pid").write_text("456")
    if failure != "ledger":
        with sqlite3.connect(tmp_path / "cost.sqlite") as connection:
            connection.execute("CREATE TABLE calls(estimate REAL,charged REAL,status TEXT)")
            connection.execute("INSERT INTO calls VALUES(0.01,?,'ok')", (0.1 if failure == "budget" else 0.01,))
    monkeypatch.setattr(monitor, "process_matches", lambda process_id, script, root: script.stem != failure)
    stop = Mock()
    monkeypatch.setattr(monitor, "stop_orphans", stop)
    result = monitor.observe(tmp_path)
    assert result["watchdog_stop"]
    assert (tmp_path / "STOP").exists()
    stop.assert_called_once_with(tmp_path)
