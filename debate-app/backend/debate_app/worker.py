"""One serial model executor in a killable process; service ingestion never waits on it."""
import asyncio
import multiprocessing as mp
import os
import queue
import signal
import traceback
import uuid


def worker_main(config, root, commands, results):
    # Apply before any engine / torch imports, only inside this app's child process.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if os.name == "posix":
        os.setsid()
    from .engine_adapter import DemoEngine, TreeDebaterEngine

    engine = (DemoEngine if config["engine"] == "demo" else TreeDebaterEngine)(
        config, root
    )
    while True:
        item = commands.get()
        if item is None:
            return
        key, operation, args = item
        try:
            if operation == "generate":
                args["emit"] = lambda chunk: results.put((key, "chunk", chunk))
            value = getattr(engine, operation)(**args)
            results.put((key, "result", value))
        except Exception as e:
            traceback.print_exc()
            results.put((key, "error", f"{type(e).__name__}: {e}"))


class Worker:
    def __init__(self, config, root):
        context = mp.get_context("spawn")
        self.commands, self.results = context.Queue(16), context.Queue(64)
        self.process = context.Process(
            target=worker_main,
            args=(config, str(root), self.commands, self.results),
            daemon=True,
        )
        self.process.start()
        self.lock = asyncio.Lock()
        self.closed = False
        self.stop_task = None
        self.timeout = config["model_timeout_seconds"]

    async def call(self, operation, on_chunk=None, **args):
        async with self.lock:
            if self.closed:
                raise RuntimeError("Debate worker has stopped")
            key = uuid.uuid4().hex
            self.commands.put_nowait((key, operation, args))
            deadline = asyncio.get_running_loop().time() + self.timeout
            while True:
                if self.closed or not self.process.is_alive():
                    await self.stop()
                    raise RuntimeError("Debate worker exited")
                if asyncio.get_running_loop().time() >= deadline:
                    await self.stop()
                    raise TimeoutError(
                        f"{operation} exceeded the model timeout; the worker was stopped"
                    )
                try:
                    response_key, kind, value = self.results.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.025)
                    continue
                if response_key != key:
                    raise RuntimeError("Unexpected worker reply")
                if kind == "chunk":
                    if on_chunk:
                        await on_chunk(value)
                elif kind == "error":
                    raise RuntimeError(value)
                else:
                    return value

    async def stop(self):
        if self.stop_task is None:
            self.stop_task = asyncio.create_task(self._stop())
        await asyncio.shield(self.stop_task)

    async def _stop(self):
        self.closed = True
        try:
            self.commands.put_nowait(None)
        except queue.Full:
            pass
        await asyncio.to_thread(self.process.join, 0.2)
        if self.process.is_alive():
            if os.name == "posix":
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    self.process.terminate()
            else:
                self.process.terminate()
            await asyncio.to_thread(self.process.join, 1)
        if self.process.is_alive():
            self.process.kill()
            await asyncio.to_thread(self.process.join, 1)
        if self.process.is_alive():
            raise RuntimeError("Debate worker did not exit after being killed")
        for channel in [self.commands, self.results]:
            channel.cancel_join_thread()
            channel.close()
