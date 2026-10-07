"""Resident offline CPU encoder in an isolated process (no global torch tuning).

Model loading never downloads weights. Parent requests are serialized; startup and
material encoding belong to preparation, not the steady-state query budget.
"""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
from collections import OrderedDict

import numpy as np


class LocalEncoder:
    def __init__(self, model='sentence-transformers/all-MiniLM-L6-v2', threads=2):
        if type(threads) is not int or threads < 1:
            raise ValueError('encoder threads must be positive')
        self.model = model
        self.threads = threads
        self.process = None
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        atexit.register(self.close)

    def cache_identity(self):
        """Identify local weights/configuration without loading the model or downloading."""
        if hasattr(self, '_disk_identity'):
            return self._disk_identity
        import hashlib
        from importlib.metadata import version
        from huggingface_hub import snapshot_download
        path = Path(self.model).expanduser()
        if not path.is_dir():
            path = Path(snapshot_download(self.model, local_files_only=True))
        files = [(str(p.relative_to(path)), str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns)
                 for p in sorted(path.rglob('*')) if p.is_file()]
        self._disk_identity = dict(model=self.model, files=hashlib.sha256(json.dumps(files).encode()).hexdigest(),
                                   max_seq_length=256, device='cpu', threads=self.threads,
                                   versions={p:version(p) for p in ('sentence-transformers','transformers','torch')})
        return self._disk_identity

    def _start(self):
        if self.process is not None:
            return
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS=str(self.threads),
                   MKL_NUM_THREADS=str(self.threads))
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), self.model, str(self.threads)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env,
        )

    def warmup(self):
        """Keep first-query model startup in preparation even on a disk-index hit."""
        self.encode([''])

    def encode(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0,0),dtype=np.float32)
        with self.lock:
            missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
            if missing:
                self._start()
                try:
                    self.process.stdin.write(json.dumps(missing)+'\n')
                    self.process.stdin.flush()
                    ready, _, _ = select.select([self.process.stdout], [], [], 180)
                    if not ready:
                        raise TimeoutError('Local encoder did not respond within 180s')
                    line = self.process.stdout.readline()
                    if not line:
                        raise RuntimeError('Local encoder exited; check local model files')
                    response = json.loads(line)
                    if 'error' in response:
                        raise RuntimeError(response['error'])
                    vectors = np.asarray(response['vectors'],dtype=np.float32)
                    if vectors.ndim != 2 or len(vectors) != len(missing) or not np.isfinite(vectors).all():
                        raise ValueError('Invalid local encoder output')
                    self.cache.update(zip(missing, vectors))
                except BaseException:
                    self.close()
                    raise
            result = np.stack([self.cache[t] for t in texts])
            for t in texts:
                self.cache.move_to_end(t)
            while len(self.cache) > 8192:
                self.cache.popitem(last=False)
            return result

    def close(self):
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream:
                stream.close()


_ENCODERS = {}
_ENCODERS_LOCK = threading.Lock()


def get_local_encoder(model, threads=2):
    with _ENCODERS_LOCK:
        key = (model, threads)
        if key not in _ENCODERS:
            _ENCODERS[key] = LocalEncoder(model, threads)
        return _ENCODERS[key]


def _worker():
    # Offline flags are also set before transformers imports for direct invocation.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from contextlib import redirect_stdout
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError('Network is disabled in the local encoder')
    socket.socket.connect = blocked
    socket.create_connection = blocked
    try:
        with redirect_stdout(sys.stderr):
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(int(sys.argv[2]))
            model = SentenceTransformer(sys.argv[1], device='cpu', local_files_only=True)
            model.max_seq_length = 256
        for line in sys.stdin:
            try:
                texts = json.loads(line)
                with redirect_stdout(sys.stderr):
                    vectors = model.encode(texts, batch_size=32, show_progress_bar=False,
                                           convert_to_numpy=True, normalize_embeddings=True)
                print(json.dumps({'vectors':vectors.tolist()}),flush=True)
            except Exception as exc:
                print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}),flush=True)
    except Exception as exc:
        print(json.dumps({'error':f'Offline encoder unavailable: {type(exc).__name__}: {exc}'}),flush=True)


if __name__ == '__main__':
    _worker()
