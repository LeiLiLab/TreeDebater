"""One unpublished first-body synthesis, owned and settled by one speech."""
from concurrent.futures import ThreadPoolExecutor
import threading
import time


class FirstBodyAudio:
    def __init__(self, config, start):
        self.config = config
        self.start = start
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='first-body-audio')
        self._future = None
        self._key = None
        self._trace = dict(status='not_prepared', matched=False, reused=False)

    def _update(self, **values):
        with self._lock:
            self._trace.update(values)

    def prepare(self, tail, remaining):
        """Start at most one optional request; never publish or wait for audio."""
        import tts_streaming as tts
        try:
            segments = tts.split_body_chunks(tail, remaining, self.config)
            if not segments or self._future is not None:
                return
            self.prepare_chunk(segments[0])
        except Exception as exc:
            self._update(status='failed', error=f'{type(exc).__name__}: {exc}')

    def prepare_chunk(self, text):
        """Pre-synthesize an already-final stream boundary without re-splitting."""
        import tts_streaming as tts
        try:
            if self._future is not None:
                return
            tts._validate_tts_input(text)
            key = (text, self.config.voice, self.config.model, self.config.tts_backend)
            self._update(status='running', text=text, start_seconds=time.perf_counter() - self.start)

            def synthesize():
                client = None
                try:
                    client = tts.create_tts_client(self.config)
                    result = tts.synthesize_audio(client, text, self.config)
                    self._update(status='ready', ready_seconds=time.perf_counter() - self.start)
                    return result
                except Exception as exc:
                    self._update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    if client is not None:
                        client.close()

            # A consumer sees either no offer or a fully registered future.
            with self._lock:
                if self._future is not None:
                    return
                self._key = key
                self._future = self._pool.submit(synthesize)
        except Exception as exc:
            self._update(status='failed', error=f'{type(exc).__name__}: {exc}')

    def match(self, text, voice, model, tts_backend="openai"):
        """Compare against the FINAL chunk (after native splitting/early cuts)."""
        with self._lock:
            matched = self._future is not None and self._key == (text, voice, model, tts_backend)
            self._trace['matched'] = matched
            return self._future if matched else None

    def mark_reused(self):
        self._update(reused=True)

    def close(self):
        # Discarded/failed requests still belong to this speech and its accounting.
        self._pool.shutdown(wait=True, cancel_futures=True)

    def snapshot(self):
        with self._lock:
            return dict(self._trace)
