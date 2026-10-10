"""Ordered text batches triggered by word count, elapsed wait or final drain."""
import threading
import time


class TimedTextBatcher:
    """Serialize delivery while recognition and a single pending timer run independently.

    emit must enqueue work, not perform slow analysis. A timer generation prevents
    a cancelled callback from flushing a newer batch; close waits for any callback.
    """
    def __init__(self, config, emit, on_error, *, clock=time.monotonic, timer_factory=threading.Timer):
        self.config, self.emit, self.on_error = config, emit, on_error
        self.clock, self.timer_factory = clock, timer_factory
        self.lock = threading.RLock()
        self.pending, self.timer = [], None
        self.generation = 0
        self.closed = False
        self.first_seen = None

    def append(self, item):
        with self.lock:
            if self.closed:
                raise RuntimeError('Text batcher is closed')
            if not self.pending:
                self.first_seen = self.clock()
            self.pending.append(item)
            if sum(len(part['text'].split()) for part in self.pending) >= self.config.min_text_words:
                self._flush('threshold')
            elif self.timer is None and self.config.max_text_wait_seconds > 0:
                generation = self.generation
                self.timer = self.timer_factory(self.config.max_text_wait_seconds,
                                                lambda: self._timeout(generation))
                self.timer.daemon = True
                self.timer.start()

    def _timeout(self, generation):
        with self.lock:
            if self.closed or generation != self.generation:
                return
            try:
                self._flush('timeout')
            except BaseException as exc:
                self.closed = True
                self.on_error(exc)

    def _flush(self, reason):
        self.generation += 1
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if not self.pending:
            return
        members, self.pending = self.pending, []
        waited = self.clock() - self.first_seen
        self.first_seen = None
        self.emit(members, reason, waited)

    def close(self, *, drain=False):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if drain:
                self._flush('final')
            else:
                self.generation += 1
                if self.timer is not None:
                    self.timer.cancel()
                    self.timer = None
                self.pending.clear()
