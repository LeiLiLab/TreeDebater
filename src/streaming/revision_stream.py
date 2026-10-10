"""An append-only whole-speech revision feeding the existing TTS length loop.

Only complete paragraphs cross the handoff. Splitting never depends on text
that has not arrived, so a paragraph cannot later move to another audio chunk.
"""
import re
import threading
import time

from utils.speech_text import clean_spoken_revision
from utils.tool import remove_citation, remove_subtitles
from .full_speech import remaining_text


class RevisionStream:
    def __init__(self, *, prefix, draft, n_words, config, on_chunk=None):
        self.prefix, self.config = prefix, config
        self._on_chunk = on_chunk
        chars_per_word = len(draft) / max(len(draft.split()), 1)
        self.expected_chars = max(len(draft), n_words * chars_per_word)
        self._condition = threading.Condition()
        self._raw = ''
        self._paragraphs = []
        self._pending = ''
        self._ready = []
        self._produced_chars = 0
        self._finished = False
        self._error = None
        self._started = time.perf_counter()
        self._first_ready = None
        self._finished_at = None
        self.reference = ''

    def _emit(self, text):
        from tts_streaming import _pack_sentences
        parts = (_pack_sentences(text, self.config.max_chunk_chars)
                 if len(text) > self.config.max_chunk_chars else [text])
        for part in parts:
            self._ready.append(part)
            self._produced_chars += len(part)
            if self._on_chunk is not None:
                self._on_chunk(part)
        if self._first_ready is None:
            self._first_ready = time.perf_counter() - self._started
        self._condition.notify_all()

    def _advance(self, *, final=False):
        raw = self._raw.replace('\r\n', '\n').replace('\r', '\n')
        if not raw.strip():
            if final:
                raise ValueError('Revision omitted the remaining speech')
            return
        # JSON/fenced legacy responses require full-envelope decoding. Never
        # send their syntax or an incomplete JSON string to speech synthesis.
        if not final and raw.lstrip().startswith(('{', '[', '`')):
            return
        spoken = clean_spoken_revision(raw)
        if self.prefix:
            spoken = remaining_text(spoken, self.prefix)
        spoken, self.reference = remove_citation(spoken)
        spoken = remove_subtitles(spoken)
        paragraphs = [p.strip() for p in re.split(r'\n[ \t]*\n', spoken) if p.strip()]
        if not final and not re.search(r'\n[ \t]*\n[ \t]*$', raw):
            paragraphs = paragraphs[:-1]
        if paragraphs[:len(self._paragraphs)] != self._paragraphs:
            raise ValueError('Streaming revision changed an already released paragraph')
        for paragraph in paragraphs[len(self._paragraphs):]:
            self._paragraphs.append(paragraph)
            self._pending = '\n\n'.join(filter(None, (self._pending, paragraph)))
            if len(self._pending.split()) >= self.config.min_chunk_words:
                self._emit(self._pending)
                self._pending = ''
        if final:
            if self._pending:
                self._emit(self._pending)
                self._pending = ''
            if not self._paragraphs:
                raise ValueError('Revision omitted the remaining speech')

    def feed(self, delta):
        if not isinstance(delta, str):
            raise TypeError('Revision deltas must be text')
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._finished:
                raise ValueError('Revision stream already finished')
            self._raw += delta
            try:
                self._advance()
            except BaseException as exc:
                self.fail(exc)
                raise

    def finish(self, text):
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._finished:
                if text != self._raw:
                    raise ValueError('Final revision differs from streamed text')
                return
            try:
                if self._raw and text != self._raw:
                    raise ValueError('Final revision differs from streamed text')
                self._raw = text  # Also supports a non-streaming helper implementation.
                self._advance(final=True)
                self._finished = True
                self._finished_at = time.perf_counter() - self._started
                self._condition.notify_all()
            except BaseException as exc:
                self.fail(exc)
                raise

    def fail(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()

    def read(self, *, wait=False):
        """Drain ready chunks and estimate only the not-yet-arrived remainder."""
        with self._condition:
            while wait and not self._ready and not self._finished and self._error is None:
                self._condition.wait()
            if self._error is not None:
                raise self._error
            chunks, self._ready = self._ready, []
            # Until EOF, the last available chunk must not consume all remaining
            # time. Once EOF arrives, the native exact proportional budget wins.
            reserve = min(self.config.max_chunk_chars, max(1, self.expected_chars * .1))
            unseen = 0 if self._finished else max(reserve, self.expected_chars - self._produced_chars)
            return chunks, self._finished, unseen

    def snapshot(self):
        with self._condition:
            return dict(finished=self._finished, paragraphs=len(self._paragraphs),
                        first_chunk_ready_seconds=self._first_ready,
                        text_complete_seconds=self._finished_at,
                        error=type(self._error).__name__ if self._error is not None else None)
