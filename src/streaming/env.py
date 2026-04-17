"""
Streaming input environment: watch a directory for chunked audio, transcribe,
buffer text, and incrementally update a TreeDebater's debate trees.

This mirrors debate-anonymous ``chunked_pipeline.py`` (watch → min-audio buffer →
ASR → min-text buffer → analyze) but routes tree updates through
``TreeDebater._analyze_statement`` instead of a standalone ``analyze_statement``.

**Full debate + streaming listen**

Use ``StreamingDebateEnv`` (same YAML as ``env.py``): runs ``Env.play()`` stages, and on each
speech turn starts ``StreamingInputEnv`` on the **listener** (opponent TreeDebater), runs the
speaker’s generation + TTS into ``log_files/N_outputs``, then streams that MP3 into per-turn
watch subdirs so the listener does streaming ASR + ``_analyze_statement`` while chunks arrive.

CLI (from ``TreeDebater/src``)::

    python -m streaming.env --debate --config configs/base_st_io.yml

Watch-only (listener alone, no ``env.py``)::

    python -m streaming.env --config ... --watch-dir /tmp/w --debater-side for

Requires: ``pydub``, ``openai`` (Whisper), and API keys as in ``utils.constants``.

Run scripts from ``TreeDebater/src``, or set ``PYTHONPATH`` to include ``src``.

When run as ``python -m streaming.env``, debate ``env`` is imported before ``utils.tool``
where applicable so logging matches ``env.py``.
"""

from __future__ import annotations

import sys

import argparse
import json
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from pydub import AudioSegment

from utils.constants import CLOSING_TIME, OPENING_TIME, REBUTTAL_TIME
from utils.tool import logger


def transcribe_audio_segment(segment: AudioSegment, audio_format: str = "mp3") -> str:
    """Transcribe a pydub ``AudioSegment`` using OpenAI Whisper (``whisper-1``)."""
    buf = BytesIO()
    segment.export(buf, format=audio_format)
    buf.seek(0)
    buf.name = f"audio.{audio_format}"
    client = OpenAI()
    transcript = client.audio.transcriptions.create(model="whisper-1", file=buf, language="en")
    return (transcript.text or "").strip()


def _log_id_from_filename(filename: str) -> str:
    """Match ``chunked_pipeline`` grouping: stem before first underscore."""
    return Path(filename).stem.split("_")[0]


def opponent_side(side: str) -> str:
    """Return the other debate side."""
    if side == "for":
        return "against"
    if side == "against":
        return "for"
    raise ValueError(f"side must be 'for' or 'against', got {side!r}")


@dataclass
class StreamingInputConfig:
    watch_dir: Path
    motion: str
    stage: str  # opening | rebuttal | closing — sets TreeDebater.status for analysis
    statement_side: str  # side whose speech is in the audio (here: opponent of debater.side)
    min_audio_seconds: float = 30.0
    min_text_words: int = 50
    poll_interval: float = 1.0
    audio_file_glob: str = "*.mp3"
    max_audio_wait_seconds: Optional[float] = None
    max_text_wait_seconds: Optional[float] = None
    max_total_audio_seconds: Optional[float] = None
    audio_format: str = "mp3"
    playback_cursor: Optional[List[float]] = None  # shared cursor for real-time synchronization


class StreamingInputEnv:
    """
    Watch ``watch_dir`` for new audio files, buffer by duration and text by word
    count, then call ``TreeDebater._analyze_statement`` on each flushed text batch.

    Configure ``statement_side`` as the **speaker** in the audio (typically the
    opponent of ``debater.side``). Read updated trees from ``self.debater`` anytime.

    Intended for one ``log_id`` stream (e.g. ``speaker_chunk001.mp3``); multiple
    log_ids are supported with independent buffers like ``chunked_pipeline``.
    """

    def __init__(self, debater: "TreeDebater", config: StreamingInputConfig) -> None:
        self.debater = debater
        self.config = config
        self._tree_lock = threading.Lock()
        self._stop = threading.Event()

        if not debater.use_debate_flow_tree:
            logger.warning(
                "[StreamingInputEnv] use_debate_flow_tree is False; "
                "_analyze_statement will no-op. Enable debate flow tree for updates."
            )

    def stop(self) -> None:
        self._stop.set()

    def _flush_text(self, chunks: List[str]) -> None:
        merged = " ".join(chunks).strip()
        if not merged:
            return
        self.debater.status = self.config.stage
        wc = len(merged.split())
        logger.debug(f"[StreamingInputEnv] tree_update_start words={wc} text_preview={merged[:50]}... t={time.time():.3f}")
        tree_start = time.time()
        with self._tree_lock:
            self.debater._analyze_statement(merged, self.config.statement_side)
        tree_end = time.time()
        logger.debug(f"[StreamingInputEnv] tree_update_end words={wc} update_time={tree_end - tree_start:.3f}s t={tree_end:.3f}")
        logger.info(
            f"[StreamingInputEnv] analyze_statement flush: side={self.config.statement_side} "
            f"stage={self.config.stage} words={wc} chunks={len(chunks)}"
        )

    def run(self) -> None:
        """Block until :meth:`stop` or ``KeyboardInterrupt``."""
        cfg = self.config
        watch_dir = Path(cfg.watch_dir)
        watch_dir.mkdir(parents=True, exist_ok=True)

        min_audio_ms = int(cfg.min_audio_seconds * 1000)
        cap_audio_ms = (
            int(cfg.max_total_audio_seconds * 1000)
            if cfg.max_total_audio_seconds and cfg.max_total_audio_seconds > 0
            else None
        )

        seen: set = set()
        audio_buf: Dict[str, List[Tuple[str, Any]]] = {}
        audio_first_seen: Dict[str, float] = {}
        emitted_ms: Dict[str, int] = {}
        completed: set = set()

        text_buf: Dict[str, List[str]] = {}
        text_first_seen: Dict[str, float] = {}

        use_cursor = cfg.playback_cursor is not None
        continuous_file = watch_dir / "continuous_audio.mp3" if use_cursor else None

        logger.debug(f"[StreamingInputEnv] thread_start stage={cfg.stage} statement_side={cfg.statement_side} cursor_mode={use_cursor} t={time.time():.3f}")
        logger.info(
            f"[StreamingInputEnv] watching {watch_dir} ({cfg.audio_file_glob}), "
            f"min_audio_s={cfg.min_audio_seconds}, min_text_words={cfg.min_text_words}, "
            f"statement_side={cfg.statement_side}, stage={cfg.stage}, "
            f"cursor_mode={use_cursor}"
        )

        try:
            while not self._stop.is_set():
                if use_cursor and continuous_file and continuous_file.is_file():
                    # Cursor mode: read from continuous audio up to cursor position
                    try:
                        available_audio = self._read_audio_up_to_cursor(continuous_file, cfg.playback_cursor[0])
                        if available_audio is not None and len(available_audio) > 0:
                            self._process_cursor_audio(
                                available_audio,
                                emitted_ms,
                                min_audio_ms,
                                cap_audio_ms,
                                text_buf,
                                text_first_seen,
                            )
                    except Exception as e:
                        logger.warning(f"[StreamingInputEnv] Cursor audio processing failed: {e}")
                else:
                    # Chunk mode: original behavior
                    try:
                        paths = sorted(watch_dir.glob(cfg.audio_file_glob), key=lambda p: p.stat().st_mtime)
                    except Exception:
                        time.sleep(cfg.poll_interval)
                        continue

                    for path in paths:
                        path_str = str(path.resolve())
                        if path_str in seen:
                            continue
                        try:
                            seg = AudioSegment.from_file(path_str)
                        except Exception as e:
                            logger.warning(f"[StreamingInputEnv] Skip unreadable audio {path_str}: {e}")
                            continue
                        seen.add(path_str)
                        log_id = _log_id_from_filename(path.name)
                        if log_id in completed:
                            continue
                        audio_buf.setdefault(log_id, []).append((path_str, seg))
                        if log_id not in audio_first_seen:
                            audio_first_seen[log_id] = time.perf_counter()

                    for log_id in list(audio_buf.keys()):
                        self._process_one_log_audio(
                            log_id,
                            audio_buf,
                            audio_first_seen,
                            emitted_ms,
                            completed,
                            min_audio_ms,
                            cap_audio_ms,
                            text_buf,
                            text_first_seen,
                        )

                self._idle_text_flush(text_buf, text_first_seen)
                time.sleep(cfg.poll_interval)
        except KeyboardInterrupt:
            logger.info("[StreamingInputEnv] KeyboardInterrupt; stopping.")
        finally:
            for log_id in list(text_buf.keys()):
                if text_buf.get(log_id):
                    self._flush_text(text_buf[log_id])
                    text_buf[log_id] = []
            logger.debug(f"[StreamingInputEnv] thread_end stage={cfg.stage} statement_side={cfg.statement_side} t={time.time():.3f}")

    def _idle_text_flush(self, text_buf: Dict[str, List[str]], text_first_seen: Dict[str, float]) -> None:
        cfg = self.config
        if cfg.max_text_wait_seconds is None or cfg.max_text_wait_seconds <= 0:
            return
        now = time.perf_counter()
        for log_id in list(text_buf.keys()):
            if log_id not in text_first_seen or not text_buf.get(log_id):
                continue
            if now - text_first_seen[log_id] >= cfg.max_text_wait_seconds:
                total_words = sum(len(s.split()) for s in text_buf[log_id])
                logger.info(
                    f"[StreamingInputEnv] Max text wait exceeded for log_id={log_id!r} "
                    f"(words={total_words}); flushing."
                )
                self._flush_text(text_buf[log_id])
                text_buf[log_id] = []
                text_first_seen.pop(log_id, None)

    def _process_one_log_audio(
        self,
        log_id: str,
        audio_buf: Dict[str, List[Tuple[str, Any]]],
        audio_first_seen: Dict[str, float],
        emitted_ms: Dict[str, int],
        completed: set,
        min_audio_ms: int,
        cap_audio_ms: Optional[int],
        text_buf: Dict[str, List[str]],
        text_first_seen: Dict[str, float],
    ) -> None:
        cfg = self.config
        acc_ms = 0
        to_emit: List[Tuple[str, Any]] = []
        while audio_buf.get(log_id) and acc_ms < min_audio_ms:
            path_str, seg = audio_buf[log_id].pop(0)
            to_emit.append((path_str, seg))
            acc_ms += len(seg)

        elapsed = None
        if log_id in audio_first_seen:
            elapsed = time.perf_counter() - audio_first_seen[log_id]

        should_emit = False
        wait_time_exceeded = False
        total_so_far_ms = emitted_ms.get(log_id, 0)

        if to_emit and acc_ms >= min_audio_ms:
            should_emit = True
        elif to_emit and cfg.max_audio_wait_seconds and cfg.max_audio_wait_seconds > 0:
            if elapsed is not None and elapsed >= cfg.max_audio_wait_seconds:
                should_emit = True
                wait_time_exceeded = True
                logger.info(
                    f"[StreamingInputEnv] Max audio wait exceeded for log_id={log_id!r}; "
                    f"emitting partial audio {acc_ms / 1000.0:.2f}s."
                )

        if should_emit and cap_audio_ms is not None:
            if total_so_far_ms >= cap_audio_ms:
                should_emit = False
                audio_buf[log_id] = []
                completed.add(log_id)
                audio_first_seen.pop(log_id, None)
                logger.info(f"[StreamingInputEnv] Total audio cap already reached for log_id={log_id!r}; dropping rest.")
            elif total_so_far_ms + acc_ms > cap_audio_ms:
                logger.info(f"[StreamingInputEnv] Total audio cap crossed for log_id={log_id!r} on this batch.")

        if not should_emit:
            for item in reversed(to_emit):
                audio_buf.setdefault(log_id, []).insert(0, item)
            if not audio_buf.get(log_id):
                del audio_buf[log_id]
                audio_first_seen.pop(log_id, None)
            return

        combined = to_emit[0][1]
        for _, s in to_emit[1:]:
            combined += s

        force_flush = wait_time_exceeded or (
            cap_audio_ms is not None and (total_so_far_ms + acc_ms) >= cap_audio_ms
        )

        try:
            text = transcribe_audio_segment(combined, audio_format=cfg.audio_format)
        except Exception as e:
            logger.warning(f"[StreamingInputEnv] Transcription failed for log_id={log_id!r}: {e}")
            for item in reversed(to_emit):
                audio_buf.setdefault(log_id, []).insert(0, item)
            return

        emitted_ms[log_id] = total_so_far_ms + acc_ms
        logger.info(
            f"[StreamingInputEnv] Transcribed log_id={log_id!r} batch_s={len(combined) / 1000.0:.2f} "
            f"text_words={len(text.split())} force_flush={force_flush}"
        )

        if cap_audio_ms is not None and emitted_ms[log_id] >= cap_audio_ms:
            completed.add(log_id)
            logger.info(f"[StreamingInputEnv] Total audio cap reached for log_id={log_id!r}; future audio ignored.")

        self._append_transcript_text(log_id, text, force_flush, text_buf, text_first_seen)

        if not audio_buf.get(log_id):
            del audio_buf[log_id]
            audio_first_seen.pop(log_id, None)

    def _append_transcript_text(
        self,
        log_id: str,
        text: str,
        force_flush: bool,
        text_buf: Dict[str, List[str]],
        text_first_seen: Dict[str, float],
    ) -> None:
        cfg = self.config
        if not text.strip():
            return
        text_buf.setdefault(log_id, []).append(text)
        if log_id not in text_first_seen:
            text_first_seen[log_id] = time.perf_counter()

        total_words = sum(len(s.split()) for s in text_buf[log_id])
        elapsed = time.perf_counter() - text_first_seen[log_id]
        should_flush = False
        if total_words >= cfg.min_text_words:
            should_flush = True
        elif cfg.max_text_wait_seconds and cfg.max_text_wait_seconds > 0 and elapsed >= cfg.max_text_wait_seconds:
            should_flush = True
            logger.info(f"[StreamingInputEnv] Max text wait exceeded for log_id={log_id!r}; flushing ({total_words} words).")
        if force_flush:
            should_flush = True
            logger.info(f"[StreamingInputEnv] Force flush for log_id={log_id!r} ({total_words} words).")

        if should_flush:
            self._flush_text(text_buf[log_id])
            text_buf[log_id] = []
            text_first_seen.pop(log_id, None)

    def _read_audio_up_to_cursor(self, continuous_file: Path, cursor_seconds: float) -> Optional[AudioSegment]:
        """Read continuous audio file up to the cursor position."""
        try:
            read_start = time.time()
            full_audio = AudioSegment.from_file(str(continuous_file))
            cursor_ms = int(cursor_seconds * 1000)
            if cursor_ms <= 0:
                return None
            result = full_audio[:cursor_ms]
            read_end = time.time()
            logger.debug(f"[StreamingInputEnv] file_read cursor={cursor_seconds:.2f}s "
                        f"available={len(full_audio)/1000.0:.2f}s read_time={read_end - read_start:.3f}s t={read_end:.3f}")
            return result
        except Exception as e:
            logger.warning(f"[StreamingInputEnv] Failed to read continuous audio: {e}")
            return None

    def _process_cursor_audio(
        self,
        available_audio: AudioSegment,
        emitted_ms: Dict[str, int],
        min_audio_ms: int,
        cap_audio_ms: Optional[int],
        text_buf: Dict[str, List[str]],
        text_first_seen: Dict[str, float],
    ) -> None:
        """Process audio from cursor-based continuous file."""
        cfg = self.config
        log_id = "continuous"

        available_ms = len(available_audio)
        already_processed_ms = emitted_ms.get(log_id, 0)

        # Check if there's new audio to process
        if available_ms <= already_processed_ms:
            return

        # Check cap
        if cap_audio_ms is not None and already_processed_ms >= cap_audio_ms:
            return

        # Extract new audio segment
        new_audio = available_audio[already_processed_ms:]
        new_audio_ms = len(new_audio)

        # Check if we have enough audio to process
        if new_audio_ms < min_audio_ms:
            logger.debug(f"[StreamingInputEnv] wait_audio_accumulation available={new_audio_ms/1000.0:.2f}s "
                        f"need={min_audio_ms/1000.0:.2f}s t={time.time():.3f}")
            return

        # Decide how much to process (in chunks of min_audio_ms)
        to_process_ms = (new_audio_ms // min_audio_ms) * min_audio_ms
        if to_process_ms == 0:
            return

        # Apply cap if needed
        if cap_audio_ms is not None:
            remaining_cap = cap_audio_ms - already_processed_ms
            to_process_ms = min(to_process_ms, remaining_cap)

        segment_to_transcribe = new_audio[:to_process_ms]
        audio_start_sec = already_processed_ms / 1000.0
        audio_end_sec = (already_processed_ms + to_process_ms) / 1000.0

        logger.debug(f"[StreamingInputEnv] asr_start audio_range={audio_start_sec:.2f}-{audio_end_sec:.2f}s t={time.time():.3f}")
        asr_start = time.time()
        try:
            text = transcribe_audio_segment(segment_to_transcribe, audio_format=cfg.audio_format)
            asr_end = time.time()
            asr_duration = asr_end - asr_start
            audio_duration = to_process_ms / 1000.0
            logger.debug(f"[StreamingInputEnv] asr_end audio_range={audio_start_sec:.2f}-{audio_end_sec:.2f}s "
                        f"text_len={len(text)} asr_time={asr_duration:.3f}s t={asr_end:.3f}")

            emitted_ms[log_id] = already_processed_ms + to_process_ms
            logger.info(
                f"[StreamingInputEnv] Cursor transcribed s={len(segment_to_transcribe) / 1000.0:.2f} "
                f"words={len(text.split())} total_processed={emitted_ms[log_id] / 1000.0:.2f}s"
            )

            force_flush = cap_audio_ms is not None and emitted_ms[log_id] >= cap_audio_ms
            self._append_transcript_text(log_id, text, force_flush, text_buf, text_first_seen)
        except Exception as e:
            logger.warning(f"[StreamingInputEnv] Cursor transcription failed: {e}")


def load_treedebater_for_side(config_path: Path, side: str) -> Tuple[dict, "TreeDebater"]:
    from agents import DebaterConfig
    from ouragents import TreeDebater

    with config_path.open("r", encoding="utf-8") as f:
        full = yaml.load(f, Loader=yaml.FullLoader)
    debater_cfgs = full["debater"]
    chosen = None
    for d in debater_cfgs:
        if d.get("side") == side and d.get("type") == "treedebater":
            chosen = d
            break
    if chosen is None:
        raise ValueError(
            f"No treedebater entry with side={side!r} in {config_path}. "
            "Streaming tree updates require a TreeDebater config for that side."
        )
    motion = full["env"]["motion"]
    debater = TreeDebater(DebaterConfig(**chosen), motion=motion)
    return full, debater


def tts_outputs_dir_from_log() -> Path:
    """Resolved ``log_files/<N>_outputs`` for the current debate log session."""
    from utils.tool import log_file_path

    if not log_file_path:
        return Path("log_files") / "1_outputs"
    lp = Path(log_file_path)
    return (lp.parent / f"{lp.stem}_outputs").resolve()


def default_watch_root_from_log() -> Path:
    """Resolved ``log_files/<N>_watch`` for per-turn streaming watch directories."""
    from utils.tool import log_file_path

    if not log_file_path:
        return Path("log_files") / "1_watch"
    lp = Path(log_file_path)
    return (lp.parent / f"{lp.stem}_watch").resolve()


# Backward-compatible names
_tts_outputs_dir_from_log = tts_outputs_dir_from_log
_default_watch_root_from_log = default_watch_root_from_log


class StreamingDebateEnv:
    """
    Full debate (``Env``) plus streaming ASR for the **listener** on each speech turn.

    When the opponent is a ``TreeDebater``, starts ``StreamingInputEnv`` before the speaker
    generates; after TTS writes ``{type}_{stage}_{side}.mp3`` under ``log_files/N_outputs``,
    splits that file into timed chunks and streams them into a per-turn watch dir so the
    listener updates trees incrementally while the debate proceeds.
    """

    def __init__(
        self,
        env_config,
        debug: bool,
        watch_root: Optional[Path] = None,
        *,
        min_audio_seconds: float = 30.0,
        min_text_words: int = 50,
        poll_interval: float = 1.0,
        audio_format: str = "mp3",
        split_mode: str = "fixed",
        chunk_seconds: float = 10.0,
        silence_window_seconds: float = 0.7,
        max_audio_wait_seconds: float = 0.0,
        max_text_wait_seconds: float = 0.0,
        max_total_audio_seconds: float = 0.0,
        listener_join_timeout: float = 300.0,
        min_playback_increment: float = 3.0,
    ) -> None:
        from env import Env

        self._env = Env(env_config, debug)
        self._listener_join_timeout = listener_join_timeout
        self._watch_root = Path(watch_root).resolve() if watch_root else default_watch_root_from_log()
        self._watch_root.mkdir(parents=True, exist_ok=True)
        self._min_audio_seconds = min_audio_seconds
        self._min_text_words = min_text_words
        self._poll_interval = poll_interval
        self._audio_format = audio_format
        self._split_mode = split_mode
        self._chunk_seconds = chunk_seconds
        self._silence_window_seconds = silence_window_seconds
        self._max_audio_wait = max_audio_wait_seconds if max_audio_wait_seconds > 0 else None
        self._max_text_wait = max_text_wait_seconds if max_text_wait_seconds > 0 else None
        self._max_total_audio = max_total_audio_seconds if max_total_audio_seconds > 0 else None
        self._min_playback_increment = min_playback_increment

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def _clear_dir(self, d: Path) -> None:
        if not d.exists():
            return
        for p in d.iterdir():
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass

    def _play_speech_turn(self, stage_key: str, side: str, max_time: float, generate_fn: Callable[[], str]) -> None:
        from .chunk_audio import split_audio, stream_chunks_to_directory

        listener = opponent_side(side)
        player = self._env.debaters[side]
        listener_deb = self._env.debaters[listener]

        if listener_deb.type != "treedebater":
            response = generate_fn()
            self._env.debate_process.append({"stage": stage_key, "side": side, "content": response})
            return

        turn_watch = self._watch_root / f"{stage_key}_{side}"
        turn_watch.mkdir(parents=True, exist_ok=True)
        self._clear_dir(turn_watch)

        sic = StreamingInputConfig(
            watch_dir=turn_watch,
            motion=self._env.motion,
            stage=stage_key,
            statement_side=side,
            min_audio_seconds=self._min_audio_seconds,
            min_text_words=self._min_text_words,
            poll_interval=self._poll_interval,
            audio_file_glob=f"*.{self._audio_format}",
            max_audio_wait_seconds=self._max_audio_wait,
            max_text_wait_seconds=self._max_text_wait,
            max_total_audio_seconds=self._max_total_audio,
            audio_format=self._audio_format,
        )
        sin_env = StreamingInputEnv(listener_deb, sic)
        listener_thread = threading.Thread(target=sin_env.run, name="StreamingInputEnv", daemon=True)
        listener_thread.start()
        time.sleep(max(0.5, self._poll_interval))

        response: Optional[str] = None
        try:
            response = generate_fn()
        finally:
            try:
                mp3_path = tts_outputs_dir_from_log() / f"{player.config.type}_{player.status}_{player.side}.mp3"
                if mp3_path.is_file() and mp3_path.stat().st_size > 2048:
                    audio = AudioSegment.from_file(str(mp3_path))
                    if self._split_mode == "fixed":
                        chunks = split_audio(audio, mode="fixed", time_seconds=self._chunk_seconds)
                    else:
                        chunks = split_audio(
                            audio, mode="silence", time_seconds=self._silence_window_seconds
                        )
                    logger.info(
                        f"[StreamingDebateEnv] Streaming {len(chunks)} chunk(s) from {mp3_path.name} → {turn_watch} "
                        f"(listener={listener!r})"
                    )
                    stream_chunks_to_directory(
                        chunks,
                        turn_watch,
                        side,
                        audio_format=self._audio_format,
                        dry_run=False,
                        max_total_seconds=None,
                        chunk_index_start=1,
                    )
                else:
                    logger.warning(f"[StreamingDebateEnv] No TTS MP3 at {mp3_path} (skip chunk stream for listener).")
            except Exception as e:
                logger.error(f"[StreamingDebateEnv] Chunk stream failed: {e}")
            sin_env.stop()
            listener_thread.join(timeout=self._listener_join_timeout)

        if response is not None:
            self._env.debate_process.append({"stage": stage_key, "side": side, "content": response})

    def play(self, pre_only: bool = False) -> None:
        order = ["for", "against"] if not self._env.reverse else ["against", "for"]
        for stage in ["preparation", "opening", "rebuttal", "closing"]:
            logger.info(f"[{stage}] Start")
            if stage == "preparation":
                for side in order:
                    if self._env.debaters[side].type in ["treedebater"]:
                        self._env.debaters[side].claim_generation(self._env.claim_pool_size, temperature=1)
            elif stage == "opening":
                for side in order:

                    def _gen_opening(side=side):
                        return self._env.debaters[side].opening_generation(
                            history=self._env.debate_process[1:],
                            max_time=OPENING_TIME,
                            time_control=self._env.time_control,
                            streaming_tts=getattr(self._env.debaters[side].config, "streaming_tts", False),
                        )

                    self._play_speech_turn("opening", side, OPENING_TIME, _gen_opening)
            elif stage == "rebuttal":
                for side in order:

                    def _gen_rebuttal(side=side):
                        return self._env.debaters[side].rebuttal_generation(
                            history=self._env.debate_process[1:],
                            max_time=REBUTTAL_TIME,
                            time_control=self._env.time_control,
                            streaming_tts=getattr(self._env.debaters[side].config, "streaming_tts", False),
                        )

                    self._play_speech_turn("rebuttal", side, REBUTTAL_TIME, _gen_rebuttal)
            elif stage == "closing":
                for side in order:

                    def _gen_closing(side=side):
                        return self._env.debaters[side].closing_generation(
                            history=self._env.debate_process[1:],
                            max_time=CLOSING_TIME,
                            time_control=self._env.time_control,
                            streaming_tts=getattr(self._env.debaters[side].config, "streaming_tts", False),
                        )

                    self._play_speech_turn("closing", side, CLOSING_TIME, _gen_closing)
            logger.info(f"[{stage}] Done")
            if self._env.debug:
                if input("Press N to stop: ").lower() == "n":
                    break


__all__ = [
    "transcribe_audio_segment",
    "opponent_side",
    "StreamingInputConfig",
    "StreamingInputEnv",
    "load_treedebater_for_side",
    "tts_outputs_dir_from_log",
    "default_watch_root_from_log",
    "_tts_outputs_dir_from_log",
    "_default_watch_root_from_log",
    "StreamingDebateEnv",
    "parse_args",
    "main",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch-only: stream ASR on a folder. --debate: full Env debate + streaming listener per turn."
    )
    p.add_argument("--config", type=str, required=True, help="YAML config (same layout as env.py).")
    p.add_argument(
        "--debate",
        action="store_true",
        help="Run full debate (like env.py) with StreamingInputEnv on opponent TreeDebater each speech turn.",
    )
    p.add_argument(
        "--watch-dir",
        type=str,
        default=None,
        help="Watch-only: required chunk directory. Debate mode: optional root for per-turn subdirs (default: log_files/<stem>_watch).",
    )
    p.add_argument(
        "--debater-side",
        type=str,
        choices=["for", "against"],
        default=None,
        help="Watch-only: load treedebater for this listener side. Not used with --debate.",
    )
    p.add_argument("--debug", action="store_true", default=False, help="Debate mode: pause prompts like env.py.")
    p.add_argument(
        "--stage",
        type=str,
        choices=["opening", "rebuttal", "closing"],
        default="opening",
        help="Watch-only: stage label for extract_statement.",
    )
    p.add_argument("--min-audio-seconds", type=float, default=30.0)
    p.add_argument("--min-text-words", type=int, default=50)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--audio-glob", type=str, default="*.mp3")
    p.add_argument("--audio-format", type=str, default="mp3")
    p.add_argument("--split-mode", type=str, choices=["fixed", "silence"], default="fixed")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--silence-window-seconds", type=float, default=0.7)
    p.add_argument("--max-audio-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-text-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-total-audio-seconds", type=float, default=0.0)
    p.add_argument("--listener-join-timeout", type=float, default=300.0, help="Debate mode: max seconds to join listener after each turn.")
    p.add_argument("--min-playback-increment", type=float, default=3.0, help="Minimum playback increment in seconds (env-level cursor update frequency).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if args.debate:
        from agents import AudienceConfig, DebaterConfig, JudgeConfig
        from env import EnvConfig

        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"Config: {full_config}")
        env_config = EnvConfig(
            debater_config=[DebaterConfig(**c) for c in full_config["debater"]],
            judge_config=JudgeConfig(**full_config["judge"]),
            audience_config=AudienceConfig(**full_config["audience"]),
            **full_config["env"],
        )
        watch_root = Path(args.watch_dir).resolve() if args.watch_dir else None
        sde = StreamingDebateEnv(
            env_config,
            args.debug,
            watch_root,
            min_audio_seconds=args.min_audio_seconds,
            min_text_words=args.min_text_words,
            poll_interval=args.poll_interval,
            audio_format=args.audio_format,
            split_mode=args.split_mode,
            chunk_seconds=args.chunk_seconds,
            silence_window_seconds=args.silence_window_seconds,
            max_audio_wait_seconds=args.max_audio_wait_seconds,
            max_text_wait_seconds=args.max_text_wait_seconds,
            max_total_audio_seconds=args.max_total_audio_seconds,
            listener_join_timeout=args.listener_join_timeout,
            min_playback_increment=args.min_playback_increment,
        )
        sde.play()
        log_file = logger.handlers[0].baseFilename
        save_file = log_file.replace(".log", ".json")
        logger.info(f"Saving to {save_file}")
        record = {
            "motion": sde.motion,
            "config": full_config,
            "debate_process": sde.debate_process[1:],
            "debate_thoughts": {
                "for": sde.debaters["for"].debate_thoughts,
                "against": sde.debaters["against"].debate_thoughts,
            },
            "debate_tree": {
                "for": [
                    (
                        sde.debaters["for"].debate_tree.get_tree_info()
                        if sde.debaters["for"].type in ["treedebater"]
                        else {}
                    ),
                    (
                        sde.debaters["for"].oppo_debate_tree.get_tree_info()
                        if sde.debaters["for"].type in ["treedebater"]
                        else {}
                    ),
                ],
                "against": [
                    (
                        sde.debaters["against"].debate_tree.get_tree_info()
                        if sde.debaters["against"].type in ["treedebater"]
                        else {}
                    ),
                    (
                        sde.debaters["against"].oppo_debate_tree.get_tree_info()
                        if sde.debaters["against"].type in ["treedebater"]
                        else {}
                    ),
                ],
            },
            "conversation": {
                "for": sde.debaters["for"].conversation,
                "against": sde.debaters["against"].conversation,
            },
        }
        json.dump(record, open(save_file, "w"), indent=2)
        if not args.debug:
            evaluation, side_into = sde.eval()
            logger.info(f"Result: {evaluation}")
            record.update({"evaluation": evaluation, "eval_side_info": side_into})
            json.dump(record, open(save_file, "w"), indent=2)
        return

    if not args.watch_dir or not args.debater_side:
        raise SystemExit("Watch-only mode requires --watch-dir and --debater-side (or use --debate).")

    debater_side = args.debater_side
    _, debater = load_treedebater_for_side(config_path, debater_side)
    if debater.side != debater_side:
        raise RuntimeError("Debater side mismatch after load.")

    statement_side = opponent_side(debater_side)
    logger.info(
        f"[StreamingInputEnv] TreeDebater (listener) side={debater.side!r}; "
        f"chunk audio is speaker side={statement_side!r}."
    )

    sic = StreamingInputConfig(
        watch_dir=Path(args.watch_dir).resolve(),
        motion=debater.motion,
        stage=args.stage,
        statement_side=statement_side,
        min_audio_seconds=args.min_audio_seconds,
        min_text_words=args.min_text_words,
        poll_interval=args.poll_interval,
        audio_file_glob=args.audio_glob,
        max_audio_wait_seconds=args.max_audio_wait_seconds if args.max_audio_wait_seconds > 0 else None,
        max_text_wait_seconds=args.max_text_wait_seconds if args.max_text_wait_seconds > 0 else None,
        max_total_audio_seconds=args.max_total_audio_seconds if args.max_total_audio_seconds > 0 else None,
        audio_format=args.audio_format,
    )

    env = StreamingInputEnv(debater, sic)
    env.run()


if __name__ == "__main__":
    # Run as: ``python -m streaming.env`` from ``TreeDebater/src`` (or with ``src`` on ``PYTHONPATH``).
    src_dir = Path(__file__).resolve().parent.parent
    root = src_dir.parent
    for p in (src_dir, root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    main()
