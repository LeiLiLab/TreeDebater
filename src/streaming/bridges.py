"""
Bridges speaker audio into a **watch directory** for :class:`streaming.env.StreamingInputEnv`.

1. **Full TTS MP3** (``run_live_chunk_bridge``): poll ``log_files/N_outputs`` for stable
   ``{type}_{stage}_{speaker}.mp3`` from ``agents.post_process``, split with pydub, write
   ``{chunk_log_id}_chunkNNN.*`` — for running **alongside** ``env.py`` / demos.

2. **Streaming TTS partial files** (``run_streaming_tts_chunk_copy_bridge``): poll a
   per-turn ``*_chunks`` directory for stable ``chunk_NNN.*`` from streaming TTS, **copy**
   (no re-encode) to ``{speaker_side}_chunkNNN.*`` for overlap debate + playback-driven env.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydub import AudioSegment

from .chunk_audio import split_audio, stream_chunks_to_directory
from utils.tool import logger

_REPO_ROOT = Path(__file__).resolve().parents[2]

_STAGE_ORDER = {"opening": 0, "rebuttal": 1, "closing": 2}

__all__ = [
    "default_outputs_dir",
    "infer_session_log_id",
    "list_speaker_mp3s",
    "run_live_chunk_bridge",
    "run_streaming_tts_chunk_copy_bridge",
]


def default_outputs_dir(log_id: str) -> Path:
    return (_REPO_ROOT / "log_files" / f"{log_id}_outputs").resolve()


def infer_session_log_id(repo_root: Path | None = None) -> str | None:
    """Largest numeric ``N`` with ``log_files/N_outputs`` present."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    log_dir = root / "log_files"
    if not log_dir.is_dir():
        return None
    best: int | None = None
    for p in log_dir.iterdir():
        if not p.is_dir() or not p.name.endswith("_outputs"):
            continue
        stem = p.name[: -len("_outputs")]
        if stem.isdigit():
            n = int(stem)
            if best is None or n > best:
                best = n
    return str(best) if best is not None else None


def list_speaker_mp3s(outputs_dir: Path, speaker_side: str, ext: str) -> list[Path]:
    """``agents.py`` TTS names: ``{type}_{stage}_{side}.mp3`` — keep files for ``speaker_side``."""
    ext = ext.lower().lstrip(".")
    out: list[Path] = []
    for p in outputs_dir.glob(f"*.{ext}"):
        if not p.is_file():
            continue
        parts = p.stem.split("_")
        if len(parts) >= 2 and parts[-1] == speaker_side:
            out.append(p)

    def sort_key(p: Path) -> tuple:
        parts = p.stem.split("_")
        if len(parts) >= 2:
            rank = _STAGE_ORDER.get(parts[-2], 99)
        else:
            rank = 99
        return (rank, p.name.lower())

    return sorted(out, key=sort_key)


def run_live_chunk_bridge(
    outputs_dir: Path,
    watch_dir: Path,
    speaker_side: str,
    chunk_log_id: str,
    stop_event: threading.Event,
    *,
    audio_format: str = "mp3",
    split_mode: str = "fixed",
    chunk_seconds: float = 10.0,
    silence_window_seconds: float = 0.7,
    poll_interval: float = 1.0,
    stable_polls: int = 2,
    skip_initial_files: bool = True,
) -> None:
    """
    Until ``stop_event`` is set, poll ``outputs_dir`` for new ``*_*_{speaker_side}.mp3``,
    split each stable file into chunks, and append to ``watch_dir`` using monotonic
    ``{chunk_log_id}_chunkNNN`` indices.
    """
    outputs_dir = Path(outputs_dir).resolve()
    watch_dir = Path(watch_dir).resolve()
    ext = audio_format.lower().lstrip(".")

    streamed: Set[str] = set()
    if skip_initial_files:
        for p in list_speaker_mp3s(outputs_dir, speaker_side, ext):
            streamed.add(str(p.resolve()))

    size_stable: Dict[str, tuple[int, int]] = {}
    next_chunk_idx = 1

    print(
        f"[live_chunk_bridge] Watching {outputs_dir} for new {speaker_side!r} speech "
        f"(skip_initial={skip_initial_files}), writing chunks to {watch_dir} as log_id={chunk_log_id!r}."
    )

    while not stop_event.is_set():
        for path in list_speaker_mp3s(outputs_dir, speaker_side, ext):
            key = str(path.resolve())
            if key in streamed:
                continue
            try:
                sz = path.stat().st_size
            except OSError:
                continue
            if sz < 2048:
                continue
            prev = size_stable.get(key)
            if prev is None or prev[0] != sz:
                size_stable[key] = (sz, 1)
                continue
            last_sz, cnt = prev
            cnt += 1
            size_stable[key] = (last_sz, cnt)
            if cnt < stable_polls:
                continue

            try:
                audio = AudioSegment.from_file(str(path))
            except Exception as e:
                print(f"[live_chunk_bridge] Could not load {path.name} yet ({e}); retrying later.")
                size_stable.pop(key, None)
                continue

            if split_mode == "fixed":
                chunks = split_audio(audio, mode="fixed", time_seconds=chunk_seconds)
            else:
                chunks = split_audio(audio, mode="silence", time_seconds=silence_window_seconds)

            print(
                f"[live_chunk_bridge] {path.name} ({len(audio) / 1000.0:.1f}s) → {len(chunks)} chunk(s), "
                f"starting at chunk index {next_chunk_idx}."
            )
            next_chunk_idx = stream_chunks_to_directory(
                chunks,
                watch_dir,
                chunk_log_id,
                audio_format=audio_format,
                dry_run=False,
                max_total_seconds=None,
                chunk_index_start=next_chunk_idx,
            )
            streamed.add(key)
            size_stable.pop(key, None)

        time.sleep(poll_interval)

    print("[live_chunk_bridge] Stopped.")


def run_streaming_tts_chunk_copy_bridge(
    chunks_dir: Path,
    watch_dir: Path,
    speaker_side: str,
    stop_event: threading.Event,
    live_chunk_counter: List[int],
    *,
    audio_format: str = "mp3",
    poll_interval: float = 0.4,
    stable_rounds: int = 2,
    min_bytes: int = 512,
) -> None:
    """
    Poll ``chunks_dir`` for ``chunk_NNN.<ext>`` from streaming TTS; when byte-size is
    stable, copy to ``watch_dir`` as ``{speaker_side}_chunkNNN.<ext>`` (same contract as
    :func:`streaming.chunk_audio.stream_chunks_to_directory` filenames for that log id).

    Increments ``live_chunk_counter[0]`` for each successful copy (used to skip post-hoc
    split when overlap debate already fed chunks).
    """
    chunks_dir = Path(chunks_dir).resolve()
    watch_dir = Path(watch_dir).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    ext = audio_format.lower().lstrip(".")
    size_stable: dict[str, tuple[int, int]] = {}
    copied_indices: set[int] = set()
    chunk_detected_time: dict[int, float] = {}

    def _chunk_index(path: Path) -> Optional[int]:
        if path.suffix.lower() != f".{ext}":
            return None
        parts = path.stem.split("_")
        if len(parts) != 2 or parts[0] != "chunk":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    while not stop_event.is_set():
        if not chunks_dir.is_dir():
            time.sleep(poll_interval)
            continue

        for path in sorted(chunks_dir.glob(f"chunk_*.{ext}"), key=lambda p: p.stat().st_mtime):
            idx = _chunk_index(path)
            if idx is None or idx in copied_indices:
                continue
            key = str(path.resolve())
            try:
                sz = path.stat().st_size
            except OSError:
                continue
            if sz < min_bytes:
                continue

            if idx not in chunk_detected_time:
                chunk_detected_time[idx] = time.time()
                logger.debug(
                    f"[TtsChunkBridge] chunk_detected chunk_idx={idx} size={sz} t={time.time():.3f}"
                )

            prev = size_stable.get(key)
            if prev is None or prev[0] != sz:
                size_stable[key] = (sz, 1)
                continue
            _, cnt = size_stable[key]
            cnt += 1
            size_stable[key] = (sz, cnt)
            if cnt < stable_rounds:
                continue

            # PlaybackMain consumes chunk001+, while streaming TTS emits chunk_000+.
            # Shift to 1-based indices when copying into watch_dir.
            playback_idx = idx + 1
            dest = watch_dir / f"{speaker_side}_chunk{playback_idx:03d}.{ext}"
            try:
                copy_start = time.time()
                shutil.copy2(path, dest)
                copy_end = time.time()
                copied_indices.add(idx)
                live_chunk_counter[0] += 1
                detection_latency = copy_end - chunk_detected_time[idx]
                logger.debug(
                    f"[TtsChunkBridge] chunk_copied chunk_idx={idx} playback_chunk_idx={playback_idx} "
                    f"detection_latency={detection_latency:.3f}s "
                    f"copy_time={copy_end - copy_start:.3f}s t={time.time():.3f}"
                )
                logger.info(f"[TtsChunkBridge] {path.name} → {dest.name} (n={live_chunk_counter[0]})")
            except OSError as e:
                logger.warning(f"[TtsChunkBridge] copy failed {path} → {dest}: {e}")

        time.sleep(poll_interval)
