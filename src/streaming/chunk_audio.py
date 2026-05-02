"""
Write timed audio chunks into a watch directory (same contract as
debate-anonymous ``simulate_online_audio_stream.py``): filenames
``<log_id>_chunkNNN.<format>`` so :mod:`streaming.env` groups by ``log_id``.

Use an **opponent** recording with :mod:`streaming.env` / :mod:`streaming.run_listen_demo`
(``--debater-side`` is your agent; audio is attributed to the other side).

Depends on ``pydub`` only for chunk I/O. :func:`clear_watch_chunk_files` is a small helper
for demos (no extra deps). Run from ``TreeDebater/src``::

    python -m streaming.chunk_audio --audio-file path/to/opponent_speech.mp3 --watch-dir /tmp/watch_demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

from pydub import AudioSegment
from pydub.silence import split_on_silence


def clear_watch_chunk_files(watch_dir: Path, audio_format: str = "mp3") -> None:
    """Remove ``*.{audio_format}`` files under ``watch_dir`` (ignore subdirs / other suffixes)."""
    watch_dir = Path(watch_dir)
    if not watch_dir.exists():
        return
    suf = f".{audio_format.lower().lstrip('.')}"
    for entry in watch_dir.iterdir():
        try:
            if entry.is_file() and entry.suffix.lower() == suf:
                entry.unlink()
        except OSError as e:
            print(f"Warning: could not remove {entry}: {e}", file=sys.stderr)


def split_audio(audio: AudioSegment, mode: str, time_seconds: float) -> List[AudioSegment]:
    """Split ``audio`` into chunks (``time_seconds`` is chunk length for ``fixed``, min silence for ``silence``)."""
    if mode == "fixed":
        chunk_ms = int(time_seconds * 1000)
        chunks: List[AudioSegment] = []
        for i in range(0, len(audio), chunk_ms):
            chunk = audio[i : i + chunk_ms]
            if len(chunk) > 500:
                chunks.append(chunk)
        return chunks
    if mode == "silence":
        chunks = split_on_silence(
            audio,
            min_silence_len=int(time_seconds * 1000),
            silence_thresh=-60,
            keep_silence=200,
            seek_step=10,
        )
        return [c for c in chunks if len(c) > 500]
    raise ValueError(f"Invalid mode: {mode}")


def stream_chunks_to_directory(
    chunks: List[AudioSegment],
    watch_dir: Path,
    log_id: str,
    audio_format: str = "mp3",
    dry_run: bool = False,
    max_total_seconds: float | None = None,
    chunk_index_start: int = 1,
    realtime_pace: bool = True,
) -> int:
    """
    Write each chunk to ``watch_dir``, sleeping ~chunk duration between writes (approx. real-time).

    ``chunk_index_start`` continues ``{log_id}_chunkNNN`` across multiple source files.
    Returns the next chunk index after this batch.

    If ``realtime_pace`` is False, writes all chunks back-to-back (no sleep); use when a
    separate thread simulates playback timing (e.g. playback-driven debate env).
    """
    watch_dir = Path(watch_dir)
    watch_dir.mkdir(parents=True, exist_ok=True)

    total = len(chunks)
    accumulated_seconds = 0.0
    last_written_idx = chunk_index_start - 1
    for i, chunk in enumerate(chunks):
        idx = chunk_index_start + i
        filename = f"{log_id}_chunk{idx:03d}.{audio_format}"
        out_path = watch_dir / filename

        duration_seconds = len(chunk) / 1000.0
        accumulated_seconds += duration_seconds
        if dry_run:
            print(
                f"[dry-run] Would write {out_path} "
                f"(chunk_duration_s={duration_seconds:.3f}, accumulated_s={accumulated_seconds:.3f})"
            )
        else:
            chunk.export(str(out_path), format=audio_format)
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                f"[{ts}] Wrote {out_path} "
                f"(chunk_duration_s={duration_seconds:.3f}, accumulated_s={accumulated_seconds:.3f})"
            )
        last_written_idx = idx

        if max_total_seconds is not None and accumulated_seconds >= max_total_seconds:
            print(
                f"Reached max_total_seconds={max_total_seconds:.3f}s for log_id='{log_id}'. "
                "Stopping streaming; remaining chunks will be dropped."
            )
            return last_written_idx + 1

        if realtime_pace and i < total - 1:
            sleep_seconds = len(chunk) / 1000.0
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return last_written_idx + 1


__all__ = [
    "clear_watch_chunk_files",
    "split_audio",
    "stream_chunks_to_directory",
    "parse_args",
    "main",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split an audio file and stream chunk files into a directory in (approximate) real time."
    )
    p.add_argument("--audio-file", type=str, required=True)
    p.add_argument("--watch-dir", type=str, required=True)
    p.add_argument(
        "--log-id",
        type=str,
        default=None,
        help="Filename prefix before _chunkNNN. Default: stem of audio file before first underscore.",
    )
    p.add_argument("--split-mode", type=str, choices=["fixed", "silence"], default="fixed")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--silence-window-seconds", type=float, default=0.7)
    p.add_argument("--audio-format", type=str, default="mp3")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-total-seconds", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    if args.log_id is not None:
        log_id: str = args.log_id
    else:
        stem = audio_path.stem
        log_id = stem.split("_")[0] if "_" in stem else stem

    print(f"Using log_id='{log_id}'")
    audio = AudioSegment.from_file(str(audio_path))
    if args.split_mode == "fixed":
        print(f"Splitting into fixed {args.chunk_seconds:.2f}s chunks...")
        chunks = split_audio(audio, mode="fixed", time_seconds=args.chunk_seconds)
    else:
        print(f"Splitting on silence (min {args.silence_window_seconds:.2f}s)...")
        chunks = split_audio(audio, mode="silence", time_seconds=args.silence_window_seconds)

    print(f"Created {len(chunks)} chunks.")
    watch_dir = Path(args.watch_dir)
    print(f"Streaming to '{watch_dir}'...")
    stream_chunks_to_directory(
        chunks=chunks,
        watch_dir=watch_dir,
        log_id=log_id,
        audio_format=args.audio_format,
        dry_run=bool(args.dry_run),
        max_total_seconds=args.max_total_seconds,
    )
    print("Done streaming chunks.")


if __name__ == "__main__":
    _src = Path(__file__).resolve().parent.parent
    _root = _src.parent
    for _p in (_src, _root):
        _ps = str(_p)
        if _ps not in sys.path:
            sys.path.insert(0, _ps)
    main()
