"""
Live streaming demo (three roles):

1. **Speaker** — ``env.py`` (or any TTS) writes ``log_files/N_outputs/{type}_{stage}_{side}.mp3``
   for the side that is speaking.
2. **Chunk bridge** — polls that folder for **new** speaker-side MP3s, waits until each file is
   stable, splits into timed chunks, and writes ``{speaker}_chunkNNN.mp3`` into ``--watch-dir``.
3. **Listener** — ``StreamingInputEnv`` (TreeDebater for ``--debater-side``) watches
   ``--watch-dir``, runs streaming ASR, and calls ``_analyze_statement`` with the speaker as
   ``statement_side``.

Start this script **before** or **while** running ``env.py`` so TTS drops new files into
``N_outputs``; the bridge feeds the listener in (approximate) real time.

**Audio source**

- Default: latest ``log_files/N_outputs`` (see ``streaming.bridges.infer_session_log_id``),
  or ``--log-id`` / ``--outputs-dir``.
- ``--audio-file``: one-shot test (no bridge): stream a single file then stop (no live poll).

Example (listener = ``for``, speaker = ``against``, run while ``env.py`` produces TTS)::

    python -m streaming.run_listen_demo \\
      --config configs/base_st.yml \\
      --watch-dir /tmp/td_stream_demo \\
      --debater-side for

Requires ``OPENAI_API_KEY`` (Whisper) and keys for ``TreeDebater`` / ``HelperClient``.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

_src_dir = Path(__file__).resolve().parent.parent
for _p in (_src_dir, _src_dir.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from pydub import AudioSegment

from .bridges import default_outputs_dir, infer_session_log_id, run_live_chunk_bridge
from .chunk_audio import clear_watch_chunk_files, split_audio, stream_chunks_to_directory
from .env import StreamingInputConfig, StreamingInputEnv, load_treedebater_for_side, opponent_side


def _resolve_chunk_log_id(args: argparse.Namespace, speaker_side: str, audio_path: Path | None) -> str:
    if args.log_id is not None:
        return args.log_id
    if audio_path is not None:
        stem = audio_path.stem
        return stem.split("_")[0] if "_" in stem else stem
    return speaker_side


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live bridge: speaker TTS → log_files/N_outputs → chunks → watch-dir; "
        "listener TreeDebater runs streaming ASR on watch-dir."
    )
    p.add_argument("--config", type=str, required=True, help="YAML config (same as env.py).")
    p.add_argument(
        "--log-id",
        type=str,
        default=None,
        help="Session folder log_files/<id>_outputs. If omitted, uses latest existing N_outputs.",
    )
    p.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Override TTS output directory (same naming as agents.post_process).",
    )
    p.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="If set, run a one-shot chunk stream from this file (no live bridge on N_outputs).",
    )
    p.add_argument("--watch-dir", type=str, required=True)
    p.add_argument(
        "--debater-side",
        type=str,
        choices=["for", "against"],
        required=True,
        help="TreeDebater / listener side in YAML (the agent that watches watch-dir and runs ASR).",
    )
    p.add_argument(
        "--speaker-side",
        type=str,
        choices=["for", "against"],
        default=None,
        help="Who is speaking into log_files/N_outputs (default: opponent of --debater-side).",
    )
    p.add_argument("--stage", type=str, choices=["opening", "rebuttal", "closing"], default="opening")
    p.add_argument("--split-mode", type=str, choices=["fixed", "silence"], default="fixed")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--silence-window-seconds", type=float, default=0.7)
    p.add_argument("--audio-format", type=str, default="mp3")
    p.add_argument("--min-audio-seconds", type=float, default=30.0)
    p.add_argument("--min-text-words", type=int, default=50)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--bridge-stable-polls", type=int, default=2, help="Consecutive polls with same file size before loading.")
    p.add_argument(
        "--process-existing-speaker-mp3s",
        action="store_true",
        help="Also chunk speaker MP3s already in N_outputs when the bridge starts (default: only new files).",
    )
    p.add_argument("--max-audio-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-text-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-total-audio-seconds", type=float, default=0.0)
    p.add_argument("--max-total-seconds", type=float, default=None, help="Only for --audio-file one-shot cap.")
    p.add_argument(
        "--max-wall-seconds",
        type=float,
        default=0.0,
        help="Live mode: stop bridge and listener after this many wall-clock seconds (0 = until Ctrl+C).",
    )
    p.add_argument(
        "--pipeline-timeout-seconds",
        type=float,
        default=90.0,
        help="After stop, wait up to this long for the listener thread to finish draining.",
    )
    p.add_argument("--no-clear-watch", action="store_true", help="Do not delete existing chunk files in watch-dir before run.")
    return p.parse_args()


def _run_one_shot_audio_file(
    args: argparse.Namespace,
    watch_dir: Path,
    debater,
    statement_side: str,
    chunk_log_id: str,
    env: StreamingInputEnv,
    worker: threading.Thread,
) -> None:
    audio_path = Path(args.audio_file).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    print(f"One-shot mode: loading {audio_path}...")
    audio = AudioSegment.from_file(str(audio_path))
    if args.split_mode == "fixed":
        chunks = split_audio(audio, mode="fixed", time_seconds=args.chunk_seconds)
    else:
        chunks = split_audio(audio, mode="silence", time_seconds=args.silence_window_seconds)
    print(f"Streaming {len(chunks)} chunks to {watch_dir}...")
    stream_chunks_to_directory(
        chunks,
        watch_dir,
        chunk_log_id,
        audio_format=args.audio_format,
        dry_run=False,
        max_total_seconds=args.max_total_seconds,
        chunk_index_start=1,
    )
    print("One-shot stream finished; draining listener...")
    timeout = args.pipeline_timeout_seconds
    if timeout and timeout > 0:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if not worker.is_alive():
                break
            time.sleep(1.0)
    env.stop()
    worker.join(timeout=min(timeout, 30.0) if timeout else 30.0)


def main() -> None:
    args = parse_args()

    watch_dir = Path(args.watch_dir).resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_clear_watch:
        clear_watch_chunk_files(watch_dir, args.audio_format)

    config_path = Path(args.config).resolve()
    debater_side = args.debater_side
    _, debater = load_treedebater_for_side(config_path, debater_side)
    speaker_side = args.speaker_side or opponent_side(debater_side)
    statement_side = speaker_side
    chunk_log_id = speaker_side

    print(
        f"Listener TreeDebater side={debater.side!r}; speaker (TTS → N_outputs) side={speaker_side!r}; "
        f"chunks use log_id={chunk_log_id!r}."
    )

    audio_glob = f"*.{args.audio_format}"
    sic = StreamingInputConfig(
        watch_dir=watch_dir,
        motion=debater.motion,
        stage=args.stage,
        statement_side=statement_side,
        min_audio_seconds=args.min_audio_seconds,
        min_text_words=args.min_text_words,
        poll_interval=args.poll_interval,
        audio_file_glob=audio_glob,
        max_audio_wait_seconds=args.max_audio_wait_seconds if args.max_audio_wait_seconds > 0 else None,
        max_text_wait_seconds=args.max_text_wait_seconds if args.max_text_wait_seconds > 0 else None,
        max_total_audio_seconds=args.max_total_audio_seconds if args.max_total_audio_seconds > 0 else None,
        audio_format=args.audio_format,
    )

    env = StreamingInputEnv(debater, sic)
    listener = threading.Thread(target=env.run, name="StreamingInputEnv", daemon=False)
    print(f"Starting listener on {watch_dir} (glob {audio_glob}).")
    listener.start()
    time.sleep(max(0.5, args.poll_interval))

    stop_bridge = threading.Event()

    try:
        if args.audio_file:
            chunk_log_id = _resolve_chunk_log_id(args, speaker_side, Path(args.audio_file).resolve())
            _run_one_shot_audio_file(args, watch_dir, debater, statement_side, chunk_log_id, env, listener)
            return

        if args.outputs_dir:
            outputs_dir = Path(args.outputs_dir).resolve()
        else:
            session_log_id = args.log_id or infer_session_log_id()
            if not session_log_id:
                raise SystemExit(
                    "No log_files/N_outputs found. Create one (e.g. run env.py once) or pass --log-id / --outputs-dir."
                )
            if args.log_id is None:
                print(f"Inferred session log-id={session_log_id!r} (latest log_files/N_outputs).")
            outputs_dir = default_outputs_dir(session_log_id)

        if not outputs_dir.is_dir():
            raise NotADirectoryError(f"Speaker TTS directory missing: {outputs_dir}")

        bridge_thread = threading.Thread(
            target=run_live_chunk_bridge,
            kwargs={
                "outputs_dir": outputs_dir,
                "watch_dir": watch_dir,
                "speaker_side": speaker_side,
                "chunk_log_id": chunk_log_id,
                "stop_event": stop_bridge,
                "audio_format": args.audio_format,
                "split_mode": args.split_mode,
                "chunk_seconds": args.chunk_seconds,
                "silence_window_seconds": args.silence_window_seconds,
                "poll_interval": args.poll_interval,
                "stable_polls": args.bridge_stable_polls,
                "skip_initial_files": not args.process_existing_speaker_mp3s,
            },
            name="LiveChunkBridge",
            daemon=True,
        )
        bridge_thread.start()

        wall = args.max_wall_seconds
        t0 = time.time()
        print("Live mode running (Ctrl+C to stop).")
        while True:
            if not listener.is_alive():
                print("Listener thread exited unexpectedly.")
                break
            if wall and wall > 0 and (time.time() - t0) >= wall:
                print(f"--max-wall-seconds={wall} reached; stopping.")
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        stop_bridge.set()
        time.sleep(0.5)
        env.stop()
        drain = args.pipeline_timeout_seconds if args.pipeline_timeout_seconds and args.pipeline_timeout_seconds > 0 else 90.0
        listener.join(timeout=drain)
        if listener.is_alive():
            print("Listener still running after drain timeout.")

    if not args.no_clear_watch:
        clear_watch_chunk_files(watch_dir, args.audio_format)


if __name__ == "__main__":
    main()
