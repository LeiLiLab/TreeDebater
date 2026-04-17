"""
Full debate with **true overlap** and a **playback-driven main thread**.

The **main thread** only waits for stable ``{side}_chunkNNN.mp3`` files in the turn watch
directory and simulates speaking (sleeps for each chunk's duration). The **speaker** runs
``generate_fn()`` (LLM + refinements + ``streaming_tts``) on a **background thread**, with
the TTS chunk **bridge** started/stopped in that thread. The **listener** runs
:class:`StreamingInputEnv` via :meth:`TreeDebater.start_streaming_listen` on another
background thread, so ASR and tree updates can proceed while chunks appear.

``streaming_listen: true`` on the listener enables ``StreamingInputEnv``; if false, only
the speaker worker + main playback run (no duplicate tree ingest).

Run from ``TreeDebater/src``::

    python -m streaming.overlap --config configs/overlap_debate.yml
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import yaml
from pydub import AudioSegment

from .bridges import run_streaming_tts_chunk_copy_bridge
from .env import StreamingDebateEnv, opponent_side, tts_outputs_dir_from_log
from utils.tool import logger


class OverlappingStreamingDebateEnv(StreamingDebateEnv):
    """
    Extends :class:`StreamingDebateEnv` with playback-driven main: simulate debate delivery
    on the main thread while speaker (LLM + TTS + bridge) and optional listener ASR run in
    background threads.
    """

    @staticmethod
    def _playback_chunk_ready(path: Path, min_size: int, poll_interval: float) -> bool:
        if not path.is_file():
            return False
        try:
            s1 = path.stat().st_size
        except OSError:
            return False
        if s1 < min_size:
            return False
        time.sleep(poll_interval)
        try:
            s2 = path.stat().st_size
        except OSError:
            return False
        return s1 == s2

    def _playback_main_loop(
        self,
        turn_watch: Path,
        side: str,
        stage: str,
        speaker_done: threading.Event,
        error_holder: List[Optional[BaseException]],
        done_ts: List[Optional[float]],
        playback_cursor: List[float],
        *,
        grace_seconds: float = 4.0,
    ) -> None:
        """Main thread: assemble continuous audio and sleep in fixed increments, updating cursor."""
        ext = self._audio_format.lower().lstrip(".")
        poll = min(0.2, self._poll_interval)
        next_idx = 1
        t0 = time.monotonic()
        deadline = t0 + max(600.0, self._listener_join_timeout * 6)
        min_size = 512

        continuous_audio = AudioSegment.empty()
        continuous_file = turn_watch / "continuous_audio.mp3"

        logger.debug(
            f"[PlaybackMain] playback_start stage={stage} side={side} t={time.time():.3f}"
        )

        while time.monotonic() < deadline:
            if error_holder[0] is not None:
                logger.info("[PlaybackMain] stopping playback due to speaker thread error.")
                return

            path = turn_watch / f"{side}_chunk{next_idx:03d}.{ext}"
            should_stop = False

            # Start waiting for chunk
            wait_start = time.time()
            logger.debug(
                f"[PlaybackMain] wait_chunk_start stage={stage} side={side} chunk_idx={next_idx} "
                f"cursor={playback_cursor[0]:.2f}s t={wait_start:.3f}"
            )

            while time.monotonic() < deadline:
                if error_holder[0] is not None:
                    return
                if self._playback_chunk_ready(path, min_size, poll):
                    break
                if speaker_done.is_set() and done_ts[0] is not None:
                    try:
                        chunk_ok = path.is_file() and path.stat().st_size >= min_size
                    except OSError:
                        chunk_ok = False
                    if not chunk_ok and time.monotonic() - done_ts[0] >= grace_seconds:
                        should_stop = True
                        break
                time.sleep(poll)
            else:
                logger.warning("[PlaybackMain] deadline waiting for next chunk.")
                return

            # End waiting for chunk
            wait_end = time.time()
            wait_duration = wait_end - wait_start
            if not should_stop:
                logger.debug(
                    f"[PlaybackMain] wait_chunk_end stage={stage} side={side} chunk_idx={next_idx} "
                    f"wait_time={wait_duration:.3f}s t={wait_end:.3f}"
                )

            if should_stop:
                logger.info(f"[PlaybackMain] no further chunks after index {next_idx - 1}; done.")
                logger.debug(
                    f"[PlaybackMain] playback_end stage={stage} side={side} "
                    f"cursor={playback_cursor[0]:.2f}s t={time.time():.3f}"
                )
                return

            try:
                seg = AudioSegment.from_file(str(path))
                chunk_duration = len(seg) / 1000.0
            except Exception as e:
                logger.warning(f"[PlaybackMain] skip unreadable {path}: {e}")
                next_idx += 1
                continue

            # Assemble into continuous audio
            assemble_start = time.time()
            continuous_audio += seg
            logger.debug(
                f"[PlaybackMain] chunk_assembled stage={stage} side={side} chunk_idx={next_idx} "
                f"duration={chunk_duration:.2f}s total_audio={len(continuous_audio)/1000.0:.2f}s t={time.time():.3f}"
            )

            try:
                write_start = time.time()
                continuous_audio.export(str(continuous_file), format=ext)
                write_end = time.time()
                logger.debug(
                    f"[PlaybackMain] file_write stage={stage} side={side} chunk_idx={next_idx} "
                    f"size_sec={len(continuous_audio)/1000.0:.2f}s write_time={write_end - write_start:.3f}s t={time.time():.3f}"
                )
            except Exception as e:
                logger.warning(f"[PlaybackMain] failed to export continuous audio: {e}")

            # Start playback of this chunk
            playback_start = time.time()
            logger.debug(
                f"[PlaybackMain] chunk_playback_start stage={stage} side={side} chunk_idx={next_idx} "
                f"duration={chunk_duration:.2f}s t={playback_start:.3f}"
            )

            # Simulate speaking in fixed increments
            elapsed_in_chunk = 0.0
            while elapsed_in_chunk < chunk_duration:
                sleep_time = min(self._min_playback_increment, chunk_duration - elapsed_in_chunk)
                time.sleep(sleep_time)

                # Advance cursor - listener can now access this audio
                playback_cursor[0] += sleep_time
                elapsed_in_chunk += sleep_time
                logger.debug(f"[PlaybackMain] cursor={playback_cursor[0]:.1f}s (chunk {next_idx}, elapsed={elapsed_in_chunk:.1f}s/{chunk_duration:.1f}s)")

            # End playback of this chunk
            logger.debug(
                f"[PlaybackMain] chunk_playback_end stage={stage} side={side} chunk_idx={next_idx} "
                f"cursor={playback_cursor[0]:.2f}s t={time.time():.3f}"
            )

            next_idx += 1

        logger.debug(
            f"[PlaybackMain] playback_end stage={stage} side={side} cursor={playback_cursor[0]:.2f}s t={time.time():.3f}"
        )

    def _play_speech_turn(self, stage_key: str, side: str, max_time: float, generate_fn: Callable[[], str]) -> None:
        from .chunk_audio import split_audio, stream_chunks_to_directory

        listener = opponent_side(side)
        player = self._env.debaters[side]
        listener_deb = self._env.debaters[listener]

        logger.debug(f"[Turn] turn_start stage={stage_key} side={side} t={time.time():.3f}")

        if listener_deb.type != "treedebater":
            response = generate_fn()
            self._env.debate_process.append({"stage": stage_key, "side": side, "content": response})
            logger.debug(f"[Turn] turn_end stage={stage_key} side={side} t={time.time():.3f}")
            return

        use_streaming_listen = getattr(listener_deb.config, "streaming_listen", False)
        use_streaming_tts = self._env.time_control and getattr(player.config, "streaming_tts", False)

        # Log mode configuration
        mode = f"tts={'stream' if use_streaming_tts else 'batch'}_listen={'stream' if use_streaming_listen else 'batch'}"
        logger.debug(f"[Turn] mode_config stage={stage_key} side={side} streaming_tts={use_streaming_tts} "
                    f"streaming_listen={use_streaming_listen} mode={mode} t={time.time():.3f}")

        turn_watch = self._watch_root / f"{stage_key}_{side}"
        turn_watch.mkdir(parents=True, exist_ok=True)
        self._clear_dir(turn_watch)

        # Create playback cursor for synchronization
        playback_cursor = [0.0]  # seconds of audio "played" so far

        if use_streaming_listen:
            listener_deb.start_streaming_listen(
                turn_watch,
                stage_key,
                min_audio_seconds=self._min_audio_seconds,
                min_text_words=self._min_text_words,
                poll_interval=self._poll_interval,
                audio_format=self._audio_format,
                max_audio_wait_seconds=self._max_audio_wait,
                max_text_wait_seconds=self._max_text_wait,
                max_total_audio_seconds=self._max_total_audio,
                playback_cursor=playback_cursor,
            )
            time.sleep(max(0.5, self._poll_interval))

        use_live_bridge = self._env.time_control and getattr(player.config, "streaming_tts", False)

        response_holder: List[Optional[str]] = [None]
        error_holder: List[Optional[BaseException]] = [None]
        live_counts: List[int] = [0]
        speaker_done = threading.Event()
        done_ts: List[Optional[float]] = [None]

        def speaker_worker() -> None:
            logger.debug(f"[SpeakerWorker] thread_start stage={stage_key} side={side} t={time.time():.3f}")
            bridge_stop = threading.Event()
            bridge_thread: Optional[threading.Thread] = None
            if use_live_bridge:
                chunks_dir = tts_outputs_dir_from_log() / f"{player.config.type}_{stage_key}_{side}_chunks"
                bridge_thread = threading.Thread(
                    target=run_streaming_tts_chunk_copy_bridge,
                    name="TtsChunkBridge",
                    args=(chunks_dir, turn_watch, side, bridge_stop, live_counts),
                    kwargs={
                        "audio_format": self._audio_format,
                        "poll_interval": min(0.5, self._poll_interval),
                    },
                    daemon=True,
                )
                bridge_thread.start()
            try:
                logger.debug(f"[SpeakerWorker] generation_start stage={stage_key} side={side} t={time.time():.3f}")
                response_holder[0] = generate_fn()
                logger.debug(f"[SpeakerWorker] generation_end stage={stage_key} side={side} response_len={len(response_holder[0]) if response_holder[0] else 0} t={time.time():.3f}")
            except BaseException as e:
                error_holder[0] = e
                logger.exception("[OverlappingStreamingDebateEnv] Speaker worker failed")
            finally:
                bridge_stop.set()
                if bridge_thread is not None:
                    bridge_thread.join(timeout=self._listener_join_timeout)

            if error_holder[0] is None:
                try:
                    skip_posthoc = use_live_bridge and live_counts[0] > 0
                    mp3_path = tts_outputs_dir_from_log() / f"{player.config.type}_{stage_key}_{player.side}.mp3"
                    if not skip_posthoc and mp3_path.is_file() and mp3_path.stat().st_size > 2048:
                        logger.debug(f"[SpeakerWorker] posthoc_chunk_start mode=batch_tts mp3_path={mp3_path.name} t={time.time():.3f}")
                        audio = AudioSegment.from_file(str(mp3_path))
                        if self._split_mode == "fixed":
                            chunks = split_audio(audio, mode="fixed", time_seconds=self._chunk_seconds)
                        else:
                            chunks = split_audio(
                                audio, mode="silence", time_seconds=self._silence_window_seconds
                            )
                        logger.info(
                            f"[OverlappingStreamingDebateEnv] Post-hoc {len(chunks)} chunk(s) from "
                            f"{mp3_path.name} → {turn_watch} (realtime_pace=False)"
                        )
                        logger.debug(f"[SpeakerWorker] posthoc_chunk_split audio_duration={len(audio)/1000.0:.2f}s "
                                    f"num_chunks={len(chunks)} t={time.time():.3f}")
                        stream_start = time.time()
                        stream_chunks_to_directory(
                            chunks,
                            turn_watch,
                            side,
                            audio_format=self._audio_format,
                            dry_run=False,
                            max_total_seconds=None,
                            chunk_index_start=1,
                            realtime_pace=False,
                        )
                        stream_end = time.time()
                        logger.debug(f"[SpeakerWorker] posthoc_chunk_end num_chunks={len(chunks)} "
                                    f"stream_time={stream_end - stream_start:.3f}s t={stream_end:.3f}")
                    elif not skip_posthoc:
                        logger.warning(
                            f"[OverlappingStreamingDebateEnv] No TTS MP3 at {mp3_path} (skip chunk stream)."
                        )
                except Exception as e:
                    logger.error(f"[OverlappingStreamingDebateEnv] Post-hoc chunk stream failed: {e}")

            done_ts[0] = time.monotonic()
            speaker_done.set()
            logger.debug(f"[SpeakerWorker] thread_end stage={stage_key} side={side} t={time.time():.3f}")

        wt = threading.Thread(target=speaker_worker, name="SpeakerTurn", daemon=True)
        wt.start()

        self._playback_main_loop(
            turn_watch,
            side,
            stage_key,
            speaker_done,
            error_holder,
            done_ts,
            playback_cursor,
            grace_seconds=4.0,
        )

        wt.join(timeout=self._listener_join_timeout)
        if wt.is_alive():
            logger.warning("[OverlappingStreamingDebateEnv] Speaker thread still alive after join timeout.")

        if use_streaming_listen:
            listener_deb.stop_streaming_listen(self._listener_join_timeout)
        else:
            # Non-streaming listener: process after playback completes
            logger.debug(f"[NonStreamingListener] batch_listen_start stage={stage_key} side={listener} t={time.time():.3f}")
            # The listener will process via regular listen() call in next turn
            # Just log that we're in batch mode
            logger.debug(f"[NonStreamingListener] batch_listen_mode stage={stage_key} side={listener} "
                        f"listener_will_process_in_next_turn=True t={time.time():.3f}")

        if error_holder[0] is not None:
            err = error_holder[0]
            if isinstance(err, Exception):
                raise err
            raise RuntimeError(str(err))

        response = response_holder[0]
        if response is not None:
            rec: dict = {"stage": stage_key, "side": side, "content": response}
            if use_streaming_listen:
                rec["tree_via_streaming"] = True
            self._env.debate_process.append(rec)

        logger.debug(f"[Turn] turn_end stage={stage_key} side={side} t={time.time():.3f}")


__all__ = ["OverlappingStreamingDebateEnv", "parse_args", "main"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full debate with overlapping streaming listen (see module docstring)."
    )
    p.add_argument("--config", type=str, required=True, help="YAML config (same layout as env.py).")
    p.add_argument("--watch-dir", type=str, default=None, help="Optional root for per-turn watch subdirs.")
    p.add_argument("--debug", action="store_true", default=False)
    p.add_argument("--min-audio-seconds", type=float, default=15.0, help="Lower default than streaming.env for overlap.")
    p.add_argument("--min-text-words", type=int, default=40)
    p.add_argument("--poll-interval", type=float, default=1.0)
    p.add_argument("--audio-format", type=str, default="mp3")
    p.add_argument("--split-mode", type=str, choices=["fixed", "silence"], default="fixed")
    p.add_argument("--chunk-seconds", type=float, default=10.0)
    p.add_argument("--silence-window-seconds", type=float, default=0.7)
    p.add_argument("--max-audio-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-text-wait-seconds", type=float, default=0.0)
    p.add_argument("--max-total-audio-seconds", type=float, default=0.0)
    p.add_argument("--listener-join-timeout", type=float, default=300.0)
    p.add_argument("--min-playback-increment", type=float, default=3.0, help="Minimum playback increment in seconds (env-level cursor update frequency).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

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
    sde = OverlappingStreamingDebateEnv(
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


if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent.parent
    root = src_dir.parent
    for p in (src_dir, root):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    main()
