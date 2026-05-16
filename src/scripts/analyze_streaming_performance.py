#!/usr/bin/env python3
"""
Analyze streaming debate performance from log files.

Supports symmetric and mixed overlap modes (streaming/batch TTS × streaming/batch listen).
Extracts timing metrics from DEBUG logs to calculate:
- Speaker bubbles (waiting for TTS chunks or batch planning)
- Listen sessions (StreamingInputEnv stream + BatchListener batch)
- ASR real-time factors (stream listen only)
- End-to-end chunk latency
- File I/O overhead
- Tree update costs
- Pipeline efficiency

Usage:
    python analyze_streaming_performance.py <log_file> [--output output.json] [--verbose]
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE_ORDER = ["opening", "rebuttal", "closing"]
SIDE_ORDER = ["for", "against"]


def opponent_side(side: str) -> str:
    """Return the other debate side."""
    if side == "for":
        return "against"
    if side == "against":
        return "for"
    raise ValueError(f"Unknown side: {side!r}")


def stage_speech_order(reverse: bool = False) -> List[str]:
    """Speaking order within each stage (for then against, or reversed)."""
    return list(reversed(SIDE_ORDER)) if reverse else list(SIDE_ORDER)


def next_stage(stage: str) -> Optional[str]:
    try:
        i = STAGE_ORDER.index(stage)
    except ValueError:
        return None
    if i + 1 < len(STAGE_ORDER):
        return STAGE_ORDER[i + 1]
    return None


def listen_metrics_turn_key(
    stage: str, statement_side: str, reverse: bool = False
) -> Tuple[str, str]:
    """
    Turn key for StreamingInputEnv (listen) metrics on the debater who is listening.

    Attribution follows when the listener will use that speech:
      - Second speaker in stage listens to first in same stage
        (e.g. opening against listens to opening for).
      - First speaker in next stage listens to second speaker in previous stage
        (e.g. rebuttal for listens to opening against; closing for listens to rebuttal against).
    """
    sides = stage_speech_order(reverse)
    first, second = sides[0], sides[1]

    if statement_side == first:
        return (stage, second)

    if statement_side == second:
        nxt = next_stage(stage)
        if nxt is not None:
            return (nxt, first)
        return (stage, second)

    return (stage, opponent_side(statement_side))

_LOG_WALL_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})")
_TIMING_LINE_RE = re.compile(r"\[timing\]\s+phase=(\S+)\s+duration_s=([\d.]+)(.*)$")
_TIMING_KV_RE = re.compile(r"(\w+)=([^\s]+)")
_TTS_START_RE = re.compile(r"\[TTS-Start\]\s+Starting TTS for \S+ (\w+) (for|against)")
_CONFIG_STREAMING_RE = re.compile(
    r"streaming_tts['\"]?\s*:\s*(True|False).*?streaming_listen['\"]?\s*:\s*(True|False)",
    re.DOTALL,
)
_CONFIG_REVERSE_RE = re.compile(r"reverse['\"]?\s*:\s*(True|False)")


@dataclass
class Event:
    """Single log event with timestamp."""
    timestamp: float
    component: str
    event_type: str
    data: Dict[str, str]
    raw_line: str


@dataclass
class ChunkMetrics:
    """Metrics for a single TTS chunk."""
    chunk_idx: int
    detected_time: Optional[float] = None
    copied_time: Optional[float] = None
    assembled_time: Optional[float] = None
    playback_start_time: Optional[float] = None
    playback_end_time: Optional[float] = None
    duration: Optional[float] = None
    detection_latency: Optional[float] = None

    @property
    def e2e_latency(self) -> Optional[float]:
        """End-to-end latency: detection to playback complete."""
        if self.detected_time and self.playback_end_time:
            return self.playback_end_time - self.detected_time
        return None


@dataclass
class ASRMetrics:
    """Metrics for an ASR operation."""
    audio_start: float
    audio_end: float
    asr_start_time: float
    asr_end_time: float
    text_len: int

    @property
    def audio_duration(self) -> float:
        return self.audio_end - self.audio_start

    @property
    def asr_time(self) -> float:
        return self.asr_end_time - self.asr_start_time

    @property
    def rtf(self) -> float:
        """Real-time factor: ASR time / audio duration."""
        if self.audio_duration > 0:
            return self.asr_time / self.audio_duration
        return 0.0


@dataclass
class TreeUpdateMetrics:
    """Metrics for a tree update."""
    start_time: float
    end_time: float
    word_count: int

    @property
    def update_time(self) -> float:
        return self.end_time - self.start_time


@dataclass
class BubbleMetrics:
    """Wait/bubble timing."""
    wait_start: float
    wait_end: float
    context: str

    @property
    def duration(self) -> float:
        return self.wait_end - self.wait_start


@dataclass
class ListenSessionMetrics:
    """One listen session (streaming ASR or batch analyze_statement)."""
    statement_side: str
    listener_side: str
    listen_mode: str = "stream"  # "stream" (StreamingInputEnv) or "batch" (BatchListener)
    statement_stage: Optional[str] = None
    thread_start: Optional[float] = None
    thread_end: Optional[float] = None
    asr_operations: List[ASRMetrics] = None
    tree_updates: List[TreeUpdateMetrics] = None

    def __post_init__(self):
        if self.asr_operations is None:
            self.asr_operations = []
        if self.tree_updates is None:
            self.tree_updates = []

    @property
    def duration(self) -> Optional[float]:
        if self.thread_start is not None and self.thread_end is not None:
            return self.thread_end - self.thread_start
        return None


@dataclass
class TurnMetrics:
    """Complete metrics for one debate turn."""
    stage: str
    side: str
    turn_start: Optional[float] = None
    turn_end: Optional[float] = None

    # Mode detection
    streaming_tts: Optional[bool] = None
    streaming_listen: Optional[bool] = None
    mode: Optional[str] = None

    # Thread lifecycle
    speaker_thread_start: Optional[float] = None
    speaker_thread_end: Optional[float] = None
    listener_thread_start: Optional[float] = None
    listener_thread_end: Optional[float] = None
    playback_start: Optional[float] = None
    playback_end: Optional[float] = None

    # Generation timing
    generation_start: Optional[float] = None
    generation_end: Optional[float] = None

    # Batch processing timing
    posthoc_chunk_start: Optional[float] = None
    posthoc_chunk_end: Optional[float] = None
    batch_analyze_start: Optional[float] = None
    batch_analyze_end: Optional[float] = None

    # Chunk metrics
    chunks: Dict[int, ChunkMetrics] = None

    # ASR metrics
    asr_operations: List[ASRMetrics] = None

    # Tree updates
    tree_updates: List[TreeUpdateMetrics] = None

    # Bubbles
    speaker_bubbles: List[BubbleMetrics] = None

    # Streaming listen sessions (re-attributed to this debater's turn, not the speaker's)
    listen_sessions: List[ListenSessionMetrics] = None

    # File I/O
    file_writes: List[Tuple[float, float]] = None  # (start, end)
    file_reads: List[Tuple[float, float]] = None   # (start, end)

    # Batch sequential (tts=batch, listen=batch): metrics mapped onto streaming report fields
    batch_sequential: bool = False
    batch_audio_duration: Optional[float] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = {}
        if self.asr_operations is None:
            self.asr_operations = []
        if self.tree_updates is None:
            self.tree_updates = []
        if self.speaker_bubbles is None:
            self.speaker_bubbles = []
        if self.file_writes is None:
            self.file_writes = []
        if self.file_reads is None:
            self.file_reads = []
        if self.listen_sessions is None:
            self.listen_sessions = []

    @property
    def total_duration(self) -> Optional[float]:
        if self.turn_start and self.turn_end:
            return self.turn_end - self.turn_start
        return None

    @property
    def playback_duration(self) -> Optional[float]:
        if self.playback_start and self.playback_end:
            return self.playback_end - self.playback_start
        return None

    @property
    def speaker_duration(self) -> Optional[float]:
        if self.speaker_thread_start and self.speaker_thread_end:
            return self.speaker_thread_end - self.speaker_thread_start
        return None

    @property
    def listener_duration(self) -> Optional[float]:
        if self.listen_sessions:
            durations = [s.duration for s in self.listen_sessions if s.duration is not None]
            if durations:
                return sum(durations)
        if self.listener_thread_start and self.listener_thread_end:
            return self.listener_thread_end - self.listener_thread_start
        return None

    def _listen_to_opponent_sessions(self) -> List[ListenSessionMetrics]:
        opp = opponent_side(self.side)
        return [s for s in self.listen_sessions if s.statement_side == opp]

    @property
    def listen_to_opponent_duration(self) -> Optional[float]:
        """Time spent processing opponent speech (stream + batch listen sessions)."""
        durations = [s.duration for s in self._listen_to_opponent_sessions() if s.duration is not None]
        return sum(durations) if durations else None

    @property
    def listen_to_opponent_stream_duration(self) -> Optional[float]:
        durations = [
            s.duration for s in self._listen_to_opponent_sessions()
            if s.listen_mode == "stream" and s.duration is not None
        ]
        return sum(durations) if durations else None

    @property
    def listen_to_opponent_batch_duration(self) -> Optional[float]:
        durations = [
            s.duration for s in self._listen_to_opponent_sessions()
            if s.listen_mode == "batch" and s.duration is not None
        ]
        return sum(durations) if durations else None

    @property
    def listen_during_speech_duration(self) -> Optional[float]:
        """Opponent listen session while this debater is speaking."""
        durations = [
            s.duration for s in self.listen_sessions
            if s.statement_side == self.side and s.duration is not None
        ]
        return sum(durations) if durations else None

    @property
    def generation_time(self) -> Optional[float]:
        if self.generation_start and self.generation_end:
            return self.generation_end - self.generation_start
        return None

    @property
    def total_speaker_bubble(self) -> float:
        return sum(b.duration for b in self.speaker_bubbles)

    @property
    def time_to_first_chunk(self) -> Optional[float]:
        """Wait time for the first playable chunk (chunk_1)."""
        for b in self.speaker_bubbles:
            if b.context == "chunk_1":
                return b.duration
        return None

    @property
    def time_between_chunks(self) -> float:
        """Speaker bubble excluding the first chunk wait."""
        return sum(b.duration for b in self.speaker_bubbles if b.context != "chunk_1")

    @property
    def listener_bubble(self) -> Optional[float]:
        """Time from playback end until opponent stream-listen session ends (while we speak)."""
        for session in self.listen_sessions:
            if (
                session.listen_mode == "stream"
                and session.statement_side == self.side
                and session.thread_end is not None
                and self.playback_end is not None
            ):
                return session.thread_end - self.playback_end
        return None

    @property
    def true_overlap(self) -> Optional[float]:
        """Playback time minus speaker bubbles."""
        if self.playback_duration is not None:
            return self.playback_duration - self.total_speaker_bubble
        return None

    @property
    def avg_asr_rtf(self) -> Optional[float]:
        if self.asr_operations:
            return sum(op.rtf for op in self.asr_operations) / len(self.asr_operations)
        return None

    @property
    def audio_duration(self) -> Optional[float]:
        """Best-effort total audio duration for the turn."""
        if self.batch_audio_duration is not None:
            return self.batch_audio_duration
        chunk_durations = [
            chunk.duration for chunk in self.chunks.values() if chunk.duration is not None
        ]
        if chunk_durations:
            return sum(chunk_durations)

        if self.asr_operations:
            return max(op.audio_end for op in self.asr_operations)

        return None

    @property
    def total_tree_update_time(self) -> float:
        return sum(u.update_time for u in self.tree_updates)

    @property
    def avg_tree_update_time(self) -> Optional[float]:
        if self.tree_updates:
            return self.total_tree_update_time / len(self.tree_updates)
        return None

    @property
    def total_file_write_time(self) -> float:
        return sum(end - start for start, end in self.file_writes)

    @property
    def total_file_read_time(self) -> float:
        return sum(end - start for start, end in self.file_reads)

    @property
    def bottleneck(self) -> Optional[str]:
        """Identify bottleneck using bubble/wait time."""
        times = []
        if self.total_speaker_bubble > 0:
            times.append(('SPEAKER', self.total_speaker_bubble))
        if self.listener_bubble is not None and self.listener_bubble > 0:
            times.append(('LISTENER', self.listener_bubble))

        if times:
            return max(times, key=lambda x: x[1])[0]
        return None

    @property
    def overlap_efficiency(self) -> Optional[float]:
        """True overlap / (playback + listener bubble)."""
        if self.true_overlap and self.playback_duration and self.listener_bubble is not None:
            total = self.playback_duration + self.listener_bubble
            if total > 0:
                return self.true_overlap / total
        return None

    @property
    def posthoc_chunk_time(self) -> Optional[float]:
        """Time to split and stream chunks post-hoc (batch TTS mode)."""
        if self.posthoc_chunk_start and self.posthoc_chunk_end:
            return self.posthoc_chunk_end - self.posthoc_chunk_start
        return None

    @property
    def batch_analyze_time(self) -> Optional[float]:
        """Time to analyze statement in batch mode (non-streaming listener)."""
        if self.batch_analyze_start and self.batch_analyze_end:
            return self.batch_analyze_end - self.batch_analyze_start
        return None


def parse_mode_flags(mode: Optional[str]) -> Tuple[Optional[bool], Optional[bool]]:
    """Parse mode_config string into (speaker_tts_streaming, opponent_listen_streaming)."""
    if not mode:
        return None, None
    tts_m = re.search(r"tts=(stream|batch)", mode)
    listen_m = re.search(r"listen=(stream|batch)", mode)
    tts = True if tts_m and tts_m.group(1) == "stream" else False if tts_m else None
    listen = True if listen_m and listen_m.group(1) == "stream" else False if listen_m else None
    return tts, listen


def pipeline_label(turn: TurnMetrics) -> str:
    """Human-readable pipeline for this speak turn (speaker TTS + opponent listen)."""
    tts, listen = turn.streaming_tts, turn.streaming_listen
    if tts is None or listen is None:
        pt, pl = parse_mode_flags(turn.mode)
        if tts is None:
            tts = pt
        if listen is None:
            listen = pl
    tts_s = "stream" if tts is True else "batch" if tts is False else "?"
    listen_s = "stream" if listen is True else "batch" if listen is False else "?"
    return f"speaker_tts={tts_s}, opponent_listen={listen_s}"


def is_batch_sequential_turn(
    turn: TurnMetrics,
    default_streaming_tts: Optional[bool] = None,
    default_streaming_listen: Optional[bool] = None,
) -> bool:
    """
    Pure batch/sequential debate (no overlap chunk playback in log).

    Mixed modes (batch TTS + post-hoc chunks, stream listen, etc.) keep playback_start
    and use the standard overlap report path.
    """
    if turn.playback_start is not None:
        return False
    if turn.mode and str(turn.mode).startswith("sequential"):
        return True
    tts = turn.streaming_tts if turn.streaming_tts is not None else default_streaming_tts
    listen = turn.streaming_listen if turn.streaming_listen is not None else default_streaming_listen
    if tts is False and listen is False:
        return True
    if (
        tts is None
        and listen is None
        and turn.speaker_thread_start is None
    ):
        return True
    return False


def parse_config_from_log(log_path: Path) -> Tuple[Optional[bool], Optional[bool], bool]:
    default_tts: Optional[bool] = None
    default_listen: Optional[bool] = None
    reverse = False
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Config:" not in line:
                    continue
                m = _CONFIG_STREAMING_RE.search(line)
                if m:
                    default_tts = m.group(1) == "True"
                    default_listen = m.group(2) == "True"
                mr = _CONFIG_REVERSE_RE.search(line)
                if mr:
                    reverse = mr.group(1) == "True"
                break
    except OSError:
        pass
    return default_tts, default_listen, reverse


def load_debater_streaming_config(log_path: Path) -> Dict[str, Dict[str, bool]]:
    """Per-debater streaming_tts / streaming_listen from the logged YAML config."""
    out: Dict[str, Dict[str, bool]] = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "Config:" not in line:
                    continue
                for side in SIDE_ORDER:
                    block_m = re.search(
                        rf"['\"]side['\"]\s*:\s*['\"]{side}['\"](.*?)(?=['\"]side['\"]\s*:|'judge'|\Z)",
                        line,
                        re.DOTALL,
                    )
                    if not block_m:
                        continue
                    block = block_m.group(1)
                    tts_m = re.search(r"['\"]streaming_tts['\"]\s*:\s*(True|False)", block)
                    listen_m = re.search(r"['\"]streaming_listen['\"]\s*:\s*(True|False)", block)
                    if tts_m or listen_m:
                        out[side] = {
                            "streaming_tts": tts_m.group(1) == "True" if tts_m else False,
                            "streaming_listen": listen_m.group(1) == "True" if listen_m else False,
                        }
                break
    except OSError:
        pass
    return out


def debate_turn_order(
    turns: Dict[Tuple[str, str], TurnMetrics], reverse: bool = False
) -> List[Tuple[str, str]]:
    sides = list(reversed(SIDE_ORDER)) if reverse else SIDE_ORDER
    order: List[Tuple[str, str]] = []
    for stage in STAGE_ORDER:
        for side in sides:
            key = (stage, side)
            if key in turns:
                order.append(key)
    return order


def _parse_timing_kv(tail: str) -> Dict[str, str]:
    return {m.group(1): m.group(2) for m in _TIMING_KV_RE.finditer(tail)}


def load_timing_sums(log_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    sums: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[timing]" not in line:
                continue
            idx = line.find("[timing]")
            m = _TIMING_LINE_RE.search(line[idx:])
            if not m:
                continue
            fields = _parse_timing_kv(m.group(3) or "")
            stage, side = fields.get("stage"), fields.get("side")
            if stage and side:
                sums[(stage, side)][m.group(1)] += float(m.group(2))
    return {k: dict(v) for k, v in sums.items()}


def load_final_audio_durations(log_path: Path) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "phase=tts_trim_wall_clock" not in line:
                continue
            idx = line.find("[timing]")
            if idx < 0:
                continue
            m = _TIMING_LINE_RE.search(line[idx:])
            if not m:
                continue
            fields = _parse_timing_kv(m.group(3) or "")
            stage, side = fields.get("stage"), fields.get("side")
            audio = fields.get("audio_duration_s")
            if stage and side and audio:
                out[(stage, side)] = float(audio)
    return out


def load_tts_start_times(log_path: Path) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "[TTS-Start]" not in line:
                continue
            m = _TTS_START_RE.search(line)
            if not m:
                continue
            ts_m = _LOG_WALL_TS_RE.match(line)
            if not ts_m:
                continue
            try:
                ts = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            except ValueError:
                continue
            out[(m.group(1), m.group(2))] = ts
    return out


def _response_gen_time(
    turn: TurnMetrics,
    key: Tuple[str, str],
    timing: Dict[str, float],
    tts_starts: Dict[Tuple[str, str], float],
) -> float:
    """LLM / baseline API wall time before TTS (response generation)."""
    if turn.generation_time is not None and turn.generation_time > 0:
        return turn.generation_time
    api_s = timing.get("baseline_api_http", 0.0)
    if api_s > 0:
        return api_s
    if turn.turn_start is not None and key in tts_starts:
        return max(0.0, tts_starts[key] - turn.turn_start)
    active = (turn.turn_end - turn.turn_start) if turn.turn_start and turn.turn_end else 0.0
    tts_work = timing.get("tts_wall_clock", 0.0) + timing.get("tts_trim_wall_clock", 0.0)
    return max(0.0, active - tts_work)


def apply_batch_sequential_metrics(
    turns: Dict[Tuple[str, str], TurnMetrics],
    log_path: Path,
    default_streaming_tts: Optional[bool],
    default_streaming_listen: Optional[bool],
    reverse: bool,
) -> int:
    """
    Map batch/sequential turns onto the standard streaming report fields.

    While the opponent speaks, this side sleeps; at turn start it generates one chunk
    (LLM then batch TTS). Metric mapping:
      - speaker bubble  = response generation time
      - listener bubble = 0 (generation starts immediately after opponent finishes)
      - time to first chunk = LLM + TTS work
      - time between chunks = 0
    """
    timing_sums = load_timing_sums(log_path)
    final_audio = load_final_audio_durations(log_path)
    tts_starts = load_tts_start_times(log_path)
    order = debate_turn_order(turns, reverse=reverse)
    n = 0

    for i, key in enumerate(order):
        turn = turns[key]
        if not is_batch_sequential_turn(turn, default_streaming_tts, default_streaming_listen):
            continue

        turn.batch_sequential = True
        n += 1
        ph = timing_sums.get(key, {})
        audio = final_audio.get(key)
        if audio is None:
            audio = turn.audio_duration
        turn.batch_audio_duration = audio

        wait_opponent = 0.0
        if i > 0:
            prev_key = order[i - 1]
            prev = turns[prev_key]
            wait_opponent = (
                final_audio.get(prev_key)
                or prev.batch_audio_duration
                or prev.audio_duration
                or 0.0
            )

        # Planning time = turn wall clock from log (turn_end - turn_start): LLM + batch TTS work.
        planning_time = turn.total_duration or 0.0
        audio_s = audio or 0.0
        total_duration = audio_s + planning_time
        gen_s = _response_gen_time(turn, key, ph, tts_starts)
        tts_encode_s = ph.get("tts_wall_clock", 0.0)
        tts_trim_s = ph.get("tts_trim_wall_clock", 0.0)
        tts_work_s = tts_encode_s + tts_trim_s
        time_to_first = gen_s + tts_work_s
        if time_to_first <= 0 and planning_time > 0:
            time_to_first = planning_time

        playback = audio if audio is not None else planning_time
        speaker_bubble = gen_s
        listener_bubble = 0.0
        true_overlap = (playback - speaker_bubble) if playback is not None else None
        overlap_eff = None
        if true_overlap is not None and playback and playback > 0:
            overlap_eff = true_overlap / playback

        turn.speaker_bubbles = []
        turn.chunks = {1: ChunkMetrics(chunk_idx=1, duration=audio)} if audio else {}
        turn.batch_report: Dict[str, Any] = {
            "total_duration": total_duration,
            "playback_duration": playback,
            "speaker_duration": planning_time,
            "listener_duration": total_duration,
            "planning_time": planning_time,
            "generation_time": gen_s,
            "audio_duration": audio,
            "speaker_bubble_total": speaker_bubble,
            "time_to_first_chunk": time_to_first,
            "time_between_chunks": 0.0,
            "listener_bubble": listener_bubble,
            "listener_bubble_pct": 0.0,
            "speaker_bubble_pct": (speaker_bubble / playback * 100) if playback else None,
            "true_overlap": true_overlap,
            "overlap_efficiency": overlap_eff,
            "bottleneck": "SPEAKER" if speaker_bubble > 0 else None,
            "chunk_count": 1,
            "wait_opponent_s": wait_opponent,
        }

    return n


def parse_log_line(line: str) -> Optional[Event]:
    """Parse a single log line into an Event."""
    # Match: [Component] event_type key1=value1 key2=value2 t=timestamp
    match = re.search(r'\[([^\]]+)\]\s+(\w+)\s+(.*?)\s+t=([\d.]+)', line)
    if not match:
        return None

    component, event_type, data_str, timestamp = match.groups()

    # Parse key=value pairs
    data = {}
    for kv_match in re.finditer(r'(\w+)=([^\s]+)', data_str):
        key, value = kv_match.groups()
        data[key] = value

    return Event(
        timestamp=float(timestamp),
        component=component,
        event_type=event_type,
        data=data,
        raw_line=line.strip()
    )


def extract_turn_key(event: Event) -> Optional[Tuple[str, str]]:
    """Extract (stage, side) from event data for speaker/playback events."""
    stage = event.data.get('stage')
    side = event.data.get('side')
    if stage and side:
        return (stage, side)
    return None


def _active_listen_session(
    turns: Dict[Tuple[str, str], TurnMetrics],
    active: Optional[Tuple[Tuple[str, str], str, str]],
) -> Optional[ListenSessionMetrics]:
    if active is None:
        return None
    turn_key, statement_side, listen_mode = active
    turn = turns.get(turn_key)
    if turn is None or not turn.listen_sessions:
        return None
    for session in reversed(turn.listen_sessions):
        if (
            session.statement_side == statement_side
            and session.listen_mode == listen_mode
            and session.thread_end is None
        ):
            return session
    return None


def parse_log_file(log_path: Path, reverse: bool = False) -> Dict[Tuple[str, str], TurnMetrics]:
    """Parse log file and extract all metrics per turn."""
    turns: Dict[Tuple[str, str], TurnMetrics] = {}

    # Temporary state for tracking multi-event operations
    wait_chunk_starts: Dict[Tuple[str, str, int], float] = {}  # (stage, side, chunk_idx) -> time
    asr_starts: Dict[Tuple[str, str, str, float, float], float] = {}  # (stage, owner, stmt, a0, a1) -> time
    tree_starts: Dict[Tuple[str, str, str, int], float] = {}  # (stage, owner, stmt, words) -> time
    file_write_starts: Dict[Tuple[str, str], float] = {}
    active_listen: Optional[Tuple[Tuple[str, str], str, str]] = None  # (turn_key, statement_side, listen_mode)

    with open(log_path, 'r') as f:
        for line in f:
            event = parse_log_line(line)
            if not event:
                continue

            if event.component == 'StreamingInputEnv':
                statement_side = event.data.get('statement_side') or event.data.get('side')
                stage = event.data.get('stage')
                if not stage or not statement_side:
                    if active_listen is not None and event.event_type in (
                        'asr_start', 'asr_end', 'tree_update_start', 'tree_update_end',
                    ):
                        turn_key, stmt, _listen_mode = active_listen
                        stage, side = turn_key
                        statement_side = stmt
                    else:
                        continue
                else:
                    # Logged on the speaker's turn; reattribute_listen_sessions copies to the listener's speak turn.
                    turn_key = (stage, statement_side)
                    side = statement_side

                if turn_key not in turns:
                    turns[turn_key] = TurnMetrics(stage=stage, side=side)
                turn = turns[turn_key]

                if event.event_type == 'thread_start':
                    turn.listen_sessions.append(
                        ListenSessionMetrics(
                            statement_side=statement_side,
                            listener_side=opponent_side(statement_side),
                            listen_mode="stream",
                            statement_stage=stage,
                            thread_start=event.timestamp,
                        )
                    )
                    active_listen = (turn_key, statement_side, "stream")
                elif event.event_type == 'thread_end':
                    session = _active_listen_session(turns, active_listen)
                    if session is not None:
                        session.thread_end = event.timestamp
                    active_listen = None
                elif event.event_type == 'asr_start':
                    audio_range = _parse_audio_range(event.data.get('audio_range', ''))
                    if audio_range is None:
                        continue
                    audio_start, audio_end = audio_range
                    asr_starts[(stage, side, statement_side, audio_start, audio_end)] = event.timestamp
                elif event.event_type == 'asr_end':
                    audio_range = _parse_audio_range(event.data.get('audio_range', ''))
                    if audio_range is None:
                        continue
                    audio_start, audio_end = audio_range
                    asr_key = (stage, side, statement_side, audio_start, audio_end)
                    if asr_key in asr_starts:
                        op = ASRMetrics(
                            audio_start=audio_start,
                            audio_end=audio_end,
                            asr_start_time=asr_starts[asr_key],
                            asr_end_time=event.timestamp,
                            text_len=int(event.data.get('text_len', 0)),
                        )
                        session = _active_listen_session(turns, active_listen)
                        if session is not None:
                            session.asr_operations.append(op)
                        turn.asr_operations.append(op)
                elif event.event_type == 'tree_update_start':
                    words = int(event.data.get('words', 0))
                    tree_starts[(stage, side, statement_side, words)] = event.timestamp
                elif event.event_type == 'tree_update_end':
                    words = int(event.data.get('words', 0))
                    tree_key = (stage, side, statement_side, words)
                    if tree_key in tree_starts:
                        upd = TreeUpdateMetrics(
                            start_time=tree_starts[tree_key],
                            end_time=event.timestamp,
                            word_count=words,
                        )
                        session = _active_listen_session(turns, active_listen)
                        if session is not None:
                            session.tree_updates.append(upd)
                        turn.tree_updates.append(upd)
                continue

            if event.component == 'BatchListener':
                listener_side = event.data.get('side')
                stmt_side = event.data.get('opponent_side')
                stmt_stage = event.data.get('stage')
                if event.event_type == 'analyze_start':
                    if not listener_side or not stmt_side or not stmt_stage:
                        continue
                    turn_key = listen_metrics_turn_key(stmt_stage, stmt_side, reverse)
                    if turn_key not in turns:
                        turns[turn_key] = TurnMetrics(stage=turn_key[0], side=turn_key[1])
                    turn = turns[turn_key]
                    turn.listen_sessions.append(
                        ListenSessionMetrics(
                            statement_side=stmt_side,
                            listener_side=listener_side,
                            listen_mode="batch",
                            statement_stage=stmt_stage,
                            thread_start=event.timestamp,
                        )
                    )
                    turn.batch_analyze_start = event.timestamp
                    active_listen = (turn_key, stmt_side, "batch")
                elif event.event_type == 'analyze_end':
                    if not listener_side or not stmt_stage:
                        continue
                    end_key = (stmt_stage, listener_side)
                    if end_key not in turns:
                        turns[end_key] = TurnMetrics(stage=stmt_stage, side=listener_side)
                    turn = turns[end_key]
                    session = _active_listen_session(turns, active_listen)
                    if session is None:
                        for s in reversed(turn.listen_sessions):
                            if s.listen_mode == "batch" and s.thread_end is None:
                                session = s
                                break
                    if session is not None:
                        session.thread_end = event.timestamp
                    turn.batch_analyze_end = event.timestamp
                    if active_listen and active_listen[0] == end_key:
                        active_listen = None
                continue

            turn_key = extract_turn_key(event)
            if not turn_key:
                continue

            stage, side = turn_key
            if turn_key not in turns:
                turns[turn_key] = TurnMetrics(stage=stage, side=side)

            turn = turns[turn_key]

            # Process event
            if event.component == 'Turn':
                if event.event_type == 'turn_start':
                    turn.turn_start = event.timestamp
                elif event.event_type == 'turn_end':
                    turn.turn_end = event.timestamp
                elif event.event_type == 'mode_config':
                    turn.streaming_tts = event.data.get('streaming_tts') == 'True'
                    turn.streaming_listen = event.data.get('streaming_listen') == 'True'
                    turn.mode = event.data.get('mode')

            elif event.component == 'SpeakerWorker':
                if event.event_type == 'thread_start':
                    turn.speaker_thread_start = event.timestamp
                elif event.event_type == 'thread_end':
                    turn.speaker_thread_end = event.timestamp
                elif event.event_type == 'generation_start':
                    turn.generation_start = event.timestamp
                elif event.event_type == 'generation_end':
                    turn.generation_end = event.timestamp
                elif event.event_type == 'posthoc_chunk_start':
                    turn.posthoc_chunk_start = event.timestamp
                elif event.event_type == 'posthoc_chunk_end':
                    turn.posthoc_chunk_end = event.timestamp

            elif event.component == 'PlaybackMain':
                if event.event_type == 'playback_start':
                    turn.playback_start = event.timestamp
                elif event.event_type == 'playback_end':
                    turn.playback_end = event.timestamp
                elif event.event_type == 'wait_chunk_start':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    wait_chunk_starts[(stage, side, chunk_idx)] = event.timestamp
                elif event.event_type == 'wait_chunk_end':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    wait_key = (stage, side, chunk_idx)
                    if wait_key in wait_chunk_starts:
                        turn.speaker_bubbles.append(BubbleMetrics(
                            wait_start=wait_chunk_starts[wait_key],
                            wait_end=event.timestamp,
                            context=f"chunk_{chunk_idx}"
                        ))
                elif event.event_type == 'chunk_assembled':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    if chunk_idx not in turn.chunks:
                        turn.chunks[chunk_idx] = ChunkMetrics(chunk_idx=chunk_idx)
                    turn.chunks[chunk_idx].assembled_time = event.timestamp
                    turn.chunks[chunk_idx].duration = float(event.data.get('duration', '0').rstrip('s'))
                elif event.event_type == 'chunk_playback_start':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    if chunk_idx not in turn.chunks:
                        turn.chunks[chunk_idx] = ChunkMetrics(chunk_idx=chunk_idx)
                    turn.chunks[chunk_idx].playback_start_time = event.timestamp
                elif event.event_type == 'chunk_playback_end':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    if chunk_idx not in turn.chunks:
                        turn.chunks[chunk_idx] = ChunkMetrics(chunk_idx=chunk_idx)
                    turn.chunks[chunk_idx].playback_end_time = event.timestamp
                elif event.event_type == 'file_write':
                    # File write is atomic in our case (start/end in same log line)
                    write_time = float(event.data.get('write_time', '0').rstrip('s'))
                    turn.file_writes.append((event.timestamp - write_time, event.timestamp))

            elif event.component == 'TtsChunkBridge':
                if event.event_type == 'chunk_detected':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    if chunk_idx not in turn.chunks:
                        turn.chunks[chunk_idx] = ChunkMetrics(chunk_idx=chunk_idx)
                    turn.chunks[chunk_idx].detected_time = event.timestamp
                elif event.event_type == 'chunk_copied':
                    chunk_idx = int(event.data.get('chunk_idx', 0))
                    if chunk_idx not in turn.chunks:
                        turn.chunks[chunk_idx] = ChunkMetrics(chunk_idx=chunk_idx)
                    turn.chunks[chunk_idx].copied_time = event.timestamp
                    detection_latency = float(event.data.get('detection_latency', '0').rstrip('s'))
                    turn.chunks[chunk_idx].detection_latency = detection_latency

    reattribute_listen_sessions(turns, reverse)

    for turn in turns.values():
        if not turn.listen_sessions:
            continue
        starts = [s.thread_start for s in turn.listen_sessions if s.thread_start is not None]
        ends = [s.thread_end for s in turn.listen_sessions if s.thread_end is not None]
        if starts:
            turn.listener_thread_start = min(starts)
        if ends:
            turn.listener_thread_end = max(ends)

    return turns


def reattribute_listen_sessions(
    turns: Dict[Tuple[str, str], TurnMetrics], reverse: bool = False
) -> None:
    """
    Copy listen sessions to the debater's speak turn when they consume that speech.

    Sessions stay on the speaker's turn for listener_bubble (opponent still listening
    after our playback). A copy on the listener's next speak turn drives listen_to_opponent
    (e.g. rebuttal for ← opening against, closing for ← rebuttal against).
    """
    for turn_key, turn in list(turns.items()):
        stage, speaker = turn_key
        for session in turn.listen_sessions:
            if session.listen_mode != "stream":
                continue
            stmt = session.statement_side
            if stmt != speaker:
                continue
            listener = session.listener_side
            consumer_key = listen_metrics_turn_key(stage, stmt, reverse)
            if consumer_key == turn_key:
                continue
            if consumer_key[1] != listener:
                continue
            if consumer_key not in turns:
                turns[consumer_key] = TurnMetrics(stage=consumer_key[0], side=consumer_key[1])
            if session not in turns[consumer_key].listen_sessions:
                turns[consumer_key].listen_sessions.append(session)


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_audio_range(value: str) -> Optional[Tuple[float, float]]:
    if not value or "-" not in value:
        return None
    left, right = value.split("-", 1)
    try:
        return float(left), float(right.rstrip("s"))
    except ValueError:
        return None


def load_tts_chunk_profiles(outputs_dir: Path) -> Dict[Tuple[str, str], List[Dict[str, str]]]:
    """Load per-turn streaming TTS chunk profiles from *_chunks/chunk_profile.csv."""
    profiles: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    if not outputs_dir.exists():
        return profiles

    for chunk_csv in outputs_dir.glob("*_chunks/chunk_profile.csv"):
        parent_name = chunk_csv.parent.name  # e.g., treedebater_opening_for_chunks
        m = re.match(r"^[^_]+_([^_]+)_(for|against)_chunks$", parent_name)
        if not m:
            continue
        stage, side = m.group(1), m.group(2)
        rows: List[Dict[str, str]] = []
        try:
            with chunk_csv.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row:
                        rows.append(row)
        except OSError:
            continue
        profiles[(stage, side)] = rows

    return profiles


def generate_summary(
    turns: Dict[Tuple[str, str], TurnMetrics],
    tts_profiles: Optional[Dict[Tuple[str, str], List[Dict[str, str]]]] = None,
) -> Dict:
    """Generate summary statistics across all turns."""
    summary = {
        'total_turns': len(turns),
        'turns': {}
    }
    tts_profiles = tts_profiles or {}

    for turn_key, turn in turns.items():
        stage, side = turn_key
        turn_id = f"{stage}_{side}"

        if turn.batch_sequential and hasattr(turn, "batch_report"):
            br = turn.batch_report
            turn_summary = {
                "stage": stage,
                "side": side,
                "mode": turn.mode or "tts=batch_listen=batch",
                "streaming_tts": turn.streaming_tts,
                "streaming_listen": turn.streaming_listen,
                "batch_sequential": True,
                "asr_operations": 0,
                "tree_updates": 0,
                "total_tree_update_time": 0.0,
                "avg_tree_update_time": None,
                "total_file_write_time": turn.total_file_write_time,
                "total_file_read_time": turn.total_file_read_time,
                "posthoc_chunk_time": turn.posthoc_chunk_time,
                "batch_analyze_time": turn.batch_analyze_time,
                "speaker_bubbles": [],
            }
            turn_summary.update(br)
            summary["turns"][turn_id] = turn_summary
            continue

        turn_summary = {
            'stage': stage,
            'side': side,
            'mode': turn.mode,
            'streaming_tts': turn.streaming_tts,
            'streaming_listen': turn.streaming_listen,
            'batch_sequential': False,
            'total_duration': turn.total_duration,
            'playback_duration': turn.playback_duration,
            'speaker_duration': turn.speaker_duration,
            'listener_duration': turn.listener_duration,
            'listen_to_opponent_duration': turn.listen_to_opponent_duration,
            'listen_to_opponent_stream_duration': turn.listen_to_opponent_stream_duration,
            'listen_to_opponent_batch_duration': turn.listen_to_opponent_batch_duration,
            'pipeline_label': pipeline_label(turn),
            'generation_time': turn.generation_time,
            'speaker_bubble_total': turn.total_speaker_bubble,
            'time_to_first_chunk': turn.time_to_first_chunk,
            'time_between_chunks': turn.time_between_chunks,
            'speaker_bubble_pct': (turn.total_speaker_bubble / turn.playback_duration * 100) if turn.playback_duration else None,
            'listener_bubble': turn.listener_bubble,
            'listener_bubble_pct': (
                turn.listener_bubble / turn.listen_during_speech_duration * 100
            ) if turn.listen_during_speech_duration and turn.listener_bubble else None,
            'true_overlap': turn.true_overlap,
            'overlap_efficiency': turn.overlap_efficiency,
            'bottleneck': turn.bottleneck,
            'chunk_count': len(turn.chunks),
            'audio_duration': turn.audio_duration,
            'asr_operations': len(turn.asr_operations),
            'avg_asr_rtf': turn.avg_asr_rtf,
            'tree_updates': len(turn.tree_updates),
            'total_tree_update_time': turn.total_tree_update_time,
            'avg_tree_update_time': turn.avg_tree_update_time,
            'total_file_write_time': turn.total_file_write_time,
            'total_file_read_time': turn.total_file_read_time,
            'posthoc_chunk_time': turn.posthoc_chunk_time,
            'batch_analyze_time': turn.batch_analyze_time,
        }

        # Chunk details
        if turn.chunks:
            chunk_latencies = [c.e2e_latency for c in turn.chunks.values() if c.e2e_latency]
            if chunk_latencies:
                turn_summary['chunk_latency_avg'] = sum(chunk_latencies) / len(chunk_latencies)
                turn_summary['chunk_latency_min'] = min(chunk_latencies)
                turn_summary['chunk_latency_max'] = max(chunk_latencies)

        # ASR details
        if turn.asr_operations:
            rtfs = [op.rtf for op in turn.asr_operations]
            turn_summary['asr_rtf_min'] = min(rtfs)
            turn_summary['asr_rtf_max'] = max(rtfs)
            turn_summary['asr_real_time'] = all(rtf < 1.0 for rtf in rtfs)

        # Bubble breakdown
        turn_summary['speaker_bubbles'] = [
            {'duration': b.duration, 'context': b.context}
            for b in turn.speaker_bubbles
        ]
        if turn.listen_sessions:
            turn_summary['listen_sessions'] = [
                {
                    'statement_side': s.statement_side,
                    'listener_side': s.listener_side,
                    'listen_mode': s.listen_mode,
                    'statement_stage': s.statement_stage,
                    'duration': s.duration,
                    'asr_operations': len(s.asr_operations),
                    'tree_updates': len(s.tree_updates),
                }
                for s in turn.listen_sessions
            ]

        # Streaming TTS chunk-profile stats (if available)
        profile_rows = tts_profiles.get(turn_key, [])
        if profile_rows:
            ref_counts = [_to_float(r.get("n_ref_used", "0"), 0.0) for r in profile_rows]
            refined_counts = [x for x in ref_counts if x > 0]
            timed_out = sum(1 for r in profile_rows if _to_bool(r.get("timed_out", "false")))
            chunk_total_times = [_to_float(r.get("chunk_total_s", "0"), 0.0) for r in profile_rows]
            tts_api_times = [_to_float(r.get("tts_api_s", "0"), 0.0) for r in profile_rows]
            refine_times = [_to_float(r.get("refine_total_s", "0"), 0.0) for r in profile_rows]
            first_profile_row = min(
                profile_rows,
                key=lambda r: int(_to_float(r.get("chunk_idx", "0"), 0.0)),
            )
            first_chunk_gen_time_s = _to_float(first_profile_row.get("chunk_total_s", "0"), 0.0)
            turn_summary["tts_profile_chunks"] = len(profile_rows)
            turn_summary["tts_chunks_refined"] = len(refined_counts)
            turn_summary["tts_total_refinements"] = int(sum(ref_counts))
            turn_summary["tts_avg_refinements_per_chunk"] = sum(ref_counts) / len(ref_counts)
            turn_summary["tts_avg_refinements_refined_chunks"] = (
                sum(refined_counts) / len(refined_counts) if refined_counts else 0.0
            )
            turn_summary["tts_timed_out_chunks"] = timed_out
            turn_summary["first_chunk_gen_time_s"] = first_chunk_gen_time_s
            turn_summary["chunk_gen_total_s_avg"] = sum(chunk_total_times) / len(chunk_total_times)
            turn_summary["chunk_gen_total_s_min"] = min(chunk_total_times)
            turn_summary["chunk_gen_total_s_max"] = max(chunk_total_times)
            turn_summary["chunk_tts_api_s_avg"] = sum(tts_api_times) / len(tts_api_times)
            turn_summary["chunk_refine_s_avg"] = sum(refine_times) / len(refine_times)

        # Speaker bubble from chunk_profile only for live streaming TTS (not batch/post-hoc).
        if turn_summary.get("first_chunk_gen_time_s") is not None and turn.streaming_tts is True:
            derived_speaker_bubble = turn_summary["first_chunk_gen_time_s"] + turn.time_between_chunks
            turn_summary["speaker_bubble_total"] = derived_speaker_bubble
            turn_summary["speaker_bubble_pct"] = (
                derived_speaker_bubble / turn.playback_duration * 100
            ) if turn.playback_duration else None
            turn_summary["true_overlap"] = (
                turn.playback_duration - derived_speaker_bubble
            ) if turn.playback_duration is not None else None
            if turn.playback_duration is not None and turn.listener_bubble is not None:
                denom = turn.playback_duration + turn.listener_bubble
                turn_summary["overlap_efficiency"] = (
                    turn_summary["true_overlap"] / denom if denom > 0 else None
                )
            else:
                turn_summary["overlap_efficiency"] = None
            if turn.listener_bubble is not None:
                turn_summary["bottleneck"] = (
                    "SPEAKER" if derived_speaker_bubble >= turn.listener_bubble else "LISTENER"
                )

        summary['turns'][turn_id] = turn_summary

    return summary


def print_summary(summary: Dict, verbose: bool = False):
    """Print human-readable summary."""
    print("\n" + "="*80)
    print("STREAMING DEBATE PERFORMANCE ANALYSIS")
    print("="*80 + "\n")

    print("--- Metric Definitions ---")
    print("  --- Timing Overview ---")
    print("  Total duration:      turn_end - turn_start")
    print("  Playback duration:   playback_end - playback_start")
    print("  Audio duration:      sum(chunk durations), fallback=max(ASR audio_end)")
    # print("  Speaker duration:    speaker_thread_end - speaker_thread_start")
    # print("  Listener duration:   sum(streaming listen sessions on this debater's turn)")
    print("  Listen to opponent:  stream + batch listen sessions (on this debater's speak turn)")
    print("  Generation time:     generation_end - generation_start")
    print("  --- Mixed modes ---")
    print("  mode_config per speak turn: speaker TTS + opponent listen (stream|batch)")
    print("  Batch listen: BatchListener analyze_* at start of listener's speak turn")
    print("  Stream listen: StreamingInputEnv during opponent speech (reattributed to listener turn)")
    print("  --- Turn attribution (listen sessions) ---")
    print("  Second in stage:       listen to first in same stage (e.g. opening against ← opening for)")
    print("  First in next stage:   listen to second in prev stage (e.g. rebuttal for ← opening against)")
    print("  --- Bubble Analysis ---")
    print("  Speaker bubble:      first chunk gen time + Time Between Chunks")
    print("  Listener bubble:     listener_thread_end - playback_end")
    print("  --- Efficiency Metrics ---")
    print("  Time to First Chunk: wait time for chunk_1 (LLM work + first TTS chunk + refine time)")
    print("  Time Between Chunks: sum(waits for chunk_2+)")
    # print("  True overlap:        playback_duration - speaker_bubble")
    # print("  Overlap efficiency:  true_overlap / (playback_duration + listener_bubble)")
    print("  Bottleneck:          max(speaker bubble, listener bubble)")
    print("  --- Pipeline Stats ---")
    print("    Speaking side:")
    print("  Avg chunk latency:   mean(playback_end - detected)")
    print("  Chunk gen time:      from chunk_profile.csv (chunk_total_s/tts_api_s/refine_total_s)")
    print("  TTS refinements:     from chunk_profile.csv n_ref_used stats")
    print("  TTS timeouts:        count(chunks where timed_out=True in chunk_profile.csv)")
    print("    Listening side:")
    print("  Avg ASR RTF:         mean((asr_end - asr_start) / (audio_end - audio_start))")
    print("  Avg update time:    mean(tree_update_end - tree_update_start)")
    print("  --- I/O Overhead ---")
    print("  File write time:     sum(file_write.write_time)")
    print("  File read time:      sum(file_read_end - file_read_start)")
    print("  --- Batch Mode Metrics ---")
    print("  Post-hoc chunk time: posthoc_chunk_end - posthoc_chunk_start")
    print("  Batch analyze time:  batch_analyze_end - batch_analyze_start")
    if any(t.get("batch_sequential") for t in summary["turns"].values()):
        print("  --- Batch sequential (streaming_tts=False, streaming_listen=False) ---")
        print("  Opponent speaks while this side sleeps; then one chunk (LLM → batch TTS).")
        print("  Total duration:      audio_duration + planning_time (planning = turn wall clock from log)")
        print("  Speaker bubble:      response generation time (not chunk waits)")
        print("  Listener bubble:     0 (generation starts right after opponent finishes)")
        print("  Time to First Chunk: LLM work + TTS work (single chunk)")
        print("  Time Between Chunks: 0")
        print("  Pipeline stats:      omitted (no streaming pipeline)")
    print()

    print(f"Total turns analyzed: {summary['total_turns']}\n")

    for turn_id, turn in summary['turns'].items():
        print(f"\n{'='*80}")
        print(f"Turn: {turn['stage']} ({turn['side']})")
        print(f"{'='*80}")

        ideal_audio_duration = 120 if turn['stage'] == 'closing' else 240

        print(f"\n--- Mode Configuration ---")
        mode_desc = turn.get('mode', 'unknown')
        print(f"  Mode:                {mode_desc}")
        if turn.get('pipeline_label'):
            print(f"  Pipeline:            {turn['pipeline_label']}")
        print(f"  Streaming TTS:       {turn.get('streaming_tts', 'N/A')} (this side, while speaking)")
        print(f"  Opponent listen:     {turn.get('streaming_listen', 'N/A')} (while this side speaks)")

        print(f"\n--- Timing Overview ---")
        print(f"  Total duration:      {turn['total_duration']:.2f}s" if turn['total_duration'] else "  Total duration:      N/A")
        print(f"  Audio duration:      {turn['audio_duration']:.2f}s (ideal={ideal_audio_duration}s, gap={turn['audio_duration'] - ideal_audio_duration:.2f}s)" if turn.get('audio_duration') is not None else "  Audio duration:      N/A")
        print(f"  Playback duration:   {turn['playback_duration']:.2f}s" if turn['playback_duration'] else "  Playback duration:   N/A")
        # print(f"  Speaker duration:    {turn['speaker_duration']:.2f}s" if turn.get('speaker_duration') else "  Speaker duration:    N/A")
        # if turn.get('listener_duration'):
        #     print(f"  Listener duration:   {turn['listener_duration']:.2f}s")
        # elif turn.get('listen_sessions'):
        #     print("  Listener duration:   N/A")
        # else:
        #     print("  Listener duration:   — (speak-only turn)")
        if turn.get('listen_to_opponent_duration') is not None:
            stream_s = turn.get('listen_to_opponent_stream_duration')
            batch_s = turn.get('listen_to_opponent_batch_duration')
            parts = []
            if stream_s:
                parts.append(f"stream {stream_s:.2f}s")
            if batch_s:
                parts.append(f"batch {batch_s:.2f}s")
            detail = f" ({', '.join(parts)})" if parts else ""
            print(
                f"  Listen to opponent:  {turn['listen_to_opponent_duration']:.2f}s{detail}"
            )
        print(f"  Generation time:     {turn['generation_time']:.2f}s" if turn['generation_time'] else "  Generation time:     N/A")

        print(f"\n--- Bubble Analysis ---")
        sb_total = turn.get('speaker_bubble_total')
        first_wait = turn.get('time_to_first_chunk')
        inter_wait = turn.get('time_between_chunks')
        sb_pct = turn.get('speaker_bubble_pct')
        if sb_total is not None:
            if sb_pct is not None:
                print(f"  Speaker bubble:      {sb_total:.2f}s ({sb_pct:.1f}% of playback)")
            else:
                print(f"  Speaker bubble:      {sb_total:.2f}s (playback duration N/A, no %)")
        if turn.get("batch_sequential"):
            print(f"  Listener bubble:     {turn.get('listener_bubble', 0.0):.2f}s")
        elif turn['listener_bubble'] is not None:
            lb_pct = turn.get('listener_bubble_pct')
            if lb_pct is not None:
                print(f"  Listener bubble:     {turn['listener_bubble']:.2f}s ({lb_pct:.1f}% of listener time)")
            else:
                print(f"  Listener bubble:     {turn['listener_bubble']:.2f}s")

        print(f"\n--- Efficiency Metrics ---")
        print(f"  Time to First Chunk: {first_wait:.2f}s" if first_wait is not None else "  Time to First Chunk: N/A")
        print(f"  Time Between Chunks: {inter_wait:.2f}s")
        # if turn.get('true_overlap') is not None:
        #     print(f"  True overlap:        {turn['true_overlap']:.2f}s")
        # if turn.get('overlap_efficiency') is not None:
        #     print(f"  Overlap efficiency:  {turn['overlap_efficiency']*100:.1f}%")
        if turn['bottleneck']:
            print(f"  Bottleneck:          {turn['bottleneck']}")

        if turn.get("batch_sequential"):
            if turn.get("planning_time") is not None:
                print(
                    f"\n  (batch) planning time (log turn wall): {turn['planning_time']:.2f}s; "
                    f"total = audio {turn.get('audio_duration', 0):.2f}s + planning"
                )
            if turn.get("wait_opponent_s"):
                print(
                    f"  (batch) opponent speech while sleeping: {turn['wait_opponent_s']:.2f}s "
                    f"(not included in total duration)"
                )
            print(f"\n--- I/O Overhead ---")
            print(f"  File write time:     {turn['total_file_write_time']:.3f}s")
            print(f"  File read time:      {turn['total_file_read_time']:.3f}s")
            continue

        print(f"\n--- Pipeline Stats (Speaking Side) ---")
        tts_profile_chunks = turn.get('tts_profile_chunks')
        if tts_profile_chunks is not None:
            print(
                f"  Chunks processed:    tts_profile_chunks={tts_profile_chunks} "
                f"(playback_chunks={turn['chunk_count']})"
            )
        else:
            print(f"  Chunks processed:    tts_profile_chunks=N/A (playback_chunks={turn['chunk_count']})")
        if turn.get('chunk_latency_avg'):
            print(f"  Avg chunk latency:   {turn['chunk_latency_avg']:.2f}s (min={turn['chunk_latency_min']:.2f}s, max={turn['chunk_latency_max']:.2f}s)")
        if turn.get('chunk_gen_total_s_avg') is not None:
            print(
                f"  Chunk gen time:      avg={turn['chunk_gen_total_s_avg']:.2f}s "
                f"(min={turn['chunk_gen_total_s_min']:.2f}s, max={turn['chunk_gen_total_s_max']:.2f}s)"
            )
            print(
                f"  Chunk gen breakdown: avg_tts_api={turn['chunk_tts_api_s_avg']:.2f}s "
                f"avg_refine={turn['chunk_refine_s_avg']:.2f}s"
            )
        if turn.get('tts_profile_chunks') is not None:
            print(
                f"  TTS refinements:     total={turn['tts_total_refinements']} "
                f"refined_chunks={turn['tts_chunks_refined']}/{turn['tts_profile_chunks']} "
                f"avg/chunk={turn['tts_avg_refinements_per_chunk']:.2f}"
            )
            print(
                f"  TTS timeouts:        {turn['tts_timed_out_chunks']} chunk(s)"
            )

        if turn.get('listen_sessions'):
            print(f"\n--- Listen Sessions ---")
            for i, sess in enumerate(turn['listen_sessions'], 1):
                dur = sess.get('duration')
                dur_s = f"{dur:.2f}s" if dur is not None else "N/A"
                stmt_stage = sess.get('statement_stage') or turn['stage']
                print(
                    f"  Session {i}: [{sess.get('listen_mode', 'stream')}] "
                    f"{stmt_stage} {sess['statement_side']} "
                    f"(listener={sess['listener_side']}) {dur_s}, "
                    f"ASR={sess['asr_operations']}, tree={sess['tree_updates']}"
                )

        if turn.get('listen_sessions') or turn.get('asr_operations'):
            print(f"\n--- Pipeline Stats (Listening Side) ---")
            print(f"  ASR operations:      {turn['asr_operations']}")
            if turn['avg_asr_rtf'] is not None:
                status = "✓ REAL-TIME" if turn.get('asr_real_time') else "✗ LAGGING"
                print(f"  Avg ASR RTF:         {turn['avg_asr_rtf']:.3f} {status}")
            print(f"  Tree updates:        {turn['tree_updates']}")
            print(
                f"  Avg update time:    {turn['avg_tree_update_time']:.2f}s"
                if turn.get('avg_tree_update_time') is not None
                else "  Avg update time:    N/A"
            )

        print(f"\n--- I/O Overhead ---")
        print(f"  File write time:     {turn['total_file_write_time']:.3f}s")
        print(f"  File read time:      {turn['total_file_read_time']:.3f}s")

        # Mode-specific metrics
        if turn.get('posthoc_chunk_time') is not None:
            print(f"\n--- Batch TTS Processing ---")
            print(f"  Post-hoc chunk time: {turn['posthoc_chunk_time']:.3f}s (split + stream)")

        if turn.get('batch_analyze_time') is not None and not turn.get('listen_to_opponent_batch_duration'):
            print(f"\n--- Batch Listener Processing ---")
            print(f"  Batch analyze time:  {turn['batch_analyze_time']:.3f}s (statement analysis)")

        if verbose and turn['speaker_bubbles']:
            print(f"\n--- Speaker Bubble Breakdown ---")
            for i, bubble in enumerate(turn['speaker_bubbles'], 1):
                print(f"  Bubble {i}: {bubble['duration']:.3f}s ({bubble['context']})")


def main():
    parser = argparse.ArgumentParser(description='Analyze streaming debate performance from log files')
    parser.add_argument('log_file', type=str, help='Path to log file')
    parser.add_argument('--output', '-o', type=str, help='Output JSON file (optional)')
    parser.add_argument(
        '--outputs-dir',
        type=str,
        default=None,
        help='Directory containing per-turn *_chunks/chunk_profile.csv (default: <log_stem>_outputs)',
    )
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return 1

    print(f"Parsing log file: {log_path}")
    cfg_tts, cfg_listen, reverse = parse_config_from_log(log_path)
    debater_cfg = load_debater_streaming_config(log_path)
    turns = parse_log_file(log_path, reverse=reverse)
    for turn in turns.values():
        side_cfg = debater_cfg.get(turn.side, {})
        if turn.streaming_tts is None:
            if turn.side in debater_cfg:
                turn.streaming_tts = side_cfg.get("streaming_tts")
            elif cfg_tts is not None:
                turn.streaming_tts = cfg_tts
        if turn.mode:
            pt, pl = parse_mode_flags(turn.mode)
            if turn.streaming_tts is None and pt is not None:
                turn.streaming_tts = pt
            if turn.streaming_listen is None and pl is not None:
                turn.streaming_listen = pl
        if turn.mode is None and turn.streaming_tts is False and turn.streaming_listen is False:
            turn.mode = "tts=batch_listen=batch"
    apply_batch_sequential_metrics(turns, log_path, cfg_tts, cfg_listen, reverse)
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else Path(str(log_path).replace('.log', '_outputs'))
    tts_profiles = load_tts_chunk_profiles(outputs_dir)

    if not turns:
        print("No streaming turns found in log file.")
        return 1

    summary = generate_summary(turns, tts_profiles=tts_profiles)

    print_summary(summary, verbose=args.verbose)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDetailed metrics saved to: {output_path}")

    return 0


if __name__ == '__main__':
    exit(main())
