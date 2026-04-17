#!/usr/bin/env python3
"""
Analyze streaming debate performance from log files.

Extracts timing metrics from DEBUG logs to calculate:
- Speaker bubbles (waiting for TTS chunks)
- Listener bubbles (post-playback processing)
- ASR real-time factors
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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

    # File I/O
    file_writes: List[Tuple[float, float]] = None  # (start, end)
    file_reads: List[Tuple[float, float]] = None   # (start, end)

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
        if self.listener_thread_start and self.listener_thread_end:
            return self.listener_thread_end - self.listener_thread_start
        return None

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
        """Time from playback end to listener thread end."""
        if self.playback_end and self.listener_thread_end:
            return self.listener_thread_end - self.playback_end
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
    """Extract (stage, side) from event data."""
    stage = event.data.get('stage')
    # Some components (e.g., StreamingInputEnv) emit `statement_side`
    # instead of `side`, so support both field names.
    side = event.data.get('side') or event.data.get('statement_side')
    if stage and side:
        return (stage, side)
    return None


def parse_log_file(log_path: Path) -> Dict[Tuple[str, str], TurnMetrics]:
    """Parse log file and extract all metrics per turn."""
    turns: Dict[Tuple[str, str], TurnMetrics] = {}

    # Temporary state for tracking multi-event operations
    wait_chunk_starts: Dict[Tuple[str, str, int], float] = {}  # (stage, side, chunk_idx) -> time
    asr_starts: Dict[Tuple[str, str, float, float], float] = {}  # (stage, side, start, end) -> time
    tree_starts: Dict[Tuple[str, str, int], float] = {}  # (stage, side, word_count) -> time
    file_write_starts: Dict[Tuple[str, str], float] = {}
    active_streaming_turn: Optional[Tuple[str, str]] = None

    with open(log_path, 'r') as f:
        for line in f:
            event = parse_log_line(line)
            if not event:
                continue

            turn_key = extract_turn_key(event)
            if event.component == 'StreamingInputEnv' and event.event_type == 'thread_start' and turn_key is not None:
                active_streaming_turn = turn_key
            # StreamingInputEnv lines often omit stage/side after thread_start.
            # Reuse the active listener context so ASR/tree events are still attributed.
            if turn_key is None and event.component == 'StreamingInputEnv':
                if event.event_type == 'thread_start':
                    st = event.data.get('stage')
                    sd = event.data.get('statement_side') or event.data.get('side')
                    if st and sd:
                        active_streaming_turn = (st, sd)
                        turn_key = active_streaming_turn
                elif active_streaming_turn is not None:
                    turn_key = active_streaming_turn
                    if event.event_type == 'thread_end':
                        active_streaming_turn = None
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

            elif event.component == 'BatchListener':
                if event.event_type == 'analyze_start':
                    turn.batch_analyze_start = event.timestamp
                elif event.event_type == 'analyze_end':
                    turn.batch_analyze_end = event.timestamp

            elif event.component == 'StreamingInputEnv':
                if event.event_type == 'thread_start':
                    turn.listener_thread_start = event.timestamp
                elif event.event_type == 'thread_end':
                    turn.listener_thread_end = event.timestamp
                elif event.event_type == 'asr_start':
                    audio_range = _parse_audio_range(event.data.get('audio_range', ''))
                    if audio_range is None:
                        continue
                    audio_start, audio_end = audio_range
                    asr_starts[(stage, side, audio_start, audio_end)] = event.timestamp
                elif event.event_type == 'asr_end':
                    audio_range = _parse_audio_range(event.data.get('audio_range', ''))
                    if audio_range is None:
                        continue
                    audio_start, audio_end = audio_range
                    asr_key = (stage, side, audio_start, audio_end)
                    if asr_key in asr_starts:
                        turn.asr_operations.append(ASRMetrics(
                            audio_start=audio_start,
                            audio_end=audio_end,
                            asr_start_time=asr_starts[asr_key],
                            asr_end_time=event.timestamp,
                            text_len=int(event.data.get('text_len', 0))
                        ))
                elif event.event_type == 'tree_update_start':
                    words = int(event.data.get('words', 0))
                    tree_starts[(stage, side, words)] = event.timestamp
                elif event.event_type == 'tree_update_end':
                    words = int(event.data.get('words', 0))
                    tree_key = (stage, side, words)
                    if tree_key in tree_starts:
                        turn.tree_updates.append(TreeUpdateMetrics(
                            start_time=tree_starts[tree_key],
                            end_time=event.timestamp,
                            word_count=words
                        ))

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

    return turns


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

        turn_summary = {
            'stage': stage,
            'side': side,
            'mode': turn.mode,
            'streaming_tts': turn.streaming_tts,
            'streaming_listen': turn.streaming_listen,
            'total_duration': turn.total_duration,
            'playback_duration': turn.playback_duration,
            'speaker_duration': turn.speaker_duration,
            'listener_duration': turn.listener_duration,
            'generation_time': turn.generation_time,
            'speaker_bubble_total': turn.total_speaker_bubble,
            'time_to_first_chunk': turn.time_to_first_chunk,
            'time_between_chunks': turn.time_between_chunks,
            'speaker_bubble_pct': (turn.total_speaker_bubble / turn.playback_duration * 100) if turn.playback_duration else None,
            'listener_bubble': turn.listener_bubble,
            'listener_bubble_pct': (turn.listener_bubble / turn.listener_duration * 100) if turn.listener_duration and turn.listener_bubble else None,
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

        # Speaker bubble definition for reporting:
        # first chunk generation time + time between chunks.
        if turn_summary.get("first_chunk_gen_time_s") is not None:
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
    print("  Speaker duration:    speaker_thread_end - speaker_thread_start")
    print("  Listener duration:   listener_thread_end - listener_thread_start")
    print("  Generation time:     generation_end - generation_start")
    print("  --- Bubble Analysis ---")
    print("  Speaker bubble:      first chunk gen time + Time Between Chunks")
    print("  Listener bubble:     listener_thread_end - playback_end")
    print("  --- Efficiency Metrics ---")
    print("  Time to First Chunk: wait time for chunk_1")
    print("  Time Between Chunks: sum(waits for chunk_2+)")
    print("  True overlap:        playback_duration - speaker_bubble")
    print("  Overlap efficiency:  true_overlap / (playback_duration + listener_bubble)")
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
        print(f"  Streaming TTS:       {turn.get('streaming_tts', 'N/A')}")
        print(f"  Streaming Listen:    {turn.get('streaming_listen', 'N/A')}")

        print(f"\n--- Timing Overview ---")
        print(f"  Total duration:      {turn['total_duration']:.2f}s" if turn['total_duration'] else "  Total duration:      N/A")
        print(f"  Audio duration:      {turn['audio_duration']:.2f}s (ideal={ideal_audio_duration}s, gap={turn['audio_duration'] - ideal_audio_duration:.2f}s)" if turn.get('audio_duration') is not None else "  Audio duration:      N/A")
        print(f"  Playback duration:   {turn['playback_duration']:.2f}s" if turn['playback_duration'] else "  Playback duration:   N/A")
        print(f"  Speaker duration:    {turn['speaker_duration']:.2f}s" if turn['speaker_duration'] else "  Speaker duration:    N/A")
        print(f"  Listener duration:   {turn['listener_duration']:.2f}s" if turn['listener_duration'] else "  Listener duration:   N/A")
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
        if turn['listener_bubble'] is not None:
            lb_pct = turn.get('listener_bubble_pct')
            if lb_pct is not None:
                print(f"  Listener bubble:     {turn['listener_bubble']:.2f}s ({lb_pct:.1f}% of listener time)")
            else:
                print(f"  Listener bubble:     {turn['listener_bubble']:.2f}s")

        print(f"\n--- Efficiency Metrics ---")
        print(f"  Time to First Chunk: {first_wait:.2f}s" if first_wait is not None else "  Time to First Chunk: N/A")
        print(f"  Time Between Chunks: {inter_wait:.2f}s")
        if turn['true_overlap'] is not None:
            print(f"  True overlap:        {turn['true_overlap']:.2f}s")
        if turn['overlap_efficiency'] is not None:
            print(f"  Overlap efficiency:  {turn['overlap_efficiency']*100:.1f}%")
        if turn['bottleneck']:
            print(f"  Bottleneck:          {turn['bottleneck']}")

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

        if turn.get('batch_analyze_time') is not None:
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
    turns = parse_log_file(log_path)
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
