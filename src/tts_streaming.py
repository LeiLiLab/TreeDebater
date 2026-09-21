"""
Streaming TTS pipeline with adaptive refinement.

Splits debate text into chunks, adaptively refines each chunk's length to hit its proportional time budget, and generates TTS candidates in parallel.

Key features vs the serial pipeline in tts.py:
  - Chunk-based processing: text is split by paragraphs, each chunk gets a proportional share of the total time budget.
  - Adaptive refinement: FastSpeech estimates duration; if off-target, an LLM rewrites the chunk to a target word count.  Multiple TTS candidates are submitted in parallel and the closest-to-target is picked.
  - Streaming overlap: while chunk N's audio plays, chunk N+1 is being refined and TTS-generated (time_budget for chunk N+1 = audio duration of chunk N).
  - No information loss: instead of trimming sentences at the end, text is rewritten to fit the budget.
"""

import concurrent.futures
import csv
import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mutagen.mp3 import MP3
from openai import OpenAI
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from utils.tool import remove_citation, remove_subtitles
from utils.time_estimator import LengthEstimator
from utils.fs_wrapper import get_shared_wrapper

# -------- config --------
TOLERANCE_RATIO = 0.10          # tolerance = target_s * this ratio (both sides)
TOLERANCE_RATIO_UPPER = 0.05    # last chunk upper tolerance (tighter)
MIN_TOLERANCE_S = 1.0           # floor so very short chunks aren't impossible to hit
MAX_REFINEMENTS = 10
MAX_PARALLEL_TTS = 8            # max concurrent background TTS threads per chunk
MIN_CHUNK_WORDS = 30            # chunks shorter than this (word count) are merged into the next one; the last chunk uses the same threshold
EARLY_CUT_RATIO = 1.25          # fs_est/target_s threshold to trigger early-cut
RATIO_PRESTART_THRESHOLD = 2.0  # if chunk[i+2].chars / chunk[i+1].chars >= this, pre-start chunk i+2 one chunk earlier
ABS_PRESTART_CHARS = 1000       # also pre-start when chunk[i+2] is absolutely large (>= this many chars), regardless of ratio
SPEED_ADJUST_MIN = 0.85         # TTS speed clamp lower bound
SPEED_ADJUST_MAX = 1.15         # TTS speed clamp upper bound
REFINE_DEADLINE_MARGIN_S = 2.0  # stop refining this much before the previous chunk finishes playing (decode/write/bridge/poll latency)
SPEED_ADJUST_MIN_SLACK_S = 4.0  # only re-synthesize with adjusted speed when at least this much playback slack remains
MAX_CHUNK_CHARS = 900           # paragraphs longer than this are split at sentence boundaries into ~equal pieces
TARGET_CHUNK_S = 40.0           # desired audio length per chunk when the text has too few paragraphs to stream
MIN_STREAM_CHUNKS = 3           # texts with fewer paragraphs than this are re-chunked by sentence packing
MAX_STREAM_CHUNKS = 8
SEAM_HEAD_MS = 60               # leading silence kept at the start of every chunk
SEAM_TAIL_MS = 250              # trailing silence normalized to exactly this at the end of every chunk
SEAM_FADE_MS = 10               # tiny fade in/out so the cut never clicks
# Online calibration of FastSpeech duration estimates against the real TTS voice.
# FastSpeech (LJSpeech) and tts-1 "echo" differ in speaking rate by a text-dependent
# few percent; the static "*1.11-7" fit leaves ~5.5% mean abs error (p90 11.6%),
# which is the main reason a refined chunk still lands outside its +-10% tolerance.
# Each chunk gives one (estimate, actual) observation; an EMA of the ratio is
# applied to subsequent estimates.
# OFF by default: an offline replay on 206 chunks (runs 130-138) showed the
# residual is text-dependent noise with ~0 lag-1 autocorrelation, so chasing it
# with an EMA slightly *increases* error. Kept as an ablation (TTS_STREAM_CALIB=1).
CALIBRATION_ENABLED = os.environ.get("TTS_STREAM_CALIB", "0") == "1"
CALIB_EMA_ALPHA = 0.5
CALIB_MIN, CALIB_MAX = 0.80, 1.25


# -------- dataclasses --------
@dataclass
class ChunkProfile:
    chunk_idx: int
    chunk_chars: int
    chunk_words: int
    target_s: float
    n_ref_used: int
    target_reached: bool
    timed_out: bool
    n_candidates_submitted: int
    n_candidates_done: int
    used_candidate_iter: int
    fs_estimated_s: float
    refine_total_s: float
    time_budget_s: float
    overrun_s: float
    total_elapsed_s: float
    tts_api_s: float
    mp3_parse_s: float
    audio_seconds: float
    chunk_total_s: float
    tolerance_s: float
    tol_upper_s: float
    iter_llm_times_s: str       # JSON list — combined across both workers (legacy)
    iter_fs_times_s: str        # JSON list — combined across both workers (legacy)
    iter_tts_times_s: str       # JSON list — combined across both workers (legacy)
    prep_start_lead_s: float = 0.0   # >0 only when a pre-started result was adopted (last-chunk or ratio-prestart); how much earlier than events[i-1].play_start the prep actually began
    prestart_kind: str = ""          # "" if standard refine; "ratio" if adopted from ratio-prestart; "last" if adopted from last-chunk pre-start
    # Per-worker timings. Each worker has its own ordered fs/llm/tts streams that
    # the visualizer uses to draw two parallel lanes. Empty when that worker did not run.
    prestart_llm_times_s: str = "[]"
    prestart_fs_times_s: str = "[]"
    prestart_tts_times_s: str = "[]"
    normal_llm_times_s: str = "[]"
    normal_fs_times_s: str = "[]"
    normal_tts_times_s: str = "[]"
    chosen_worker_label: str = ""    # "" / "prestart" / "normal" (chunk 0 has no worker)
    chosen_intra_iter: int = 0       # iteration within the chosen worker (0 = raw text)


@dataclass
class RoundProfile:
    n_chunks: int
    total_budget_s: float
    tolerance_ratio: float
    round_total_s: float
    refine_total_s: float
    tts_api_total_s: float
    mp3_parse_total_s: float
    audio_seconds_total: float
    overrun_total_s: float
    budget_remaining_s: float


# -------- helpers --------
def _now():
    return time.perf_counter()


def _in_range(est: float, target_s: float, tol_s: float, tol_upper_s: float) -> bool:
    return (est - target_s) <= tol_upper_s and (target_s - est) <= tol_s


class _DurationCalibrator:
    """Process-wide EMA of actual_tts_seconds / fastspeech_estimate_seconds."""

    def __init__(self) -> None:
        self.ratio = 1.0
        self.n = 0
        self._lock = threading.Lock()

    def observe(self, est_s: float, actual_s: float) -> None:
        if not CALIBRATION_ENABLED or est_s <= 0 or actual_s <= 0:
            return
        r = actual_s / est_s
        with self._lock:
            self.ratio = r if self.n == 0 else (1 - CALIB_EMA_ALPHA) * self.ratio + CALIB_EMA_ALPHA * r
            self.ratio = max(CALIB_MIN, min(CALIB_MAX, self.ratio))
            self.n += 1

    def apply(self, est_s: float) -> float:
        if not CALIBRATION_ENABLED:
            return est_s
        with self._lock:
            return est_s * self.ratio


_CALIB = _DurationCalibrator()


def _fastspeech_estimate(text: str) -> float:
    # Shared, lazily-loaded model: constructing a wrapper per call reloads the
    # checkpoint (+ vocoder) from disk every time (~5 s and GBs of GPU memory).
    wrapper = get_shared_wrapper(batch_size=2)
    lengths = wrapper.query_time(text)
    length = lengths[0] if isinstance(lengths, list) else float(lengths)
    length = length * 1.11 - 7 if length > 100 else length
    return float(length)


def _revise_to_n_words(client, text: str, n_words: int, prev_texts: List[str], next_chunk_text: str = "") -> str:
    context_block = ""
    context_word_count = 0
    if prev_texts:
        joined = "\n\n".join(prev_texts)
        context_word_count = LengthEstimator.count_words(joined)
        context_block = (
            f"Here is the debate speech text that has already been delivered "
            f"(spoken aloud before this paragraph):\n\n"
            f"{joined}\n\n"
            f"---\n\n"
        )

    context_note = (
        f" The preceding speech context contains approximately {context_word_count} words."
        if context_word_count > 0
        else ""
    )

    next_block = ""
    if next_chunk_text:
        next_block = (
            f"\n\nThe paragraph you rewrite will be followed immediately by this next paragraph "
            f"(do NOT rewrite it, just ensure your output flows naturally into it):\n\n"
            f"{next_chunk_text[:2000]}"
        )

    resp = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are helping refine a paragraph from a competitive debate speech. "
                    "The speech is delivered orally; every word will be read aloud by a text-to-speech system. "
                    "Rewrite ONLY the paragraph provided by the user. "
                    "Preserve the argument, logical flow, and debate rhetoric. "
                    "Do NOT add new arguments or repeat points already made in the preceding text. "
                    "Ensure the rewritten paragraph connects smoothly with what comes before and after it. "
                    "Output only the rewritten paragraph, no preamble."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context_block}"
                    f"Rewrite the following debate paragraph to be approximately {n_words} words."
                    f"{context_note} "
                    f"Keep the debating style and the core argument intact.\n\n"
                    f"{text[:8000]}"
                    f"{next_block}"
                ),
            },
        ],
        max_completion_tokens=4096,
        # A length-constrained rewrite needs no chain-of-thought: with the default
        # effort gpt-5-mini spends ~700-1600 reasoning tokens (~10 s) per call;
        # "minimal" gives the same word-count accuracy in ~2 s.
        reasoning_effort="minimal",
    )
    return (resp.choices[0].message.content or "").strip()


def _query_time_profiled(client, content: str, voice: str = "echo", speed: float = 1.0) -> Dict[str, Any]:
    t0 = _now()
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=content[:4096],
        response_format="mp3",
        speed=speed,
    )
    t1 = _now()

    mp3_bytes = response.content
    audio_bytes = BytesIO(mp3_bytes)

    t2 = _now()
    audio_seconds = MP3(audio_bytes).info.length
    t3 = _now()

    return {
        "audio_seconds": float(audio_seconds),
        "tts_api_s": t1 - t0,
        "mp3_parse_s": t3 - t2,
        "mp3_bytes": mp3_bytes,
    }


def _tts_with_retry(client, content: str, voice: str = "echo", speed: float = 1.0, max_attempts: int = 5) -> Dict[str, Any]:
    for attempt in range(max_attempts):
        try:
            return _query_time_profiled(client, content, voice=voice, speed=speed)
        except Exception:
            if attempt == max_attempts - 1:
                raise
            time.sleep(1.5 ** attempt)
    raise RuntimeError("unreachable")


# -------- TTS candidate tracking --------
@dataclass
class _TtsCandidate:
    iteration: int          # global index in shared candidate pool
    text: str
    fs_estimated_s: float   # calibrated estimate used for the in-range decision
    future: Any             # concurrent.futures.Future -> Dict from _query_time_profiled
    worker_label: str = ""  # "prestart" / "normal" — which worker produced this
    intra_iter: int = 0     # iteration index within the producing worker (0 = raw, k = k-th refine)
    fs_raw_s: float = 0.0   # uncalibrated FastSpeech estimate (for calibration updates)


def _pick_best_completed(
    candidates: List[_TtsCandidate],
    target_s: float,
) -> Tuple[_TtsCandidate, Dict]:
    def _collect_done() -> List[Tuple[_TtsCandidate, Dict]]:
        out = []
        for c in candidates:
            if c.future.done():
                try:
                    out.append((c, c.future.result()))
                except Exception:
                    pass
        return out

    done = _collect_done()

    if not done:
        concurrent.futures.wait(
            [c.future for c in candidates],
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        done = _collect_done()

    if not done:
        concurrent.futures.wait([c.future for c in candidates])
        done = _collect_done()

    if not done:
        raise RuntimeError("All parallel TTS candidates failed.")

    return min(done, key=lambda x: abs(x[1]["audio_seconds"] - target_s))


# -------- shared refine context + worker (used by both prestart and normal refine) --------
class _ChunkRefineContext:
    """
    Holds the shared state for refining ONE chunk. Multiple workers (a prestart
    worker, a normal-refine worker) can run concurrently against the same context,
    contributing candidates to a shared pool until either:
      - any candidate's fs_estimate hits target -> first worker to confirm sets done_event
      - main loop's deadline (= prev_audio_s) elapses -> external code stops everything
    """
    def __init__(
        self,
        client,
        original_text: str,
        target_s: float,
        tol_s: float,
        tol_upper_s: float,
        prev_texts: List[str],
        next_chunk_text: str,
        voice: str,
        max_ref: int,
        kickoff_iter: int,
        kickoff_kind: str,         # "" | "ratio" | "last"
    ):
        self.client = client
        self.original_text = original_text

        self._target_s = target_s
        self._tol_s = tol_s
        self._tol_upper_s = tol_upper_s
        self._target_lock = threading.Lock()

        self.candidates: List[_TtsCandidate] = []
        self.candidates_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self._adopt_lock = threading.Lock()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_TTS)

        self.prev_texts = list(prev_texts)
        self.next_chunk_text = next_chunk_text
        self.voice = voice
        self.max_ref = max_ref

        self.kickoff_iter = kickoff_iter
        self.kickoff_kind = kickoff_kind

        # per-worker stats: label -> {"n_ref", "llm_times", "fs_times"}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._stats_lock = threading.Lock()

        self.chosen_cand: Optional[_TtsCandidate] = None
        self.chosen_tts_out: Optional[Dict[str, Any]] = None

        self.t_start_wall = _now()
        self.workers: List[threading.Thread] = []

    def get_target(self) -> Tuple[float, float, float]:
        with self._target_lock:
            return self._target_s, self._tol_s, self._tol_upper_s

    def update_target(self, target_s: float, tol_s: float, tol_upper_s: float) -> None:
        with self._target_lock:
            self._target_s = target_s
            self._tol_s = tol_s
            self._tol_upper_s = tol_upper_s

    def _stats_for(self, label: str) -> Dict[str, Any]:
        with self._stats_lock:
            if label not in self._stats:
                self._stats[label] = {"n_ref": 0, "llm_times": [], "fs_times": []}
            return self._stats[label]

    def add_fs_time(self, label: str, t: float) -> None:
        with self._stats_lock:
            self._stats.setdefault(label, {"n_ref": 0, "llm_times": [], "fs_times": []})
            self._stats[label]["fs_times"].append(t)

    def add_llm_time(self, label: str, t: float) -> None:
        with self._stats_lock:
            self._stats.setdefault(label, {"n_ref": 0, "llm_times": [], "fs_times": []})
            self._stats[label]["llm_times"].append(t)
            self._stats[label]["n_ref"] += 1

    def aggregate_stats(self) -> Tuple[int, List[float], List[float]]:
        with self._stats_lock:
            n_ref_total = sum(s["n_ref"] for s in self._stats.values())
            llm_times: List[float] = []
            fs_times: List[float] = []
            for s in self._stats.values():
                llm_times.extend(s["llm_times"])
                fs_times.extend(s["fs_times"])
            return n_ref_total, llm_times, fs_times

    def add_candidate(self, text: str, est: float, label: str, intra_iter: int, est_raw: float = 0.0) -> _TtsCandidate:
        with self.candidates_lock:
            iteration = len(self.candidates)
            cand = _TtsCandidate(
                iteration=iteration,
                text=text,
                fs_estimated_s=est,
                future=self.executor.submit(_tts_with_retry, self.client, text, self.voice),
                worker_label=label,
                intra_iter=intra_iter,
                fs_raw_s=est_raw,
            )
            self.candidates.append(cand)
        return cand

    def try_adopt(self, cand: _TtsCandidate, tts_out: Dict[str, Any]) -> bool:
        with self._adopt_lock:
            if self.done_event.is_set():
                return False
            self.chosen_cand = cand
            self.chosen_tts_out = tts_out
            self.done_event.set()
            self.stop_event.set()
            return True


def _refine_worker(ctx: _ChunkRefineContext, label: str) -> None:
    """
    Independent worker that drives one branch of refinement against ctx.
    Reads target/tolerance from ctx (which may be updated externally) at the
    start of each iteration. Pushes candidates into the shared pool.
    Sets ctx.done_event via try_adopt() when its candidate hits target.
    """
    cur = ctx.original_text

    if ctx.stop_event.is_set():
        return

    # ---- step 0: fs estimate raw text + submit raw TTS candidate ----
    t = _now()
    try:
        est_raw = _fastspeech_estimate(cur)
    except Exception:
        return
    est = _CALIB.apply(est_raw)
    ctx.add_fs_time(label, _now() - t)

    if ctx.stop_event.is_set():
        return

    target_s, tol_s, tol_upper_s = ctx.get_target()
    cand = ctx.add_candidate(cur, est, label, intra_iter=0, est_raw=est_raw)

    if _in_range(est, target_s, tol_s, tol_upper_s):
        try:
            tts_out = cand.future.result()
            if ctx.try_adopt(cand, tts_out):
                return
        except Exception:
            pass

    # ---- LLM refinement loop ----
    n_ref_local = 0
    while not ctx.stop_event.is_set() and n_ref_local < ctx.max_ref:
        target_s, tol_s, tol_upper_s = ctx.get_target()
        if target_s <= 0:
            break

        cw = LengthEstimator.count_words(cur)
        tw = max(10, round(cw * target_s / max(est, 1.0)))

        t = _now()
        try:
            cur = _revise_to_n_words(ctx.client, cur, tw, ctx.prev_texts, ctx.next_chunk_text)
        except Exception:
            break
        ctx.add_llm_time(label, _now() - t)
        n_ref_local += 1

        if ctx.stop_event.is_set():
            break

        t = _now()
        try:
            est_raw = _fastspeech_estimate(cur)
        except Exception:
            break
        est = _CALIB.apply(est_raw)
        ctx.add_fs_time(label, _now() - t)

        if ctx.stop_event.is_set():
            break

        cand = ctx.add_candidate(cur, est, label, intra_iter=n_ref_local, est_raw=est_raw)

        target_s, tol_s, tol_upper_s = ctx.get_target()
        if _in_range(est, target_s, tol_s, tol_upper_s):
            try:
                tts_out = cand.future.result()
                if ctx.try_adopt(cand, tts_out):
                    return
            except Exception:
                pass


# -------- chunk utilities --------
def _normalize_seam_silence(seg: AudioSegment) -> Tuple[AudioSegment, bool]:
    """Trim/pad head and tail silence so every chunk boundary sounds the same.

    tts-1 output carries 0-150 ms of leading and 100-1200 ms of trailing silence;
    concatenated chunk-by-chunk that yields pauses of 0.2-1.3 s at the seams.
    Returns (segment, changed).
    """
    from pydub.silence import detect_leading_silence

    if len(seg) < 1000:
        return seg, False
    lead = detect_leading_silence(seg, silence_threshold=-40, chunk_size=5)
    trail = detect_leading_silence(seg.reverse(), silence_threshold=-40, chunk_size=5)
    if len(seg) - lead - trail < 500:   # no speech detected (e.g. silent fallback) -> leave it
        return seg, False
    start = max(0, lead - SEAM_HEAD_MS)
    end = len(seg) - trail
    core = seg[start:end]
    out = core.fade_in(SEAM_FADE_MS).fade_out(SEAM_FADE_MS) + AudioSegment.silent(duration=SEAM_TAIL_MS, frame_rate=seg.frame_rate)
    return out, True


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on '.', '!', '?' boundaries."""
    import re
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def _early_cut_chunk(
    text: str,
    target_s: float,
    early_cut_ratio: float = EARLY_CUT_RATIO,
) -> Tuple[str, str]:
    """
    Split text into (head, tail) where head's FS estimate ≈ target_s.
    Returns (head, tail); tail may be empty if the whole text fits.
    Only called when fs_estimate(text) / target_s > early_cut_ratio.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text, ""

    head_sentences: List[str] = []
    for sent in sentences:
        candidate = " ".join(head_sentences + [sent])
        est = _fastspeech_estimate(candidate)
        if est > target_s and head_sentences:
            break
        head_sentences.append(sent)

    if not head_sentences:
        head_sentences = [sentences[0]]

    head = " ".join(head_sentences)
    tail_sentences = sentences[len(head_sentences):]
    tail = " ".join(tail_sentences)
    return head, tail


def _merge_short_chunks(
    segments: List[str],
    min_words: int = MIN_CHUNK_WORDS,
) -> List[str]:
    result = list(segments)
    i = 0
    while i < len(result):
        if LengthEstimator.count_words(result[i]) < min_words and i + 1 < len(result):
            result[i + 1] = result[i] + " " + result[i + 1]
            result.pop(i)
        else:
            i += 1
    if len(result) >= 2 and LengthEstimator.count_words(result[-1]) < min_words:
        result[-2] = result[-2] + " " + result[-1]
        result.pop()
    return result


def split_by_paragraphs(text: str) -> List[str]:
    """Split text on double newlines into non-empty paragraphs."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text.strip()] if text.strip() else []


def _pack_sentences(text: str, target_chars: int) -> List[str]:
    """Greedily pack whole sentences into pieces of roughly ``target_chars`` characters.

    Never cuts inside a sentence; a single sentence longer than the target becomes
    its own piece. Pieces are balanced so the last one is not a tiny remainder.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    total = sum(len(x) + 1 for x in sentences)
    n_pieces = max(1, round(total / max(target_chars, 1)))
    per_piece = total / n_pieces
    pieces: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for sent in sentences:
        if cur and cur_len + len(sent) + 1 > per_piece * 1.15 and len(pieces) < n_pieces - 1:
            pieces.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += len(sent) + 1
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_into_chunks(text: str, total_budget_s: float) -> List[str]:
    """Split a speech into streamable chunks.

    Paragraphs are the preferred unit (they are natural rhetorical units and the
    refinement prompt talks about "paragraphs"). Two failure modes of pure
    paragraph splitting are handled here:
      1. The model emitted (almost) no paragraph breaks -> a single huge chunk,
         which disables streaming and refinement entirely. In that case the text
         is re-chunked by packing sentences to ~TARGET_CHUNK_S of audio each.
      2. One paragraph is much longer than the rest (> MAX_CHUNK_CHARS) -> it is
         split at sentence boundaries into a few balanced pieces.
    """
    paras = split_by_paragraphs(text)
    if not paras:
        return []
    total_chars = sum(len(p) for p in paras)

    if len(paras) < MIN_STREAM_CHUNKS:
        n_target = int(round(total_budget_s / TARGET_CHUNK_S)) if total_budget_s > 0 else MIN_STREAM_CHUNKS
        n_target = max(MIN_STREAM_CHUNKS, min(MAX_STREAM_CHUNKS, n_target))
        target_chars = max(200, total_chars // n_target)
        out: List[str] = []
        for p in paras:
            out.extend(_pack_sentences(p, target_chars))
        return out or paras

    out = []
    for p in paras:
        if len(p) > MAX_CHUNK_CHARS:
            n_pieces = -(-len(p) // MAX_CHUNK_CHARS)  # ceil
            out.extend(_pack_sentences(p, max(200, len(p) // n_pieces)))
        else:
            out.append(p)
    return out


# -------- pipeline --------
def run_pipeline(
    client,
    segments_list: List[str],
    total_budget_s: float,
    tolerance_ratio: float = TOLERANCE_RATIO,
    voice: str = "echo",
    out_dir: Optional[Path] = None,
    enable_early_cut: bool = False,
    early_cut_ratio: float = EARLY_CUT_RATIO,
) -> Tuple[List[ChunkProfile], RoundProfile, bytes, List[str]]:
    """
    Run the streaming TTS pipeline on a list of text segments.

    Returns:
        (chunk_profiles, round_profile, combined_mp3_bytes, final_texts)
    """
    round_t0 = _now()
    chunk_profiles: List[ChunkProfile] = []

    refine_total_acc = 0.0
    tts_api_total = 0.0
    mp3_parse_total = 0.0
    audio_total = 0.0
    overrun_total = 0.0

    segments_list = list(_merge_short_chunks(segments_list))
    n_chunks = len(segments_list)
    audio_budget_remaining = total_budget_s
    total_chars_initial = sum(len(c) for c in segments_list)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    final_texts: List[str] = []
    combined_audio = AudioSegment.silent(duration=0)
    all_mp3_bytes: List[bytes] = []

    prev_audio_s: Optional[float] = None

    # Pre-start contexts indexed by target chunk idx. A context is created when
    # we kick off a prestart worker for that chunk (ratio or last-chunk variant).
    # When the main loop reaches that chunk, it pops the context and adds a
    # normal refine worker that shares the same candidate pool.
    _chunk_contexts: Dict[int, _ChunkRefineContext] = {}

    def _kickoff_ratio_prestart(iter_i: int) -> None:
        """At the start of iter iter_i, maybe kick off prestart for c[i+2].

        Triggers if EITHER:
          - chunk[i+2].chars / chunk[i+1].chars >= RATIO_PRESTART_THRESHOLD, OR
          - chunk[i+2].chars >= ABS_PRESTART_CHARS (absolutely long chunk)
        """
        target_idx = iter_i + 2
        if target_idx >= len(segments_list) - 1:   # would be the last chunk → handled by last-chunk kickoff
            return
        if target_idx in _chunk_contexts:
            return
        next_chars = len(segments_list[iter_i + 1])
        target_chars = len(segments_list[target_idx])
        ratio = (target_chars / next_chars) if next_chars > 0 else 0.0
        ratio_trigger = ratio >= RATIO_PRESTART_THRESHOLD
        abs_trigger = target_chars >= ABS_PRESTART_CHARS
        if not (ratio_trigger or abs_trigger):
            return
        target_text = segments_list[target_idx]
        tgt_s_est = total_budget_s * target_chars / max(total_chars_initial, 1)
        tol_est = max(MIN_TOLERANCE_S, tgt_s_est * tolerance_ratio)
        tol_upper_est = max(MIN_TOLERANCE_S, tgt_s_est * TOLERANCE_RATIO_UPPER)
        ctx = _ChunkRefineContext(
            client=client,
            original_text=target_text,
            target_s=tgt_s_est,
            tol_s=tol_est,
            tol_upper_s=tol_upper_est,
            prev_texts=list(final_texts),
            next_chunk_text="",
            voice=voice,
            max_ref=MAX_REFINEMENTS,
            kickoff_iter=iter_i,
            kickoff_kind="ratio",
        )
        _chunk_contexts[target_idx] = ctx
        th = threading.Thread(target=_refine_worker, args=(ctx, "prestart"), daemon=True)
        ctx.workers.append(th)
        th.start()
        triggers = []
        if ratio_trigger:
            triggers.append(f"ratio={ratio:.2f}x")
        if abs_trigger:
            triggers.append(f"abs={target_chars}c>={ABS_PRESTART_CHARS}")
        print(
            f"  [ratio-prestart] chunk {target_idx} kicked off at start of iter {iter_i}, "
            f"trigger=[{', '.join(triggers)}], target_est={tgt_s_est:.1f}s"
        )

    def _kickoff_last_chunk_prestart(iter_i: int) -> None:
        """At the start of iter iter_i, if iter_i == n-3, kick off prestart for the last chunk."""
        n_now = len(segments_list)
        if n_now < 3:
            return
        if iter_i != n_now - 3:
            return
        last_idx = n_now - 1
        if last_idx in _chunk_contexts:
            return
        last_text = segments_list[last_idx]
        last_chars = len(last_text)
        # Use *initial* allocation for consistency with ratio-prestart; main loop
        # will push the up-to-date target later via update_target().
        tgt_s_est = total_budget_s * last_chars / max(total_chars_initial, 1)
        tol_est = max(MIN_TOLERANCE_S, tgt_s_est * tolerance_ratio)
        tol_upper_est = max(MIN_TOLERANCE_S, tgt_s_est * TOLERANCE_RATIO_UPPER)
        ctx = _ChunkRefineContext(
            client=client,
            original_text=last_text,
            target_s=tgt_s_est,
            tol_s=tol_est,
            tol_upper_s=tol_upper_est,
            prev_texts=list(final_texts),
            next_chunk_text="",
            voice=voice,
            max_ref=MAX_REFINEMENTS,
            kickoff_iter=iter_i,
            kickoff_kind="last",
        )
        _chunk_contexts[last_idx] = ctx
        th = threading.Thread(target=_refine_worker, args=(ctx, "prestart"), daemon=True)
        ctx.workers.append(th)
        th.start()
        print(
            f"  [last-prestart] chunk {last_idx} kicked off at start of iter {iter_i}, "
            f"target_est={tgt_s_est:.1f}s"
        )

    i = 0
    while i < len(segments_list):
        chunk = segments_list[i]
        n_chunks = len(segments_list)  # may grow due to early-cut

        chunk_t0 = _now()
        chunk_words = LengthEstimator.count_words(chunk)
        chunk_chars = len(chunk)

        remaining_chars_total = sum(len(c) for c in segments_list[i:])
        target_s = audio_budget_remaining * (chunk_chars / remaining_chars_total)

        # ---- early-cut: if chunk is too long relative to budget, split it now ----
        if enable_early_cut and i > 0:
            fs_pre = _fastspeech_estimate(chunk)
            if fs_pre / target_s > early_cut_ratio:
                head, tail = _early_cut_chunk(chunk, target_s, early_cut_ratio)
                if tail:
                    segments_list[i] = head
                    segments_list.insert(i + 1, tail)
                    chunk = head
                    n_chunks = len(segments_list)
                    chunk_chars = len(chunk)
                    chunk_words = LengthEstimator.count_words(chunk)
                    remaining_chars_total = sum(len(c) for c in segments_list[i:])
                    target_s = audio_budget_remaining * (chunk_chars / remaining_chars_total)
                    print(f"  chunk {i:03d} | early-cut: fs_pre={fs_pre:.1f}s > {early_cut_ratio}x target={target_s:.1f}s → split into head({len(head)}c)+tail({len(tail)}c)")

        tol_s = max(MIN_TOLERANCE_S, target_s * tolerance_ratio)
        remaining_chunks = n_chunks - i
        tol_upper_s = (
            max(MIN_TOLERANCE_S, target_s * TOLERANCE_RATIO_UPPER)
            if remaining_chunks == 1
            else tol_s
        )

        max_ref = 3 if i < n_chunks // 2 else MAX_REFINEMENTS
        next_chunk_text = segments_list[i + 1] if i + 1 < len(segments_list) else ""

        seg = None
        mp3_bytes = b""
        audio_seconds = 0.0
        tts_api_s = 0.0
        mp3_parse_s = 0.0
        chunk_lead_s = 0.0
        chunk_prestart_kind = ""
        # Per-worker times (default empty; chunks 1+ branch overrides)
        prestart_fs_list: List[float] = []
        prestart_llm_list: List[float] = []
        prestart_tts_list: List[float] = []
        normal_fs_list: List[float] = []
        normal_llm_list: List[float] = []
        normal_tts_list: List[float] = []
        chosen_worker_label = ""
        chosen_intra_iter = 0

        # ---- (NEW) at start of every iteration: maybe kick off prestarts ----
        # For iter i, ratio check looks at c[i+2]/c[i+1]; last-chunk fires when i == n-3.
        # Both kickoffs run BEFORE we process the current chunk, so chunk 0's TTS
        # runs in parallel with the prestart for chunk 2 (if ratio triggered).
        _kickoff_ratio_prestart(i)
        _kickoff_last_chunk_prestart(i)

        # ---- chunk 0: no refinement, sequential TTS ----
        if i == 0:
            time_budget_s = target_s
            refined = chunk
            n_ref_used = 0
            try:
                fs_estimated_s = _fastspeech_estimate(chunk)   # ~10 ms; seeds the calibrator below
            except Exception:
                fs_estimated_s = 0.0
            refine_total_s = 0.0
            in_range = True
            timed_out = False
            n_candidates_submitted = 1
            n_candidates_done = 1
            used_candidate_iter = 0
            iter_llm_times_s = "[]"
            iter_fs_times_s = "[]"
            iter_tts_times_s = "[]"

            for attempt in range(10):
                try:
                    tts_out = _query_time_profiled(client, refined, voice=voice)
                    audio_seconds = float(tts_out["audio_seconds"])
                    tts_api_s = float(tts_out["tts_api_s"])
                    mp3_parse_s = float(tts_out["mp3_parse_s"])
                    mp3_bytes = tts_out["mp3_bytes"]
                    seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
                    break
                except (Exception, CouldntDecodeError) as e:
                    if attempt == 9:
                        if isinstance(e, CouldntDecodeError):
                            import warnings
                            warnings.warn(f"Chunk {i}: pydub decode failed after 10 attempts: {e}")
                            seg = AudioSegment.silent(duration=int(audio_seconds * 1000))
                            break
                        raise RuntimeError(f"TTS/decode failed after 10 attempts: {e}") from e
                    time.sleep(2.0 * (attempt + 1))

            iter_tts_times_s = json.dumps([round(tts_api_s, 3)])
            total_elapsed_s = tts_api_s
            overrun_s = tts_api_s
            # chunk 0 is never refined, but record whether it actually hit its target
            in_range = _in_range(audio_seconds, target_s, tol_s, tol_upper_s)
            _CALIB.observe(fs_estimated_s, audio_seconds)

        # ---- chunks 1+: shared candidate pool with prestart + normal workers ----
        else:
            time_budget_s = prev_audio_s

            # Pop existing prestart context (if any), or build a fresh context
            ctx = _chunk_contexts.pop(i, None)
            if ctx is not None:
                # Prestart was running; push the up-to-date target so its next
                # iteration uses real budget instead of the initial estimate.
                ctx.update_target(target_s, tol_s, tol_upper_s)
                # update prev_texts for normal worker via its own field
                ctx.prev_texts = list(final_texts)
                ctx.next_chunk_text = next_chunk_text
                chunk_prestart_kind = ctx.kickoff_kind
            else:
                ctx = _ChunkRefineContext(
                    client=client,
                    original_text=chunk,
                    target_s=target_s,
                    tol_s=tol_s,
                    tol_upper_s=tol_upper_s,
                    prev_texts=list(final_texts),
                    next_chunk_text=next_chunk_text,
                    voice=voice,
                    max_ref=max_ref,
                    kickoff_iter=i,
                    kickoff_kind="",
                )
                chunk_prestart_kind = ""

            # Always start a normal worker for this chunk (in addition to any
            # prestart worker that may already be running on the same context).
            normal_th = threading.Thread(target=_refine_worker, args=(ctx, "normal"), daemon=True)
            ctx.workers.append(normal_th)
            normal_th.start()

            # Wait for ANY worker to find an ok candidate, OR until deadline.
            # The deadline is the previous chunk's playback length minus a margin
            # for everything that still has to happen after selection (decode,
            # file write, bridge copy, playback-side stability poll); without the
            # margin every timed-out chunk arrives a few seconds late and the
            # listener hears a gap.
            ctx.done_event.wait(timeout=max(0.0, time_budget_s - REFINE_DEADLINE_MARGIN_S))
            ctx.stop_event.set()

            total_elapsed_s = _now() - ctx.t_start_wall

            # Pick the chosen candidate
            if ctx.chosen_cand is not None and ctx.chosen_tts_out is not None:
                chosen_cand = ctx.chosen_cand
                tts_out = ctx.chosen_tts_out
                in_range = True
                timed_out = False
            else:
                # Deadline hit before any worker confirmed ok → take best from pool
                with ctx.candidates_lock:
                    snap = list(ctx.candidates)
                if not snap:
                    raise RuntimeError(f"Chunk {i}: no candidates produced")
                target_now, _, _ = ctx.get_target()
                chosen_cand, tts_out = _pick_best_completed(snap, target_now)
                in_range = False
                timed_out = True

            # Speed adjustment if still out of range -- but only when there is
            # enough playback slack left for another TTS round trip; otherwise
            # the extra call itself causes an audible gap.
            target_now, tol_now, tol_upper_now = ctx.get_target()
            audio_s = float(tts_out["audio_seconds"])
            _CALIB.observe(chosen_cand.fs_raw_s, audio_s)
            slack_s = time_budget_s - (_now() - chunk_t0)
            if not _in_range(audio_s, target_now, tol_now, tol_upper_now) and slack_s >= SPEED_ADJUST_MIN_SLACK_S:
                raw_speed = audio_s / target_now if target_now > 0 else 1.0
                clamped = max(SPEED_ADJUST_MIN, min(SPEED_ADJUST_MAX, raw_speed))
                if abs(clamped - 1.0) > 0.01:
                    try:
                        speed_tts_out = _tts_with_retry(client, chosen_cand.text, voice=voice, speed=clamped)
                        if abs(speed_tts_out["audio_seconds"] - target_now) < abs(audio_s - target_now):
                            print(
                                f"  [speed-adjust] chunk {i}: speed={clamped:.2f} "
                                f"audio {audio_s:.1f}s -> {speed_tts_out['audio_seconds']:.1f}s (target {target_now:.1f}s)"
                            )
                            tts_out = speed_tts_out
                            audio_s = float(tts_out["audio_seconds"])
                    except Exception:
                        pass

            # Aggregate stats from all workers on this context
            n_ref_total, all_llm_times, all_fs_times = ctx.aggregate_stats()
            with ctx.candidates_lock:
                snap = list(ctx.candidates)
            iter_tts_times: List[float] = []
            for c in snap:
                if c.future.done():
                    try:
                        iter_tts_times.append(round(c.future.result()["tts_api_s"], 3))
                    except Exception:
                        iter_tts_times.append(-1.0)
                else:
                    iter_tts_times.append(-1.0)

            # Per-worker fs/llm/tts streams (in candidate-submit order within each worker)
            def _worker_tts_in_order(label: str) -> List[float]:
                out: List[float] = []
                for c in snap:
                    if c.worker_label != label:
                        continue
                    if c.future.done():
                        try:
                            out.append(round(c.future.result()["tts_api_s"], 3))
                        except Exception:
                            out.append(-1.0)
                    else:
                        out.append(-1.0)
                return out

            prestart_stats = ctx._stats.get("prestart", {"fs_times": [], "llm_times": []})
            normal_stats = ctx._stats.get("normal", {"fs_times": [], "llm_times": []})
            prestart_fs_list = list(prestart_stats.get("fs_times", []))
            prestart_llm_list = list(prestart_stats.get("llm_times", []))
            prestart_tts_list = _worker_tts_in_order("prestart")
            normal_fs_list = list(normal_stats.get("fs_times", []))
            normal_llm_list = list(normal_stats.get("llm_times", []))
            normal_tts_list = _worker_tts_in_order("normal")

            # Compute lead_s for prestarted chunks (how much earlier than the
            # would-be normal prep_start the worker actually started).
            if chunk_prestart_kind:
                if ctx.kickoff_iter == 0:
                    chunk_lead_s = chunk_profiles[0].tts_api_s + chunk_profiles[0].audio_seconds
                else:
                    # kickoff at start of iter k (k = i-2) → lead = audio_{k-1} + audio_k = audio_{i-3} + audio_{i-2}
                    chunk_lead_s = (
                        chunk_profiles[i - 3].audio_seconds + chunk_profiles[i - 2].audio_seconds
                    )
                # subtract lead from elapsed for reporting parity with the old design
                total_elapsed_s = max(0.0, total_elapsed_s - chunk_lead_s)
                print(
                    f"  [{chunk_prestart_kind}-prestart] adopted for chunk {i}: "
                    f"lead={chunk_lead_s:.1f}s, effective_elapsed={total_elapsed_s:.1f}s"
                )

            refined = chosen_cand.text
            n_ref_used = n_ref_total
            fs_estimated_s = chosen_cand.fs_estimated_s
            refine_total_s = total_elapsed_s
            n_candidates_submitted = len(snap)
            n_candidates_done = sum(1 for t in iter_tts_times if t >= 0)
            used_candidate_iter = chosen_cand.iteration
            iter_llm_times_s = json.dumps([round(t, 3) for t in all_llm_times])
            iter_fs_times_s = json.dumps([round(t, 3) for t in all_fs_times])
            iter_tts_times_s = json.dumps(iter_tts_times)
            chosen_worker_label = chosen_cand.worker_label
            chosen_intra_iter = chosen_cand.intra_iter
            audio_seconds = audio_s
            tts_api_s = float(tts_out["tts_api_s"])
            mp3_parse_s = float(tts_out["mp3_parse_s"])
            mp3_bytes = tts_out["mp3_bytes"]

            overrun_s = max(0.0, total_elapsed_s - time_budget_s)

            # Best-effort cleanup: workers are daemon threads that exit at their
            # next stop_event check. Do NOT join them here -- a worker blocked in
            # an LLM/TTS call would stall the main pipeline by the join timeout.
            ctx.executor.shutdown(wait=False)

            try:
                seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
            except CouldntDecodeError as e:
                import warnings
                warnings.warn(f"Chunk {i}: pydub decode failed: {e}")
                seg = AudioSegment.silent(duration=int(audio_seconds * 1000))

        # ---- seam normalization: uniform head/tail silence on every chunk ----
        seg, seam_changed = _normalize_seam_silence(seg)
        if seam_changed:
            buf = BytesIO()
            seg.export(buf, format="mp3")
            mp3_bytes = buf.getvalue()
            audio_seconds = len(seg) / 1000.0

        # ---- budget tracking ----
        audio_budget_remaining -= audio_seconds + overrun_s
        prev_audio_s = audio_seconds
        final_texts.append(refined)
        all_mp3_bytes.append(mp3_bytes)

        if out_dir is not None:
            (out_dir / f"chunk_{i:03d}.txt").write_text(refined, encoding="utf-8")
            # write atomically so the copy bridge never sees a partial file
            tmp_mp3 = out_dir / f"chunk_{i:03d}.mp3.tmp"
            tmp_mp3.write_bytes(mp3_bytes)
            tmp_mp3.replace(out_dir / f"chunk_{i:03d}.mp3")

        combined_audio += seg

        chunk_t1 = _now()

        refine_total_acc += refine_total_s
        tts_api_total += tts_api_s
        mp3_parse_total += mp3_parse_s
        audio_total += audio_seconds
        overrun_total += overrun_s

        cp = ChunkProfile(
            chunk_idx=i,
            chunk_chars=chunk_chars,
            chunk_words=chunk_words,
            target_s=target_s,
            n_ref_used=n_ref_used,
            target_reached=in_range,
            timed_out=timed_out,
            n_candidates_submitted=n_candidates_submitted,
            n_candidates_done=n_candidates_done,
            used_candidate_iter=used_candidate_iter,
            fs_estimated_s=fs_estimated_s,
            total_elapsed_s=total_elapsed_s,
            refine_total_s=refine_total_s,
            time_budget_s=time_budget_s,
            overrun_s=overrun_s,
            tts_api_s=tts_api_s,
            mp3_parse_s=mp3_parse_s,
            audio_seconds=audio_seconds,
            chunk_total_s=chunk_t1 - chunk_t0,
            tolerance_s=tol_s,
            tol_upper_s=tol_upper_s,
            iter_llm_times_s=iter_llm_times_s,
            iter_fs_times_s=iter_fs_times_s,
            iter_tts_times_s=iter_tts_times_s,
            prep_start_lead_s=chunk_lead_s,
            prestart_kind=chunk_prestart_kind,
            prestart_llm_times_s=json.dumps([round(t, 3) for t in prestart_llm_list]),
            prestart_fs_times_s=json.dumps([round(t, 3) for t in prestart_fs_list]),
            prestart_tts_times_s=json.dumps(prestart_tts_list),
            normal_llm_times_s=json.dumps([round(t, 3) for t in normal_llm_list]),
            normal_fs_times_s=json.dumps([round(t, 3) for t in normal_fs_list]),
            normal_tts_times_s=json.dumps(normal_tts_list),
            chosen_worker_label=chosen_worker_label,
            chosen_intra_iter=chosen_intra_iter,
        )
        chunk_profiles.append(cp)

        # Incrementally append to chunk_profile.csv
        if out_dir is not None:
            chunk_csv = out_dir / "chunk_profile.csv"
            fields = list(asdict(cp).keys())
            write_header = not chunk_csv.exists()
            with chunk_csv.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_header:
                    w.writeheader()
                w.writerow(asdict(cp))

        print(
            f"  chunk {i:03d} | target={target_s:.1f}s | tol=[{-tol_s:.1f},{tol_upper_s:+.1f}] | "
            f"max_ref={max_ref} | n_ref={n_ref_used} | "
            f"fs_est={fs_estimated_s:.1f}s | actual={audio_seconds:.1f}s | "
            f"in_range={in_range} | timed_out={timed_out} | "
            f"cands={n_candidates_submitted}(done={n_candidates_done},used={used_candidate_iter}) | "
            f"overrun={overrun_s:.2f}s | remaining={audio_budget_remaining:.1f}s | calib={_CALIB.ratio:.3f}"
        )

        i += 1

    if out_dir is not None:
        sep = "\n\n" + ("=" * 80) + "\n\n"
        (out_dir / "chunks_final.txt").write_text(sep.join(final_texts), encoding="utf-8")
        combined_audio.export(out_dir / "final.mp3", format="mp3")

    # Export combined mp3 to bytes
    combined_buffer = BytesIO()
    combined_audio.export(combined_buffer, format="mp3")
    combined_mp3_bytes = combined_buffer.getvalue()

    round_t1 = _now()
    round_profile = RoundProfile(
        n_chunks=len(segments_list),
        total_budget_s=total_budget_s,
        tolerance_ratio=tolerance_ratio,
        round_total_s=round_t1 - round_t0,
        refine_total_s=refine_total_acc,
        tts_api_total_s=tts_api_total,
        mp3_parse_total_s=mp3_parse_total,
        audio_seconds_total=audio_total,
        overrun_total_s=overrun_total,
        budget_remaining_s=audio_budget_remaining,
    )
    if out_dir is not None:
        round_csv = out_dir / "round_profile.csv"
        with round_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(round_profile).keys()))
            w.writeheader()
            w.writerow(asdict(round_profile))

    return chunk_profiles, round_profile, combined_mp3_bytes, final_texts


# -------- high-level API --------
def convert_text_to_speech_streaming(
    content: str,
    output_path: str,
    total_budget_s: float,
    voice: str = "echo",
    enable_early_cut: bool = False,
    early_cut_ratio: float = EARLY_CUT_RATIO,
) -> Tuple[str, str, float]:
    """
    Streaming TTS: split content into chunks, adaptively refine each chunk's
    length to hit its proportional time budget, generate TTS in parallel.

    Drop-in replacement for tts.convert_text_to_speech() when streaming mode
    is enabled.

    Args:
        content: Raw debate statement text (may contain citations/subtitles)
        output_path: Path to save the final combined MP3
        total_budget_s: Total time budget in seconds (e.g. 240 for opening)
        voice: TTS voice name (default "echo")

    Returns:
        (text_content, reference, duration) - same signature as
        tts.convert_text_to_speech()
    """
    audio_content, _ = remove_citation(content)
    audio_content = remove_subtitles(audio_content)

    segments = split_into_chunks(audio_content, total_budget_s)

    client = OpenAI()
    output_path = Path(output_path)

    chunk_profiles, round_profile, combined_mp3_bytes, final_texts = run_pipeline(
        client,
        segments,
        total_budget_s=total_budget_s,
        voice=voice,
        out_dir=output_path.parent / f"{output_path.stem}_chunks",
        enable_early_cut=enable_early_cut,
        early_cut_ratio=early_cut_ratio,
    )

    # Save combined audio
    output_path.write_bytes(combined_mp3_bytes)

    # Compute duration from the combined audio
    duration = MP3(BytesIO(combined_mp3_bytes)).info.length

    # Build text_content and reference matching the original API
    text_content, reference = remove_citation(content, keep_main=True)

    print(
        f"  => audio_total={round_profile.audio_seconds_total:.2f}s | "
        f"overrun_total={round_profile.overrun_total_s:.2f}s | "
        f"budget_remaining={round_profile.budget_remaining_s:.2f}s | "
        f"wall_clock={round_profile.round_total_s:.2f}s"
    )

    return text_content, reference, duration
