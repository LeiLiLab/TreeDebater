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
from utils.fs_wrapper import FastSpeechWrapper

# -------- config --------
TOLERANCE_RATIO = 0.10          # tolerance = target_s * this ratio (both sides)
TOLERANCE_RATIO_UPPER = 0.05    # last chunk upper tolerance (tighter)
MIN_TOLERANCE_S = 1.0           # floor so very short chunks aren't impossible to hit
MAX_REFINEMENTS = 10
MAX_PARALLEL_TTS = 8            # max concurrent background TTS threads per chunk
MIN_CHUNK_CHARS = 50            # chunks shorter than this are merged into the next one
MAX_CHUNK_CHARS = 900           # long paragraphs are split into sentence sub-chunks
EARLY_CUT_RATIO = 1.25          # fs_est/target_s threshold to trigger early-cut
SPEED_ADJUST_MIN = 0.85         # TTS speed clamp lower bound
SPEED_ADJUST_MAX = 1.15         # TTS speed clamp upper bound
TARGET_WORDS_PER_SECOND = 2.6   # initial compression heuristic for short-budget rounds


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
    iter_llm_times_s: str       # JSON list
    iter_fs_times_s: str        # JSON list
    iter_tts_times_s: str       # JSON list


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


def _fastspeech_estimate(text: str) -> float:
    wrapper = FastSpeechWrapper(batch_size=2)
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
    iteration: int          # 0 = original chunk text; k = after k-th rewrite
    text: str
    fs_estimated_s: float
    future: Any             # concurrent.futures.Future -> Dict from _query_time_profiled


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


# -------- parallel refinement + TTS (chunks 1+) --------
def _adaptive_refine_parallel(
    client,
    text: str,
    target_s: float,
    time_budget_s: float,
    prev_texts: List[str],
    tolerance_s: float,
    tolerance_upper_s: float,
    voice: str = "echo",
    max_ref: int = MAX_REFINEMENTS,
    next_chunk_text: str = "",
    fast_initial_compress: bool = False,
) -> Dict[str, Any]:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_TTS)
    candidates: List[_TtsCandidate] = []
    candidates_lock = threading.Lock()
    stop_event = threading.Event()
    done_event = threading.Event()

    t_wall_start = _now()
    refine_stats: Dict[str, Any] = {}

    def _refine():
        cur = text
        n_ref = 0
        llm_times: List[float] = []
        fs_times: List[float] = []
        t0 = _now()
        n_ref = 0

        with candidates_lock:
            if not stop_event.is_set():
                candidates.append(_TtsCandidate(
                    iteration=0, text=cur, fs_estimated_s=0.0,
                    future=executor.submit(_tts_with_retry, client, cur, voice),
                ))

        if fast_initial_compress and target_s > 0:
            target_words = max(12, round(target_s * TARGET_WORDS_PER_SECOND))
            _t = _now(); compressed = _revise_to_n_words(client, cur, target_words, prev_texts, next_chunk_text); llm_times.append(_now() - _t)
            n_ref = 1
            _t = _now(); compressed_est = _fastspeech_estimate(compressed); fs_times.append(_now() - _t)
            compressed_ok = _in_range(compressed_est, target_s, tolerance_s, tolerance_upper_s)
            if not stop_event.is_set():
                with candidates_lock:
                    candidates.append(_TtsCandidate(
                        iteration=n_ref, text=compressed, fs_estimated_s=compressed_est,
                        future=executor.submit(_tts_with_retry, client, compressed, voice),
                    ))
            cur = compressed
            est = compressed_est
            ok = compressed_ok
        else:
            _t = _now(); est = _fastspeech_estimate(cur); fs_times.append(_now() - _t)
            ok = _in_range(est, target_s, tolerance_s, tolerance_upper_s)
            with candidates_lock:
                if candidates:
                    candidates[0].fs_estimated_s = est

        while not ok and n_ref < max_ref and not stop_event.is_set() and target_s > 0:
            cw = LengthEstimator.count_words(cur)
            tw = max(10, round(cw * target_s / est))
            _t = _now(); cur = _revise_to_n_words(client, cur, tw, prev_texts, next_chunk_text); llm_times.append(_now() - _t)
            n_ref += 1
            _t = _now(); est = _fastspeech_estimate(cur); fs_times.append(_now() - _t)
            ok = _in_range(est, target_s, tolerance_s, tolerance_upper_s)

            if stop_event.is_set():
                break
            with candidates_lock:
                candidates.append(_TtsCandidate(
                    iteration=n_ref, text=cur, fs_estimated_s=est,
                    future=executor.submit(_tts_with_retry, client, cur, voice),
                ))

        refine_stats.update({
            "n_ref": n_ref, "ok": ok,
            "llm_times": llm_times, "fs_times": fs_times,
            "refine_total_s": _now() - t0,
        })

        if ok and not stop_event.is_set():
            with candidates_lock:
                last = candidates[-1]
            tts_out = last.future.result()
            refine_stats["chosen_cand"] = last
            refine_stats["tts_out"] = tts_out
            done_event.set()

        stop_event.set()

    refine_thread = threading.Thread(target=_refine, daemon=True)
    refine_thread.start()

    done_event.wait(timeout=time_budget_s)
    stop_event.set()

    total_elapsed_s = _now() - t_wall_start
    timed_out = not done_event.is_set()

    if done_event.is_set() and "chosen_cand" in refine_stats:
        chosen_cand = refine_stats["chosen_cand"]
        tts_out = refine_stats["tts_out"]
        used_iter = chosen_cand.iteration
    else:
        while True:
            with candidates_lock:
                snap = list(candidates)
            if snap:
                break
            time.sleep(0.05)
        chosen_cand, tts_out = _pick_best_completed(snap, target_s)
        used_iter = chosen_cand.iteration

    with candidates_lock:
        snap = list(candidates)

    iter_tts_times: List[float] = []
    for c in snap:
        if c.future.done():
            try:
                iter_tts_times.append(round(c.future.result()["tts_api_s"], 3))
            except Exception:
                iter_tts_times.append(-1.0)
        else:
            iter_tts_times.append(-1.0)

    refine_thread.join(timeout=5)
    executor.shutdown(wait=False)

    # ---- speed adjustment: re-TTS if audio is outside tolerance but speed can fix it ----
    audio_s = float(tts_out["audio_seconds"])
    raw_speed = audio_s / target_s if target_s > 0 else 1.0
    speed_used = 1.0
    if not _in_range(audio_s, target_s, tolerance_s, tolerance_upper_s):
        clamped = max(SPEED_ADJUST_MIN, min(SPEED_ADJUST_MAX, raw_speed))
        if abs(clamped - 1.0) > 0.01:
            try:
                speed_tts_out = _tts_with_retry(client, chosen_cand.text, voice=voice, speed=clamped)
                if abs(speed_tts_out["audio_seconds"] - target_s) < abs(audio_s - target_s):
                    tts_out = speed_tts_out
                    audio_s = float(tts_out["audio_seconds"])
                    speed_used = clamped
            except Exception:
                pass

    return {
        "refined_text": chosen_cand.text,
        "n_ref_used": refine_stats.get("n_ref", 0),
        "fs_estimated_s": chosen_cand.fs_estimated_s,
        "refine_total_s": refine_stats.get("refine_total_s", total_elapsed_s),
        "total_elapsed_s": total_elapsed_s,
        "target_reached": refine_stats.get("ok", False),
        "timed_out": timed_out,
        "n_candidates_submitted": len(snap),
        "n_candidates_done": sum(1 for t in iter_tts_times if t >= 0),
        "used_candidate_iter": used_iter,
        "llm_times_s": refine_stats.get("llm_times", []),
        "fs_times_s": refine_stats.get("fs_times", []),
        "iter_tts_times_s": iter_tts_times,
        "speed_used": speed_used,
        "audio_seconds": audio_s,
        "tts_api_s": float(tts_out["tts_api_s"]),
        "mp3_parse_s": float(tts_out["mp3_parse_s"]),
        "mp3_bytes": tts_out["mp3_bytes"],
    }


# -------- chunk utilities --------
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


def _merge_short_chunks(segments: List[str], min_chars: int = MIN_CHUNK_CHARS) -> List[str]:
    result = list(segments)
    i = 0
    while i < len(result):
        if len(result[i]) < min_chars and i + 1 < len(result):
            result[i + 1] = result[i] + " " + result[i + 1]
            result.pop(i)
        else:
            i += 1
    return result


def _split_long_chunks(segments: List[str], max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    result: List[str] = []
    for segment in segments:
        if len(segment) <= max_chars:
            result.append(segment)
            continue

        current: List[str] = []
        current_len = 0
        for sentence in _split_sentences(segment):
            extra_len = len(sentence) + (1 if current else 0)
            if current and current_len + extra_len > max_chars:
                result.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += extra_len
        if current:
            result.append(" ".join(current))

    return result


def split_by_paragraphs(text: str) -> List[str]:
    """Split text on double newlines into non-empty paragraphs."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts if parts else [text.strip()] if text.strip() else []


# -------- pipeline --------
def run_pipeline(
    client,
    segments_list: List[str],
    total_budget_s: float,
    tolerance_ratio: float = TOLERANCE_RATIO,
    voice: str = "echo",
    out_dir: Optional[Path] = None,
    enable_early_cut: bool = True,
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

    segments_list = list(_merge_short_chunks(_split_long_chunks(segments_list)))
    n_chunks = len(segments_list)
    audio_budget_remaining = total_budget_s

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    fast_initial_compress = total_budget_s <= 120

    final_texts: List[str] = []
    combined_audio = AudioSegment.silent(duration=0)
    all_mp3_bytes: List[bytes] = []

    prev_audio_s: Optional[float] = None

    # Pre-start future for the last chunk (filled at chunk n_chunks-2)
    _last_chunk_future: Optional[concurrent.futures.Future] = None
    _last_chunk_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
    _last_chunk_lead_s = 0.0

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

        # ---- chunk 0: optionally race original TTS against a target-sized rewrite ----
        if i == 0:
            time_budget_s = target_s
            if fast_initial_compress:
                ref_out = _adaptive_refine_parallel(
                    client, chunk,
                    target_s=target_s,
                    time_budget_s=time_budget_s,
                    prev_texts=[],
                    tolerance_s=tol_s,
                    tolerance_upper_s=tol_upper_s,
                    voice=voice,
                    max_ref=1,
                    next_chunk_text=next_chunk_text,
                    fast_initial_compress=True,
                )

                refined = ref_out["refined_text"]
                n_ref_used = ref_out["n_ref_used"]
                fs_estimated_s = ref_out["fs_estimated_s"]
                refine_total_s = ref_out["refine_total_s"]
                in_range = ref_out["target_reached"]
                timed_out = ref_out["timed_out"]
                n_candidates_submitted = ref_out["n_candidates_submitted"]
                n_candidates_done = ref_out["n_candidates_done"]
                used_candidate_iter = ref_out["used_candidate_iter"]
                iter_llm_times_s = json.dumps([round(t, 3) for t in ref_out["llm_times_s"]])
                iter_fs_times_s = json.dumps([round(t, 3) for t in ref_out["fs_times_s"]])
                iter_tts_times_s = json.dumps(ref_out["iter_tts_times_s"])
                total_elapsed_s = ref_out["total_elapsed_s"]
                audio_seconds = ref_out["audio_seconds"]
                tts_api_s = ref_out["tts_api_s"]
                mp3_parse_s = ref_out["mp3_parse_s"]
                mp3_bytes = ref_out["mp3_bytes"]
                overrun_s = max(0.0, total_elapsed_s - time_budget_s)

                try:
                    seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
                except CouldntDecodeError as e:
                    import warnings
                    warnings.warn(f"Chunk {i}: pydub decode failed: {e}")
                    seg = AudioSegment.silent(duration=int(audio_seconds * 1000))
            else:
                refined = chunk
                n_ref_used = 0
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

        # ---- chunks 1+: parallel refinement with background TTS candidates ----
        else:
            time_budget_s = prev_audio_s

            is_last = (i == len(segments_list) - 1)
            if is_last and _last_chunk_future is not None:
                try:
                    pre_out = _last_chunk_future.result(timeout=max(0.0, time_budget_s))
                except concurrent.futures.TimeoutError:
                    pre_out = _last_chunk_future.result()
                except Exception:
                    pre_out = None
                if pre_out is not None:
                    pre_out = dict(pre_out)
                    pre_out["total_elapsed_s"] = max(
                        0.0,
                        float(pre_out["total_elapsed_s"]) - _last_chunk_lead_s,
                    )
                    ref_out = pre_out
                    print(
                        f"  [pre-start] adopted pre-start result: "
                        f"effective_wait={pre_out['total_elapsed_s']:.1f}s, "
                        f"lead={_last_chunk_lead_s:.1f}s, target={target_s:.1f}s"
                    )
                else:
                    ref_out = _adaptive_refine_parallel(
                        client, chunk,
                        target_s=target_s,
                        time_budget_s=time_budget_s,
                        prev_texts=final_texts,
                        tolerance_s=tol_s,
                        tolerance_upper_s=tol_upper_s,
                        voice=voice,
                        max_ref=max_ref,
                        next_chunk_text=next_chunk_text,
                        fast_initial_compress=fast_initial_compress,
                    )
                if _last_chunk_executor is not None:
                    _last_chunk_executor.shutdown(wait=False)
                    _last_chunk_executor = None
                _last_chunk_future = None
                _last_chunk_lead_s = 0.0
            else:
                ref_out = _adaptive_refine_parallel(
                    client, chunk,
                    target_s=target_s,
                    time_budget_s=time_budget_s,
                    prev_texts=final_texts,
                    tolerance_s=tol_s,
                    tolerance_upper_s=tol_upper_s,
                    voice=voice,
                    max_ref=max_ref,
                    next_chunk_text=next_chunk_text,
                    fast_initial_compress=fast_initial_compress,
                )

            refined = ref_out["refined_text"]
            n_ref_used = ref_out["n_ref_used"]
            fs_estimated_s = ref_out["fs_estimated_s"]
            refine_total_s = ref_out["refine_total_s"]
            in_range = ref_out["target_reached"]
            timed_out = ref_out["timed_out"]
            n_candidates_submitted = ref_out["n_candidates_submitted"]
            n_candidates_done = ref_out["n_candidates_done"]
            used_candidate_iter = ref_out["used_candidate_iter"]
            iter_llm_times_s = json.dumps([round(t, 3) for t in ref_out["llm_times_s"]])
            iter_fs_times_s = json.dumps([round(t, 3) for t in ref_out["fs_times_s"]])
            iter_tts_times_s = json.dumps(ref_out["iter_tts_times_s"])
            total_elapsed_s = ref_out["total_elapsed_s"]
            audio_seconds = ref_out["audio_seconds"]
            tts_api_s = ref_out["tts_api_s"]
            mp3_parse_s = ref_out["mp3_parse_s"]
            mp3_bytes = ref_out["mp3_bytes"]

            overrun_s = max(0.0, ref_out["total_elapsed_s"] - time_budget_s)

            try:
                seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
            except CouldntDecodeError as e:
                import warnings
                warnings.warn(f"Chunk {i}: pydub decode failed: {e}")
                seg = AudioSegment.silent(duration=int(audio_seconds * 1000))

        # ---- budget tracking ----
        audio_budget_remaining -= audio_seconds + overrun_s
        prev_audio_s = audio_seconds
        final_texts.append(refined)
        all_mp3_bytes.append(mp3_bytes)

        # ---- pre-start last chunk refinement two chunks early ----
        if (
            i == len(segments_list) - 3
            and len(segments_list) >= 3
            and _last_chunk_future is None
        ):
            last_idx = len(segments_list) - 1
            last_text = segments_list[last_idx]
            last_chars = len(last_text)
            remaining_chars_est = sum(len(c) for c in segments_list[i + 1:])
            last_target_s_est = audio_budget_remaining * (last_chars / max(remaining_chars_est, 1))
            last_tol_s_est = max(MIN_TOLERANCE_S, last_target_s_est * tolerance_ratio)
            last_tol_upper_s_est = max(MIN_TOLERANCE_S, last_target_s_est * TOLERANCE_RATIO_UPPER)
            _last_chunk_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            _last_chunk_future = _last_chunk_executor.submit(
                _adaptive_refine_parallel,
                client, last_text,
                last_target_s_est,
                total_budget_s,
                list(final_texts),
                last_tol_s_est,
                last_tol_upper_s_est,
                voice,
                MAX_REFINEMENTS,
                "",
                fast_initial_compress,
            )
            _last_chunk_lead_s = prev_audio_s or 0.0
            print(f"  [pre-start] chunk {last_idx} kicked off at chunk {i}, est_target={last_target_s_est:.1f}s")

        if out_dir is not None:
            (out_dir / f"chunk_{i:03d}.txt").write_text(refined, encoding="utf-8")
            (out_dir / f"chunk_{i:03d}.mp3").write_bytes(mp3_bytes)

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
            f"overrun={overrun_s:.2f}s | remaining={audio_budget_remaining:.1f}s"
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
    enable_early_cut: bool = True,
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

    segments = split_by_paragraphs(audio_content)

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
