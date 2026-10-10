"""
Streaming TTS pipeline with adaptive refinement.

Splits debate text into chunks, adaptively refines each chunk's length to hit its proportional time budget, and generates TTS candidates in parallel.

Key features vs the serial pipeline in tts.py:
  - Chunk-based processing: text is split by paragraphs, each chunk gets a proportional share of the total time budget.
  - Adaptive refinement: The configured backend estimates duration; if off-target, an LLM rewrites the chunk to a target word count.  Multiple TTS candidates are submitted in parallel and the closest-to-target is picked.
  - Streaming overlap: while chunk N's audio plays, chunk N+1 is being refined and TTS-generated (time_budget for chunk N+1 = audio duration of chunk N).
  - Configurable length edits: shorten-only mode and optional meaning checks protect the original text; model checks remain fallible.
"""

import concurrent.futures
import csv
import json
import os
import sys
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
from utils.time_estimator import LengthEstimator, TTS_INPUT_LIMIT
from utils import speech_length
from streaming.config import OutputConfig, from_mapping
from streaming.audio_tempo import fit_audio_tempo
from streaming.delivery_edit import EditScope
from streaming.revision_stream import RevisionStream

# Compatibility aliases; canonical defaults live in streaming.config.OutputConfig.
TOLERANCE_RATIO = OutputConfig.tolerance_ratio
EARLY_CUT_RATIO = OutputConfig.early_cut_ratio
MIN_CHUNK_WORDS = OutputConfig.min_chunk_words


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
    local_tempo_input_seconds: float = 0.0
    local_tempo_speed: float = 1.0
    local_tempo_processing_s: float = 0.0
    local_tempo_status: str = 'disabled'
    local_tempo_clamped: bool = False
    local_tempo_error: str = ''


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


def _in_range(est: float, target_s: float, tol_s: float, tol_upper_s: float, *, allow_short=False) -> bool:
    return (est - target_s) <= tol_upper_s and (allow_short or (target_s - est) <= tol_s)


def duration_estimator(config=None):
    """Bind optional paid estimation to the renderer's client factory and output settings.

    Experiments replace OpenAI here with their guarded factory. The estimator
    itself never constructs a separate client or chooses another voice/model.
    """
    cfg = from_mapping(OutputConfig, config)
    def audio_duration(text):
        client = OpenAI()
        try:
            return _query_time_profiled(client, text, voice=cfg.voice, model=cfg.model)['audio_seconds']
        finally:
            client.close()
    return speech_length.statement_estimator(audio_duration=audio_duration)


def estimate_statement_seconds(text, config=None):
    return duration_estimator(config).query_time(text)


def _estimate_duration(text: str, *, measured_seconds_per_word=None, client=None, config=None, voice=None) -> float:
    cfg = from_mapping(OutputConfig, config)
    audio_duration = (lambda chunk: _query_time_profiled(client, chunk,
        voice=voice or cfg.voice, model=cfg.model)['audio_seconds']) if client is not None else None
    return speech_length.estimate_seconds(text, measured_seconds_per_word=measured_seconds_per_word,
                                         audio_duration=audio_duration)


def _text_request(client, model, messages, max_tokens, *, json_mode=False):
    """Use the configured rewrite model; Gemma may use the existing local proxy."""
    options = {}
    owned = None
    if "deepseek" in model.lower():
        owned = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
        model = model.split("/", 1)[-1]
        options["extra_body"] = {"thinking": {"type": "disabled"}}
    elif model.startswith('google.gemma') and os.environ.get('DEBATE_LLM_API_BASE'):
        owned = OpenAI(base_url=os.environ['DEBATE_LLM_API_BASE'],
                       api_key=os.environ.get('DEBATE_LLM_API_KEY') or 'local-proxy')
    elif model == 'gpt-5-mini':
        options['reasoning_effort'] = 'minimal'
    if json_mode:
        options['response_format'] = {'type': 'json_object'}
    try:
        return (owned or client).chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens, **options)
    finally:
        if owned is not None:
            owned.close()


def _revise_to_n_words(client, text: str, n_words: int, prev_texts: List[str], next_chunk_text: str = "", model: str = OutputConfig.refinement_model, motion: str = "", side: str = "", *, source_text=None) -> str:
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
            f"(read-only: do not copy, paraphrase, anticipate or move its claims into your paragraph; "
            f"it will still be spoken in full):\n\n"
            f"{next_chunk_text}"
        )

    resp = _text_request(client, model, [
            {
                "role": "system",
                "content": (
                    f"Debate motion: {motion or 'not supplied'}. Assigned side: {side or 'preserve the original position'}. "
                    "FOR supports the motion; AGAINST opposes it. This is context only: preserve the paragraph's existing position and meaning, without evaluating or correcting stance contradictions. "
                    "You are helping refine a paragraph from a competitive debate speech. "
                    "The speech is delivered orally; every word will be read aloud by a text-to-speech system. "
                    "Rewrite ONLY the paragraph provided by the user. "
                    "Preserve the argument, logical flow, and debate rhetoric. "
                    "This is a length edit, not a new debate argument: preserve the speaker's position, "
                    "claim ownership, negation, qualifications, and the distinction between quoting and endorsing a claim. "
                    "Do not turn a concession into agreement with the opposing side or strengthen it into the conclusion. "
                    "Never swap 'we argue' and 'my opponent argues'. Meaning takes priority over the target word count. "
                    "Do NOT add new arguments or repeat points already made in the preceding text. "
                    "The immutable source owns this paragraph's content. Expand or compress only that "
                    "content; neighbouring paragraphs are context, never material to fill the word target. "
                    "A previous length-edit proposal cannot authorize content absent from the immutable source. "
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
                    f"\n\nImmutable source for this paragraph:\n{source_text if source_text is not None else text}"
                    f"{next_block}"
                ),
            },
        ], 4096)
    return (resp.choices[0].message.content or "").strip()


def _validate_tts_input(content):
    if len(content) > TTS_INPUT_LIMIT:
        raise ValueError(f'TTS input exceeds {TTS_INPUT_LIMIT} characters; split it before synthesis')


def _query_time_profiled(client, content: str, voice: str = "echo", speed: float = 1.0, model: str = OutputConfig.model) -> Dict[str, Any]:
    _validate_tts_input(content)
    t0 = _now()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=content,
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


def _tts_with_retry(client, content: str, voice: str = "echo", speed: float = 1.0, max_attempts: int = 5, model: str = OutputConfig.model) -> Dict[str, Any]:
    _validate_tts_input(content)
    for attempt in range(max_attempts):
        try:
            return _query_time_profiled(client, content, voice=voice, speed=speed, model=model)
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
    fs_estimated_s: float
    future: Any             # concurrent.futures.Future -> Dict from _query_time_profiled
    worker_label: str = ""  # "prestart" / "normal" — which worker produced this
    intra_iter: int = 0     # iteration index within the producing worker (0 = raw, k = k-th refine)
    completed_at: Optional[float] = None
    edit_scope: Optional[EditScope] = None
    obsolete: bool = False
    published: bool = False
    audio_reused: bool = False


def _pick_best_completed(
    candidates: List[_TtsCandidate],
    target_s: float,
    deadline: Optional[float] = None,
) -> Tuple[_TtsCandidate, Dict]:
    def _collect_done() -> List[Tuple[_TtsCandidate, Dict]]:
        out = []
        for c in candidates:
            if c.obsolete:
                continue
            if c.future.done():
                if (deadline is not None and c.intra_iter > 0
                        and (c.completed_at is None or c.completed_at > deadline)):
                    continue
                try:
                    out.append((c, c.future.result()))
                except Exception:
                    pass
        return out

    done = _collect_done()

    if not done:
        concurrent.futures.wait(
            [c.future for c in candidates if not c.obsolete],
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        done = _collect_done()

    if not done:
        concurrent.futures.wait([c.future for c in candidates if not c.obsolete])
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
      - the remaining queued playback window expires -> no further optional edits
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
        config: Optional[OutputConfig] = None,
        motion: str = "",
        side: str = "",
    ):
        self.config = from_mapping(OutputConfig, config)
        self.client = client
        self.motion = motion
        self.side = side
        self.original_text = original_text
        self.seconds_per_word = None
        self.rewrite_checks = []

        self._target_s = target_s
        self._tol_s = tol_s
        self._tol_upper_s = tol_upper_s
        self._target_lock = threading.Lock()

        self.candidates: List[_TtsCandidate] = []
        self.candidates_lock = threading.Lock()

        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self._adopt_lock = threading.RLock()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_parallel_tts)

        self.prev_texts = list(prev_texts)
        self.next_chunk_text = next_chunk_text
        self._edit_scope = EditScope(original_text, tuple(prev_texts), next_chunk_text)
        self.voice = voice
        self.max_ref = max_ref

        self.kickoff_iter = kickoff_iter
        self.kickoff_kind = kickoff_kind

        # per-worker stats: label -> {"n_ref", "llm_times", "fs_times"}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._stats_lock = threading.Lock()

        self.estimation_error = None
        self.chosen_cand: Optional[_TtsCandidate] = None
        self.chosen_tts_out: Optional[Dict[str, Any]] = None

        self.t_start_wall = _now()
        self.workers: List[threading.Thread] = []
        self.refinement_deadline = None
        self.parallel_refinement = self.config.adaptive_delivery and self.config.max_parallel_tts > 1
        self.prepared_audio = None

    def refinement_expired(self):
        return self.refinement_deadline is not None and _now() >= self.refinement_deadline

    def fail_estimation(self, error):
        with self._adopt_lock:
            if self.estimation_error is None:
                self.estimation_error = error
            self.stop_event.set()
            self.done_event.set()

    def raise_estimation_error(self):
        if self.estimation_error is not None:
            error = self.estimation_error
            raise RuntimeError(f'Duration estimation failed: {type(error).__name__}: {error}') from error

    def get_target(self) -> Tuple[float, float, float]:
        with self._target_lock:
            return self._target_s, self._tol_s, self._tol_upper_s

    def edit_scope(self):
        with self._adopt_lock:
            return self._edit_scope

    def update_context(self, prev_texts, next_chunk_text):
        """Expire contextual edits atomically; retain immutable source audio.

        A prestart can synthesize before the preceding segment is committed. Its
        approval is speculative too: neither a ready result nor a late callback
        may promote it after that context changes. Fresh edits use one snapshot
        through writing, review and synthesis registration.
        """
        with self._adopt_lock:
            old = self._edit_scope
            if old.preceding == tuple(prev_texts) and old.following == next_chunk_text:
                return
            self._edit_scope = EditScope(self.original_text, tuple(prev_texts),
                                         next_chunk_text, old.revision + 1)
            self.prev_texts = list(prev_texts)
            self.next_chunk_text = next_chunk_text
            with self.candidates_lock:
                for candidate in self.candidates:
                    if candidate.edit_scope is not None:
                        candidate.obsolete = True
            if self.chosen_cand is not None and self.chosen_cand.obsolete:
                self.chosen_cand = None
                self.chosen_tts_out = None
                self.done_event.clear()
                self.stop_event.clear()

    def reusable_proposal(self):
        """Reuse speculative text, never its stale approval; prefer ready audio."""
        with self._adopt_lock, self.candidates_lock:
            candidates = []
            for c in self.candidates:
                if not c.obsolete or c.edit_scope is None or c.edit_scope.source != self.original_text:
                    continue
                try:
                    seconds = c.future.result()['audio_seconds'] if c.future.done() else c.fs_estimated_s
                except Exception:
                    continue
                candidates.append((c, seconds))
            if not candidates:
                return None
            target = self.get_target()[0]
            return min(candidates, key=lambda row: abs(row[1] - target))[0]

    def update_target(self, target_s: float, tol_s: float, tol_upper_s: float) -> None:
        with self._adopt_lock:
            with self._target_lock:
                self._target_s = target_s
                self._tol_s = tol_s
                self._tol_upper_s = tol_upper_s
            if (self.config.adaptive_delivery and self.chosen_tts_out is not None
                    and not _in_range(self.chosen_tts_out['audio_seconds'], target_s, tol_s, tol_upper_s,
                                      allow_short=not self.config.allow_expansion)):
                self.chosen_cand = None
                self.chosen_tts_out = None
                self.done_event.clear()
                self.stop_event.clear()

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

    def add_candidate(self, text: str, est: float, label: str, intra_iter: int,
                      *, edit_scope=None) -> Optional[_TtsCandidate]:
        # The adoption lock also closes the review-to-registration race. Workers
        # always supply their captured scope; direct callers bind to the current one.
        with self._adopt_lock, self.candidates_lock:
            if intra_iter > 0:
                edit_scope = edit_scope or self._edit_scope
                if edit_scope != self._edit_scope:
                    return None
            # Normal and prestart branches share the same raw audio and identical edits.
            key = ' '.join(text.split())
            for existing in self.candidates:
                if (not existing.obsolete and existing.edit_scope == edit_scope
                        and ' '.join(existing.text.split()) == key):
                    return existing
            iteration = len(self.candidates)
            cand = _TtsCandidate(
                iteration=iteration,
                text=text,
                fs_estimated_s=est,
                future=None,
                worker_label=label,
                intra_iter=intra_iter,
                edit_scope=edit_scope,
            )
            def synthesize():
                # Queued optional work must not start another request after publication.
                if intra_iter > 0 and (cand.obsolete or self.stop_event.is_set() or self.refinement_expired()):
                    raise concurrent.futures.CancelledError('Candidate missed playback deadline')
                try:
                    if intra_iter == 0 and self.prepared_audio is not None:
                        future = self.prepared_audio.match(text, self.voice, self.config.model)
                        if future is not None:
                            try:
                                result = future.result()
                            except Exception:
                                pass  # Failed speculation falls back to ordinary synthesis.
                            else:
                                cand.audio_reused = True
                                self.prepared_audio.mark_reused()
                                return result
                    return _tts_with_retry(self.client, text, self.voice, model=self.config.model)
                finally:
                    cand.completed_at = _now()
            # Audio is immutable and can be shared after a NEW contextual review.
            # A queued obsolete request will cancel itself, so only share audio
            # already running or successfully finished.
            reusable = next((c for c in self.candidates
                if c.text == text and (c.future.running() or
                    (c.future.done() and not c.future.cancelled() and c.future.exception() is None))), None)
            if reusable is None:
                cand.future = self.executor.submit(synthesize)
            else:
                cand.future = reusable.future
                cand.audio_reused = True
                cand.completed_at = reusable.completed_at
                cand.future.add_done_callback(lambda future: setattr(cand, 'completed_at', reusable.completed_at))
            self.candidates.append(cand)
        if self.parallel_refinement:
            # Selection must respond to ANY finished audio, even while the writer
            # is preparing another candidate. Never hold the pool lock in callbacks.
            def completed(future):
                try:
                    self.try_adopt(cand, future.result())
                except Exception:
                    pass  # Failed candidates leave the original/other audio available.
            cand.future.add_done_callback(completed)
        return cand

    def try_adopt(self, cand: _TtsCandidate, tts_out: Dict[str, Any]) -> bool:
        with self._adopt_lock:
            if cand.obsolete or (cand.edit_scope is not None and cand.edit_scope != self._edit_scope):
                return False
            if self.done_event.is_set() or (self.config.adaptive_delivery
                    and (self.stop_event.is_set() or self.refinement_expired())):
                return False
            if ((self.config.adaptive_delivery or not self.config.allow_expansion)
                    and not _in_range(tts_out['audio_seconds'], *self.get_target(),
                                      allow_short=not self.config.allow_expansion)):
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
    attempted_texts = {' '.join(cur.split())}

    if ctx.stop_event.is_set():
        return

    # ---- step 0: fs estimate raw text + submit raw TTS candidate ----
    t = _now()
    try:
        est = _estimate_duration(cur, measured_seconds_per_word=ctx.seconds_per_word,
                                 client=ctx.client, config=ctx.config, voice=ctx.voice)
    except Exception as exc:
        ctx.fail_estimation(exc)
        return
    ctx.add_fs_time(label, _now() - t)

    if ctx.stop_event.is_set():
        return

    target_s, tol_s, tol_upper_s = ctx.get_target()
    cand = ctx.add_candidate(cur, est, label, intra_iter=0)

    if (cand.future.done() or (ctx.config.adaptive_delivery and not ctx.parallel_refinement)
            or _in_range(est, target_s, tol_s, tol_upper_s, allow_short=not ctx.config.allow_expansion)):
        try:
            tts_out = cand.future.result()
            if ctx.try_adopt(cand, tts_out):
                return
            if ctx.config.adaptive_delivery:
                est = float(tts_out['audio_seconds'])
        except Exception:
            pass

    # ---- LLM refinement loop ----
    n_ref_local = 0
    worker_scope = ctx.edit_scope()
    prepared_proposal = ctx.reusable_proposal() if label == 'normal' else None
    while not ctx.stop_event.is_set() and not ctx.refinement_expired() and n_ref_local < ctx.max_ref:
        if worker_scope != ctx.edit_scope():
            break  # The normal worker owns the new context; never continue an old edit chain.
        target_s, tol_s, tol_upper_s = ctx.get_target()
        if target_s <= 0:
            break

        cw = LengthEstimator.count_words(cur)
        tw = max(1 if ctx.config.adaptive_delivery else 10,
                 round(cw * target_s / max(est, .001 if ctx.config.adaptive_delivery else 1.0)))

        if not ctx.config.allow_expansion and tw >= cw:
            break
        previous = cur
        t = _now()
        reused_proposal = prepared_proposal is not None
        if reused_proposal:
            cur = prepared_proposal.text
            prepared_proposal = None
        else:
            try:
                cur = _revise_to_n_words(ctx.client, cur, tw, list(worker_scope.preceding),
                    worker_scope.following, model=ctx.config.refinement_model, motion=ctx.motion,
                    side=ctx.side, source_text=worker_scope.source)
            except Exception:
                break
            ctx.add_llm_time(label, _now() - t)
        n_ref_local += 1
        check = {'worker': label, 'original': ctx.original_text, 'candidate': cur,
                 'accepted': True, 'verified': False, 'edit_scope': worker_scope.payload(),
                 'reused_proposal': reused_proposal}
        if not cur or (not ctx.config.allow_expansion
                       and LengthEstimator.count_words(cur) > LengthEstimator.count_words(previous)):
            check.update(accepted=False, reason='Empty or expanded length edit')
        elif ' '.join(cur.split()) in attempted_texts:
            check.update(accepted=False, reason='Unchanged or previously attempted text; reuse existing audio')
        elif ctx.stop_event.is_set() or ctx.refinement_expired():
            check.update(accepted=False, reason='Cancelled before verification or synthesis')
        elif worker_scope != ctx.edit_scope():
            check.update(accepted=False, reason='Context changed during writing; approval withheld')
        if worker_scope != ctx.edit_scope():
            check.update(accepted=False, reason='Context changed before synthesis; approval withheld')
        with ctx._stats_lock:
            ctx.rewrite_checks.append(check)
        if not check['accepted']:
            break  # Raw TTS remains in the candidate pool; never fail the whole turn for a bad edit.

        if ctx.stop_event.is_set() or ctx.refinement_expired():
            break
        attempted_texts.add(' '.join(cur.split()))

        t = _now()
        try:
            est = _estimate_duration(cur, measured_seconds_per_word=ctx.seconds_per_word,
                                 client=ctx.client, config=ctx.config, voice=ctx.voice)
        except Exception as exc:
            ctx.fail_estimation(exc)
            return
        ctx.add_fs_time(label, _now() - t)

        if ctx.stop_event.is_set() or ctx.refinement_expired():
            break

        cand = ctx.add_candidate(cur, est, label, intra_iter=n_ref_local, edit_scope=worker_scope)
        if cand is None:
            with ctx._stats_lock:
                check.update(accepted=False, reason='Context changed at synthesis registration')
            break

        target_s, tol_s, tol_upper_s = ctx.get_target()
        if (cand.future.done() or (ctx.config.adaptive_delivery and not ctx.parallel_refinement)
                or _in_range(est, target_s, tol_s, tol_upper_s, allow_short=not ctx.config.allow_expansion)):
            try:
                tts_out = cand.future.result()
                if ctx.try_adopt(cand, tts_out):
                    return
                if ctx.config.adaptive_delivery:
                    est = float(tts_out['audio_seconds'])
            except Exception:
                pass


def _wait_for_refinement(ctx, deadline):
    """Stop at the playback deadline or when no worker can improve the audio.

    Initial synthesis is mandatory even when body preparation consumed the whole
    playback buffer. Let it be submitted; candidate selection can then await its
    result. The worker's deadline prevents optional edits after that point.
    """
    while not ctx.done_event.is_set():
        with ctx.candidates_lock:
            pending_audio = any(not c.obsolete and not c.future.done() for c in ctx.candidates)
        if not any(worker.is_alive() for worker in ctx.workers) and not pending_audio:
            return _now() >= deadline
        remaining = deadline - _now()
        if remaining <= 0:
            with ctx.candidates_lock:
                if ctx.candidates:
                    return True
        ctx.done_event.wait(timeout=min(.02, remaining) if remaining > 0 else .02)
    return False


# -------- chunk utilities --------
def _normalize_seam_silence(seg: AudioSegment, config: OutputConfig) -> Tuple[AudioSegment, bool]:
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
    start = max(0, lead - config.seam_head_ms)
    end = len(seg) - trail
    core = seg[start:end]
    out = core.fade_in(config.seam_fade_ms).fade_out(config.seam_fade_ms) + AudioSegment.silent(duration=config.seam_tail_ms, frame_rate=seg.frame_rate)
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
    *, estimate=None,
) -> Tuple[str, str]:
    """
    Split text into (head, tail) where head's FS estimate ≈ target_s.
    Returns (head, tail); tail may be empty if the whole text fits.
    Only called when fs_estimate(text) / target_s > early_cut_ratio.
    """
    estimate = estimate or _estimate_duration
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return text, ""

    head_sentences: List[str] = []
    for sent in sentences:
        candidate = " ".join(head_sentences + [sent])
        est = estimate(candidate)
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


def split_into_chunks(text: str, total_budget_s: float, config: Optional[OutputConfig] = None) -> List[str]:
    """Split a speech into streamable chunks.

    Paragraphs are the preferred unit (they are natural rhetorical units and the
    refinement prompt talks about "paragraphs"). Two failure modes of pure
    paragraph splitting are handled here:
      1. The model emitted (almost) no paragraph breaks -> a single huge chunk,
         which disables streaming and refinement entirely. In that case the text
         is re-chunked by packing sentences to ~cfg.target_chunk_seconds of audio each.
      2. One paragraph is much longer than the rest (> cfg.max_chunk_chars) -> it is
         split at sentence boundaries into a few balanced pieces.
    """
    cfg = from_mapping(OutputConfig, config)
    paras = split_by_paragraphs(text)
    if not paras:
        return []
    total_chars = sum(len(p) for p in paras)

    if len(paras) < cfg.min_stream_chunks:
        n_target = int(round(total_budget_s / cfg.target_chunk_seconds)) if total_budget_s > 0 else cfg.min_stream_chunks
        n_target = max(cfg.min_stream_chunks, min(cfg.max_stream_chunks, n_target))
        target_chars = max(200, total_chars // n_target)
        out: List[str] = []
        for p in paras:
            out.extend(_pack_sentences(p, target_chars))
        return out or paras

    out = []
    for p in paras:
        if len(p) > cfg.max_chunk_chars:
            n_pieces = -(-len(p) // cfg.max_chunk_chars)  # ceil
            out.extend(_pack_sentences(p, max(200, len(p) // n_pieces)))
        else:
            out.append(p)
    return out


def split_body_chunks(tail, remaining, config):
    """Use the same paragraph boundaries for speculative and committed audio."""
    tail, _ = remove_citation(tail)
    tail = remove_subtitles(tail)
    return _merge_short_chunks(split_into_chunks(tail, remaining, config),
                               min_words=config.min_chunk_words)


# -------- pipeline --------
def _run_pipeline(
    client,
    segments_list: List[str],
    total_budget_s: float,
    tolerance_ratio: Optional[float] = None,
    voice: Optional[str] = None,
    out_dir: Optional[Path] = None,
    enable_early_cut: Optional[bool] = None,
    early_cut_ratio: Optional[float] = None,
    config: Optional[OutputConfig] = None,
    on_chunk=None,
    motion: str = "",
    side: str = "",
    tail_supplier=None,
    validate_chunk=None,
    prepared_first_audio=None,
    prepared_body_audio=None,
    _contexts=None,
) -> Tuple[List[ChunkProfile], RoundProfile, bytes, List[str]]:
    """
    Run the streaming TTS pipeline on a list of text segments.

    Returns:
        (chunk_profiles, round_profile, combined_mp3_bytes, final_texts)
    """
    cfg = from_mapping(OutputConfig, config)
    tolerance_ratio = cfg.tolerance_ratio if tolerance_ratio is None else tolerance_ratio
    voice = cfg.voice if voice is None else voice
    enable_early_cut = cfg.enable_early_cut if enable_early_cut is None else enable_early_cut
    early_cut_ratio = cfg.early_cut_ratio if early_cut_ratio is None else early_cut_ratio
    estimate = lambda text: _estimate_duration(text, client=client, config=cfg, voice=voice)
    round_t0 = _now()
    chunk_profiles: List[ChunkProfile] = []

    refine_total_acc = 0.0
    tts_api_total = 0.0
    mp3_parse_total = 0.0
    audio_total = 0.0
    overrun_total = 0.0

    has_listening_prefix = tail_supplier is not None
    if has_listening_prefix:
        if len(segments_list) != 1 or not segments_list[0].strip():
            raise ValueError('Deferred speech requires exactly one fixed prefix')
        segments_list = list(segments_list)
    else:
        segments_list = list(_merge_short_chunks(segments_list, min_words=cfg.min_chunk_words))
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
    playback_end = None  # Estimated end of all published, queued audio.
    use_playback_deadline = cfg.adaptive_delivery or tail_supplier is not None
    body_stream = None
    stream_finished = True
    unseen_chars = 0.0

    # Pre-start contexts indexed by target chunk idx. A context is created when
    # we kick off a prestart worker for that chunk (ratio or last-chunk variant).
    # When the main loop reaches that chunk, it pops the context and adds a
    # normal refine worker that shares the same candidate pool.
    _chunk_contexts: Dict[int, _ChunkRefineContext] = {}

    def _measured_rate():
        words = sum(LengthEstimator.count_words(text) for text in final_texts)
        return audio_total / words if words and cfg.adaptive_delivery else None

    def _kickoff_ratio_prestart(iter_i: int) -> None:
        """At the start of iter iter_i, maybe kick off prestart for c[i+2].

        Triggers if EITHER:
          - chunk[i+2].chars / chunk[i+1].chars >= cfg.ratio_prestart_threshold, OR
          - chunk[i+2].chars >= cfg.abs_prestart_chars (absolutely long chunk)
        """
        target_idx = iter_i + 2
        if target_idx >= len(segments_list) - 1:   # would be the last chunk → handled by last-chunk kickoff
            return
        if target_idx in _chunk_contexts:
            return
        next_chars = len(segments_list[iter_i + 1])
        target_chars = len(segments_list[target_idx])
        ratio = (target_chars / next_chars) if next_chars > 0 else 0.0
        ratio_trigger = ratio >= cfg.ratio_prestart_threshold
        abs_trigger = target_chars >= cfg.abs_prestart_chars
        if not (ratio_trigger or abs_trigger):
            return
        target_text = segments_list[target_idx]
        tgt_s_est = total_budget_s * target_chars / max(total_chars_initial, 1)
        tol_est = max(cfg.min_tolerance_seconds, tgt_s_est * tolerance_ratio)
        tol_upper_est = max(cfg.min_tolerance_seconds, tgt_s_est * cfg.last_chunk_upper_tolerance_ratio)
        ctx = _ChunkRefineContext(
            motion=motion, side=side,
            config=cfg,
            client=client,
            original_text=target_text,
            target_s=tgt_s_est,
            tol_s=tol_est,
            tol_upper_s=tol_upper_est,
            prev_texts=list(final_texts),
            next_chunk_text=segments_list[target_idx + 1] if target_idx + 1 < len(segments_list) else "",
            voice=voice,
            max_ref=cfg.max_refinements,
            kickoff_iter=iter_i,
            kickoff_kind="ratio",
        )
        ctx.seconds_per_word = _measured_rate()
        _contexts.append(ctx)
        _chunk_contexts[target_idx] = ctx
        th = threading.Thread(target=_refine_worker, args=(ctx, "prestart"), daemon=True)
        ctx.workers.append(th)
        th.start()
        triggers = []
        if ratio_trigger:
            triggers.append(f"ratio={ratio:.2f}x")
        if abs_trigger:
            triggers.append(f"abs={target_chars}c>={cfg.abs_prestart_chars}")
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
        tol_est = max(cfg.min_tolerance_seconds, tgt_s_est * tolerance_ratio)
        tol_upper_est = max(cfg.min_tolerance_seconds, tgt_s_est * cfg.last_chunk_upper_tolerance_ratio)
        ctx = _ChunkRefineContext(
            motion=motion, side=side,
            config=cfg,
            client=client,
            original_text=last_text,
            target_s=tgt_s_est,
            tol_s=tol_est,
            tol_upper_s=tol_upper_est,
            prev_texts=list(final_texts),
            next_chunk_text="",
            voice=voice,
            max_ref=cfg.max_refinements,
            kickoff_iter=iter_i,
            kickoff_kind="last",
        )
        ctx.seconds_per_word = _measured_rate()
        _contexts.append(ctx)
        _chunk_contexts[last_idx] = ctx
        th = threading.Thread(target=_refine_worker, args=(ctx, "prestart"), daemon=True)
        ctx.workers.append(th)
        th.start()
        print(
            f"  [last-prestart] chunk {last_idx} kicked off at start of iter {iter_i}, "
            f"target_est={tgt_s_est:.1f}s"
        )

    i = 0
    while i < len(segments_list) or not stream_finished:
        if body_stream is not None:
            ready, stream_finished, unseen_chars = body_stream.read(wait=i >= len(segments_list))
            segments_list.extend(ready)
            total_chars_initial = sum(len(c) for c in segments_list)
            if i >= len(segments_list):
                break
        chunk = segments_list[i]
        n_chunks = len(segments_list)  # may grow due to early-cut

        _validate_tts_input(chunk)
        chunk_t0 = _now()
        chunk_words = LengthEstimator.count_words(chunk)
        chunk_chars = len(chunk)

        remaining_chars_total = sum(len(c) for c in segments_list[i:]) + unseen_chars
        target_s = audio_budget_remaining * (chunk_chars / remaining_chars_total)
        if i == 0 and (tail_supplier is not None or cfg.first_chunk_local_tempo):
            target_s = min(cfg.first_chunk_seconds, audio_budget_remaining)

        # ---- early-cut: if chunk is too long relative to budget, split it now ----
        if enable_early_cut and i > 0:
            fs_pre = estimate(chunk)
            if fs_pre / target_s > early_cut_ratio:
                head, tail = _early_cut_chunk(chunk, target_s, early_cut_ratio, estimate=estimate)
                if tail:
                    segments_list[i] = head
                    segments_list.insert(i + 1, tail)
                    chunk = head
                    n_chunks = len(segments_list)
                    chunk_chars = len(chunk)
                    chunk_words = LengthEstimator.count_words(chunk)
                    remaining_chars_total = sum(len(c) for c in segments_list[i:]) + unseen_chars
                    target_s = audio_budget_remaining * (chunk_chars / remaining_chars_total)
                    print(f"  chunk {i:03d} | early-cut: fs_pre={fs_pre:.1f}s > {early_cut_ratio}x target={target_s:.1f}s → split into head({len(head)}c)+tail({len(tail)}c)")

        if (i == 1 and has_listening_prefix and cfg.first_body_chunk_seconds > 0
                and (not stream_finished or i + 1 < n_chunks)):
            target_s = min(target_s, cfg.first_body_chunk_seconds)
        tol_s = max(cfg.min_tolerance_seconds, target_s * tolerance_ratio)
        remaining_chunks = n_chunks - i
        tol_upper_s = (
            max(cfg.min_tolerance_seconds, target_s * cfg.last_chunk_upper_tolerance_ratio)
            if remaining_chunks == 1 and stream_finished
            else tol_s
        )

        max_ref = cfg.early_max_refinements if (not stream_finished or i < n_chunks // 2) else cfg.max_refinements
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
        if stream_finished and not ((cfg.adaptive_delivery or tail_supplier is not None) and i == 0):
            _kickoff_ratio_prestart(i)
            _kickoff_last_chunk_prestart(i)

        # ---- chunk 0: no refinement, sequential TTS ----
        if i == 0:
            time_budget_s = target_s
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
                    if prepared_first_audio and (prepared_first_audio['text'], prepared_first_audio['voice'],
                            prepared_first_audio['model']) == (refined, voice, cfg.model):
                        tts_out = dict(prepared_first_audio['tts_out'], tts_api_s=0.)
                        prepared_first_audio = None
                    else:
                        tts_out = _query_time_profiled(client, refined, voice=voice, model=cfg.model)
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

        # ---- chunks 1+: shared candidate pool with prestart + normal workers ----
        else:
            time_budget_s = (max(0., playback_end - _now())
                             if use_playback_deadline else prev_audio_s)
            margin_s = min(cfg.refine_deadline_margin_seconds, time_budget_s / 2)
            deadline = (playback_end - margin_s if use_playback_deadline
                        else _now() + time_budget_s - margin_s)

            # Pop existing prestart context (if any), or build a fresh context
            ctx = _chunk_contexts.pop(i, None)
            if ctx is not None and ctx.original_text != chunk:
                # Early cuts or inserted segments changed source ownership. Keep
                # the retired pool for lifecycle cleanup, never reuse its audio.
                ctx.stop_event.set()
                ctx = None
            if ctx is not None:
                # Prestart was running; push the up-to-date target so its next
                # iteration uses real budget instead of the initial estimate.
                ctx.update_target(target_s, tol_s, tol_upper_s)
                ctx.update_context(final_texts, next_chunk_text)
                chunk_prestart_kind = ctx.kickoff_kind
            else:
                ctx = _ChunkRefineContext(
                    motion=motion, side=side,
                    config=cfg,
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
                _contexts.append(ctx)

            ctx.seconds_per_word = _measured_rate()
            if i == 1:
                ctx.prepared_audio = prepared_body_audio
            if use_playback_deadline:
                ctx.refinement_deadline = deadline

            # Always start a normal worker for this chunk (in addition to any
            # prestart worker that may already be running on the same context).
            normal_th = threading.Thread(target=_refine_worker, args=(ctx, "normal"), daemon=True)
            ctx.workers.append(normal_th)
            normal_th.start()

            # Wait for ANY worker to find an ok candidate, OR until deadline.
            # Reserve delivery time without consuming the entire window for short chunks.
            wait_s = max(0.0, deadline - _now())
            if use_playback_deadline:
                deadline_expired = _wait_for_refinement(ctx, deadline)
            elif max_ref == 0:
                # No alternative text can arrive: a completed raw synthesis is
                # ready for selection even if its duration misses the target.
                normal_th.join(timeout=wait_s)
                deadline_expired = normal_th.is_alive()
            else:
                deadline_expired = not ctx.done_event.wait(timeout=wait_s)
            ctx.stop_event.set()

            total_elapsed_s = _now() - ctx.t_start_wall

            ctx.raise_estimation_error()

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
                chosen_cand, tts_out = _pick_best_completed(snap, target_now, deadline=ctx.refinement_deadline)
                in_range = _in_range(tts_out['audio_seconds'], *ctx.get_target(),
                                     allow_short=not cfg.allow_expansion)
                timed_out = deadline_expired

            # Speed adjustment if still out of range
            target_now, tol_now, tol_upper_now = ctx.get_target()
            audio_s = float(tts_out["audio_seconds"])
            slack_s = (playback_end - _now() if use_playback_deadline
                       else time_budget_s - (_now() - chunk_t0))
            if (not _in_range(audio_s, target_now, tol_now, tol_upper_now,
                              allow_short=not cfg.allow_expansion)
                    and slack_s >= cfg.speed_adjust_min_slack_seconds):
                raw_speed = audio_s / target_now if target_now > 0 else 1.0
                clamped = max(cfg.speed_adjust_min, min(cfg.speed_adjust_max, raw_speed))
                if abs(clamped - 1.0) > 0.01:
                    try:
                        speed_tts_out = _tts_with_retry(client, chosen_cand.text, voice=voice, speed=clamped, model=cfg.model)
                        if abs(speed_tts_out["audio_seconds"] - target_now) < abs(audio_s - target_now):
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

            # Publish the chosen candidate now. The outer lifecycle joins unused
            # requests after delivery, rather than delaying this chunk to clean up.

            try:
                seg = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
            except CouldntDecodeError as e:
                import warnings
                warnings.warn(f"Chunk {i}: pydub decode failed: {e}")
                seg = AudioSegment.silent(duration=int(audio_seconds * 1000))

        # Normalize before local tempo and delivery; all consumers use the final duration.
        audio_changed = False
        if cfg.normalize_seams:
            seg, seam_changed = _normalize_seam_silence(seg, cfg)
            audio_changed = seam_changed
        tempo_input_seconds = len(seg) / 1000.0
        tempo_speed, tempo_time = 1.0, 0.0
        tempo_status, tempo_error, tempo_clamped = 'disabled', '', False
        if i == 0 and cfg.first_chunk_local_tempo:
            tempo_start = _now()
            original_seg, original_bytes = seg, mp3_bytes
            try:
                adjusted, tempo = fit_audio_tempo(seg, target_s,
                    min_speed=cfg.local_tempo_min, max_speed=cfg.local_tempo_max,
                    deadband_seconds=cfg.local_tempo_deadband_seconds)
                tempo_status, tempo_clamped = tempo.status, tempo.clamped
                if tempo.status == 'applied':
                    buf = BytesIO()
                    adjusted.export(buf, format='mp3')
                    # The decoded artifact is authoritative for callback/profile/budget.
                    decoded = AudioSegment.from_file(BytesIO(buf.getvalue()), format='mp3')
                    if abs(len(decoded) / 1000 - target_s) < abs(tempo_input_seconds - target_s):
                        seg, mp3_bytes = decoded, buf.getvalue()
                        tempo_speed = tempo.speed
                        audio_changed = False  # Already encoded above.
                    else:
                        tempo_status = 'not_improved'
            except Exception as exc:
                # Local processing must never trigger another paid TTS call.
                seg, mp3_bytes = original_seg, original_bytes
                tempo_status, tempo_error = 'failed_original_used', type(exc).__name__
            tempo_time = _now() - tempo_start
            total_elapsed_s += tempo_time
            overrun_s += tempo_time
        if audio_changed:
            buf = BytesIO()
            seg.export(buf, format="mp3")
            mp3_bytes = buf.getvalue()
        audio_seconds = len(seg) / 1000.0
        in_range = _in_range(audio_seconds, target_s, tol_s, tol_upper_s,
                             allow_short=not cfg.allow_expansion and not (i == 0 and cfg.first_chunk_local_tempo))

        # ---- budget tracking ----
        if validate_chunk is not None:
            validate_chunk(i, refined)
        if i > 0:
            chosen_cand.published = True
        audio_budget_remaining -= audio_seconds
        if cfg.budget_mode == "experiment_elapsed":
            audio_budget_remaining -= overrun_s
        prev_audio_s = audio_seconds
        final_texts.append(refined)
        all_mp3_bytes.append(mp3_bytes)

        if out_dir is not None:
            (out_dir / f"chunk_{i:03d}.txt").write_text(refined, encoding="utf-8")
            chunk_path = out_dir / f"chunk_{i:03d}.mp3"
            temporary_path = chunk_path.with_suffix(".mp3.tmp")
            temporary_path.write_bytes(mp3_bytes)
            temporary_path.replace(chunk_path)
        ready = _now()
        playback_end = max(playback_end or ready, ready) + audio_seconds
        if out_dir is not None:
            if on_chunk is not None:
                on_chunk(i, chunk_path, refined, len(seg) / 1000.0)

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
            local_tempo_input_seconds=tempo_input_seconds,
            local_tempo_speed=tempo_speed,
            local_tempo_processing_s=tempo_time,
            local_tempo_status=tempo_status,
            local_tempo_clamped=tempo_clamped,
            local_tempo_error=tempo_error,
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
        if i == 1 and tail_supplier is not None:
            tail = tail_supplier()
            if isinstance(tail, RevisionStream):
                body_stream = tail
                stream_finished = False
            elif not isinstance(tail, str):
                raise ValueError('Remaining speech must be text')
            elif tail.strip():
                # Reuse TreeDebater's ordinary paragraph split and short-block merge.
                # The published prefix stays separate; only its actual audio duration
                # is deducted from the remaining body's budget.
                tail_segments = split_body_chunks(tail, audio_budget_remaining, cfg)
                segments_list.extend(tail_segments)
                total_chars_initial = sum(len(c) for c in segments_list)
            tail_supplier = None

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


def run_pipeline(*args, **kwargs):
    """Run synthesis and settle every worker before returning or raising.

    Publishing callbacks still run immediately; cleanup does not delay playback.
    No background rewrite/TTS request may escape a completed speech's accounting.
    """
    contexts = []
    try:
        return _run_pipeline(*args, **kwargs, _contexts=contexts)
    finally:
        active_error = sys.exc_info()[1]
        for context in contexts:
            context.stop_event.set()
        for context in contexts:
            for worker in context.workers:
                worker.join()
            context.executor.shutdown(wait=True, cancel_futures=True)
        out_dir = kwargs.get('out_dir', args[5] if len(args) > 5 else None)
        if out_dir is not None and contexts:
            audits = [{'original': context.original_text, 'checks': context.rewrite_checks,
                       'final_edit_scope': context.edit_scope().payload(),
                       'candidates': [dict(text=c.text, source_unchanged=c.edit_scope is None,
                            edit_scope=c.edit_scope.payload() if c.edit_scope else None,
                            obsolete=c.obsolete, published=c.published, audio_reused=c.audio_reused)
                            for c in context.candidates]}
                      for context in contexts]
            path = Path(out_dir) / 'rewrite_audit.json'
            temporary = path.with_suffix('.json.tmp')
            try:
                temporary.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(path)
            except OSError as exc:
                if active_error is None:
                    raise
                import warnings
                warnings.warn(f'Rewrite audit could not be saved while handling {type(active_error).__name__}: {type(exc).__name__}')

        if active_error is None:
            for context in contexts:
                context.raise_estimation_error()


# -------- high-level API --------
def convert_incremental_speech_to_audio(producer, output_path, total_budget_s, *, config=None, on_chunk=None):
    """Pull one reviewed argument, publish its audio, then prepare the next.

    Playback is driven by the existing callback/file bridge, independently of
    this producer. TTS never rewrites checked text. Oversize candidates are
    regenerated and reviewed before publication, never clipped mid-argument.
    The publication log and combined audio retain the committed prefix on error.
    """
    from streaming.flat_speaking import SegmentRejected

    cfg = from_mapping(OutputConfig, config)
    if not isinstance(total_budget_s, (int, float)) or not 0 < total_budget_s < float('inf'):
        raise ValueError('Speech budget must be positive and finite')
    output_path = Path(output_path)
    out_dir = output_path.parent / f'{output_path.stem}_chunks'
    out_dir.mkdir(parents=True, exist_ok=True)
    # A failed/repeated turn must not expose old chunks through the file bridge.
    if any(out_dir.glob('chunk_*.mp3')):
        raise FileExistsError(f'Speech chunks already exist: {out_dir}')
    client = OpenAI()
    start = _now()
    audio_total = 0.0
    gap_total = 0.0
    playback_end = None  # Producer-side queue estimate; not browser playback telemetry.
    combined = AudioSegment.silent(duration=0)
    trace = {'mode': 'flat_incremental_speaking', 'budget_mode': cfg.budget_mode,
             'total_budget_s': total_budget_s, 'chunks': [], 'status': 'running'}
    trace_path = out_dir / 'speaking.json'

    def save_trace():
        trace.update(audio_seconds=audio_total, estimated_gap_seconds=gap_total,
                     wall_seconds=_now() - start, committed_text=producer.text)
        temporary = trace_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(trace_path)

    try:
        for index in range(cfg.max_stream_chunks):
            if producer.done:
                break
            elapsed_gaps = (gap_total + max(0, _now() - playback_end)) if playback_end is not None else 0
            remaining = total_budget_s - audio_total
            if cfg.budget_mode == 'experiment_elapsed':
                remaining -= elapsed_gaps
            if remaining < 3:
                trace['status'] = 'budget_exhausted'
                break
            target = min(remaining, cfg.first_chunk_seconds if index == 0 else cfg.later_chunk_seconds)
            published = False
            # At most one shorter, fully rechecked candidate for an audio overrun.
            for attempt in range(2):
                final = index == cfg.max_stream_chunks - 1 or remaining <= target + 3
                prepare_start = _now()
                text = producer.prepare(target, final=final, max_chars=min(4000, cfg.max_chunk_chars))
                text_ready = _now()
                if text is None:
                    break
                producer.validate(text)
                if len(text) > 4000:
                    raise SegmentRejected('Checked paragraph exceeds the TTS input limit')
                tts = _tts_with_retry(client, text, voice=cfg.voice, model=cfg.model)
                seg = AudioSegment.from_file(BytesIO(tts['mp3_bytes']), format='mp3')
                mp3_bytes = tts['mp3_bytes']
                if cfg.normalize_seams:
                    seg, changed = _normalize_seam_silence(seg, cfg)
                    if changed:
                        buf = BytesIO()
                        seg.export(buf, format='mp3')
                        mp3_bytes = buf.getvalue()
                duration = len(seg) / 1000.0
                if duration <= 0:
                    raise ValueError('TTS returned empty audio')
                ready = _now()
                gap = max(0, ready - playback_end) if playback_end is not None else 0
                remaining = total_budget_s - audio_total
                if cfg.budget_mode == 'experiment_elapsed':
                    remaining -= gap_total + gap
                if duration > remaining:
                    trace.setdefault('discarded', []).append({'index': index, 'attempt': attempt,
                                                             'audio_seconds': duration,
                                                             'remaining_seconds': remaining})
                    if remaining < 3:
                        break
                    target = min(target * .7, remaining * .8)
                    continue
                # Publication is the immutability boundary, even if a player has
                # buffered this audio and has not started playing it yet.
                producer.validate(text)
                chunk_path = out_dir / f'chunk_{index:03d}.mp3'
                temporary = chunk_path.with_suffix('.mp3.tmp')
                temporary.write_bytes(mp3_bytes)
                (out_dir / f'chunk_{index:03d}.txt').write_text(text, encoding='utf-8')
                producer.commit(text, publish=lambda: temporary.replace(chunk_path))
                combined += seg
                audio_total += duration
                gap_total += gap
                playback_end = max(playback_end or ready, ready) + duration
                trace['chunks'].append({'index': index, 'text': text,
                                        'prepare_start_seconds': prepare_start - start,
                                        'text_ready_seconds': text_ready - start,
                                        'audio_ready_seconds': ready - start,
                                        'audio_seconds': duration, 'estimated_gap_seconds': gap})
                if index == 0:
                    trace['first_text_seconds'] = text_ready - start
                    trace['first_audio_seconds'] = ready - start
                save_trace()
                if on_chunk is not None:
                    on_chunk(index, chunk_path, text, duration)
                published = True
                break
            if not published:
                trace['status'] = 'budget_exhausted' if not producer.done else 'completed'
                break
        if trace['status'] == 'running':
            trace['status'] = 'completed' if producer.done else 'chunk_limit'
        if not producer.committed:
            raise SegmentRejected('No complete reviewed argument fits the speech budget')
    except Exception as exc:
        trace['status'] = 'failed'
        trace['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Do not mask the original generation/playback exception with diagnostics.
        import sys
        failure_in_flight = sys.exc_info()[0] is not None
        try:
            save_trace()
            if producer.committed:
                temporary = output_path.with_suffix('.mp3.tmp')
                combined.export(temporary, format='mp3')
                temporary.replace(output_path)
        except Exception:
            if not failure_in_flight:
                raise
    return producer.text, '', audio_total


def convert_text_to_speech_streaming(
    content: str,
    output_path: str,
    total_budget_s: float,
    voice: Optional[str] = None,
    enable_early_cut: Optional[bool] = None,
    early_cut_ratio: Optional[float] = None,
    config: Optional[OutputConfig] = None,
    on_chunk=None,
    motion: str = "",
    side: str = "",
    tail_supplier=None,
    validate_chunk=None,
    prepared_first_audio=None,
    prepared_body_audio=None,
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
    audio_content, reference = remove_citation(content)
    audio_content = remove_subtitles(audio_content)
    body_stream = None

    if tail_supplier is not None:
        original_supplier = tail_supplier
        def tail_supplier():
            nonlocal reference, body_stream
            tail = original_supplier()
            if isinstance(tail, RevisionStream):
                body_stream = tail
                return tail
            if not isinstance(tail, str):
                raise ValueError('Remaining speech must be text')
            spoken, tail_reference = remove_citation(tail)
            if tail_reference:
                reference = '\n\n'.join(filter(None, (reference, tail_reference)))
            return remove_subtitles(spoken)

    cfg = from_mapping(OutputConfig, config)
    segments = ([audio_content.strip()] if tail_supplier is not None else
                split_into_chunks(audio_content, total_budget_s, cfg))

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
        config=config,
        on_chunk=on_chunk,
        motion=motion, side=side,
        tail_supplier=tail_supplier,
        validate_chunk=validate_chunk,
        prepared_first_audio=prepared_first_audio,
        prepared_body_audio=prepared_body_audio,
    )

    if body_stream is not None and body_stream.reference:
        reference = '\n\n'.join(filter(None, (reference, body_stream.reference)))

    # Save combined audio
    output_path.write_bytes(combined_mp3_bytes)

    # Compute duration from the combined audio
    duration = MP3(BytesIO(combined_mp3_bytes)).info.length

    # Build text_content and reference matching the original API
    text_content = "\n\n".join(final_texts)

    print(
        f"  => audio_total={round_profile.audio_seconds_total:.2f}s | "
        f"overrun_total={round_profile.overrun_total_s:.2f}s | "
        f"budget_remaining={round_profile.budget_remaining_s:.2f}s | "
        f"wall_clock={round_profile.round_total_s:.2f}s"
    )

    return text_content, reference, duration
