"""One fresh six-turn debate: real TTS, wall-clock playback and real incremental ASR.

No billable calls without --execute. All model and audio requests share the existing
USD370 ledger and an atomic, durable USD280 study stop, including failed attempts.
"""
import argparse
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import sqlite3
import sys
import threading
import time
from types import MethodType, SimpleNamespace

import httpx
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(D))
from audio_probe import AudioGuard
from scripts.benchmark_incremental_planning import atomic_json, code_digest
from streaming.config import InputConfig, OutputConfig, SpeechBudgets
from utils.constants import LENGTH_MODE_FOR_DRAFT, TIME_MODE_FOR_STATEMENT
from streaming.experiment_accounting import exposure_query, reconcile_success
from streaming.experiment_client import BudgetedClient, BudgetExceeded, MODEL

RUN = 'listening-motion-live-v51-retrieval'
STUDY = 'listening-motion-live-'
LEDGER = D / 'run'
OUT = LEDGER / RUN
MANIFEST = D / f'manifest_{RUN}.json'
CAP, STUDY_CAP = 370., 280.
RUN_CAP = 10.
COMBINED_CAP = 60.
VALIDATION_RUNS = ('listening-motion-live-v48', 'listening-motion-live-v48-retrieval',
    'listening-motion-live-v49-retrieval', 'listening-motion-live-v50-retrieval', RUN)
TTS_ALLOWANCE = 1.
MOTION_FILE = Path('/mnt/data4/danqingwang/workspace/debate/data/motion_list_4.txt')
MOTION_NUMBER = 3
REHEARSAL_POOL_DIR = ROOT / 'results' / 'gemma-4-26b-a4b'
# ASR keeps played TTS boundaries; only its text output is buffered for analysis.
INPUT_CONFIG = InputConfig(min_text_words=100, max_text_wait_seconds=60)
CONFIG = OutputConfig(speech_mode='listening_prefix', listening_prefix_max_rewrites=2,
    audience_feedback_mode='compact',
    listening_prefix_max_calls=48,
    listening_prefix_pre_synthesize=True,
    listening_prefix_overlap_final_update=True,
    listening_parallel_body_feedback=True, listening_single_body_revision=True,
    listening_parallel_endpoint_revision=True,
    listening_planning_timeout_seconds=30,
    budget_mode='audio_duration', adaptive_delivery=True, first_chunk_seconds=16,
    first_body_chunk_seconds=0,  # Normal body length; no first-body duration ceiling.
    min_chunk_words=50, max_stream_chunks=5,
    first_chunk_local_tempo=False, allow_expansion=True,
    refinement_model=MODEL, max_refinements=10,
    early_max_refinements=3, max_parallel_tts=8, speed_adjust_min=1, speed_adjust_max=1)
SPEECH_BUDGETS = SpeechBudgets()
TURNS = [(stage, side, getattr(SPEECH_BUDGETS, stage))
         for stage in ('opening', 'rebuttal', 'closing') for side in ('for', 'against')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_study_guard(db, prefix=STUDY, cap=STUDY_CAP):
    """The trigger runs inside the reservation transaction, not around network I/O."""
    if not prefix or not 0 < cap <= CAP:
        raise ValueError('Invalid study cap or prefix')
    name = 'study_guard_' + hashlib.sha256(prefix.encode()).hexdigest()[:16]
    literal = "'" + prefix.replace("'", "''") + "'"
    exposure = exposure_query(f' WHERE instr(c.label,{literal})=1')
    sql = (f'CREATE TRIGGER {name} BEFORE INSERT ON calls '
           f'WHEN instr(NEW.label,{literal})=1 AND NEW.reserved + ({exposure}) > {float(cap)!r} '
           "BEGIN SELECT RAISE(ABORT, 'Study budget exceeded; no dispatch'); END")
    old = db.execute('select sql from sqlite_master where type=\'trigger\' and name=?', (name,)).fetchone()
    if old and old[0] != sql:
        raise ValueError('Cannot change an existing study stop')
    if not old:
        db.execute(sql)
        db.commit()
    return name


def combined_filter(column):
    return '(' + ' OR '.join(f"instr({column},'{run}/')=1" for run in VALIDATION_RUNS) + ')'


def arm_combined_guard(db):
    name = 'validation_v48_v51_combined_usd60'
    exposure = exposure_query(' WHERE ' + combined_filter('c.label'))
    sql = (f'CREATE TRIGGER {name} BEFORE INSERT ON calls '
           f"WHEN {combined_filter('NEW.label')} AND NEW.reserved + ({exposure}) > {COMBINED_CAP!r} "
           "BEGIN SELECT RAISE(ABORT, 'Combined validation USD60 exceeded; no dispatch'); END")
    old = db.execute("select sql from sqlite_master where type='trigger' and name=?", (name,)).fetchone()
    if old and old[0] != sql:
        raise ValueError('Cannot change an existing combined validation stop')
    if not old:
        db.execute(sql)
        db.commit()
    return name


def ledger_summary():
    client = BudgetedClient(LEDGER, cap=CAP)
    try:
        result = client.summary()
        result['study_exposure_usd'] = client.db.execute(
            exposure_query(' WHERE instr(c.label,?)=1'), (STUDY,)).fetchone()[0]
        result['study_known_usage_usd'] = client.db.execute(
            'select coalesce(sum(estimated_usd),0) from calls where instr(label,?)=1', (STUDY,)).fetchone()[0]
        result['combined_exposure_usd'] = client.db.execute(
            exposure_query(' WHERE ' + combined_filter('c.label'))).fetchone()[0]
        result['combined_known_usage_usd'] = client.db.execute(
            'select coalesce(sum(estimated_usd),0) from calls WHERE ' + combined_filter('label')).fetchone()[0]
        result['pending_calls'] = client.db.execute("select count(*) from calls where state='pending'").fetchone()[0]
        return result
    finally:
        client.db.close()


class Meter:
    """Each request has its own SQLite connection; listening and drafting overlap."""
    def __init__(self):
        self.stopped = threading.Event()
        self._stop_lock = threading.Lock()
        self.budget_stop = None

    def stop_for_budget(self, label, exc):
        # A speculative worker may catch the exception. Keep its cause durable
        # even when the shared latch interrupts playback with a generic error.
        with self._stop_lock:
            self.stopped.set()
            if self.budget_stop is None:
                self.budget_stop = dict(label=label, error=f'{type(exc).__name__}: {exc}',
                    utc=datetime.now(timezone.utc).isoformat(), dispatched=False)
                atomic_json(OUT/'budget_stop.json', self.budget_stop)

    def complete(self, label, messages, **kwargs):
        if self.stopped.is_set():
            raise BudgetExceeded('Run stopped; no new model dispatch')
        client = BudgetedClient(LEDGER, cap=CAP, label=RUN + '/' + label)
        try:
            return client.complete(messages, **kwargs)
        except (BudgetExceeded, sqlite3.IntegrityError) as exc:
            self.stop_for_budget(RUN + '/' + label, exc)
            raise
        finally:
            client.db.close()


def metered_tts_text_request(meter):
    """Route adaptive length edits through the shared ledger."""
    def request(client, model, messages, max_tokens, *, json_mode=False):
        if model != MODEL:
            raise ValueError('This motion run requires Gemma for every text call')
        phase = 'meaning_review' if json_mode else 'length_rewrite'
        content = meter.complete(f'tts/{phase}', messages, model=model,
                                 max_tokens=max_tokens, temperature=0, json_mode=json_mode)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return request


def manifest():
    sources = {}
    starting = ledger_summary()
    motion = MOTION_FILE.read_text().splitlines()[MOTION_NUMBER - 1].strip()
    for side in ('for', 'against'):
        pool = REHEARSAL_POOL_DIR / f"{motion.replace(' ', '_').lower()}_pool_{side}.json"
        sources[str(pool.relative_to(ROOT))] = digest(pool)
    result = dict(run_id=RUN, status='approved', created_utc=datetime.now(timezone.utc).isoformat(),
        authorization='User previously approved 跑一个motion验证，10刀预算 and 增加50刀预算 for this motion validation. Latest request 测试一遍 after the requested role-structured authoring fix authorizes one fresh repeat of the same six-turn motion03/Gemma/retrieval configuration. Existing cumulative USD60 validation cap covers v48, v48-retrieval, v49-retrieval, v50-retrieval (including its completion calls), and this v51 repeat. This run adds a stricter USD10 stop; no budget increase. Global USD370 and study USD280 caps unchanged. Prior usage and uncertain reservations remain accounted. Expected new cost USD1-3. No automatic rerun.',
        source_snapshot=str(OUT/'source_snapshot'),
        motion=motion, motion_number=MOTION_NUMBER, turns=TURNS,
        method='Fresh six-turn conversation with real TTS, wall-clock playback and real Whisper. Original TreeDebater authoring/revision templates, duration estimator, paragraph splitting and parallel TTS are reused. Listening planning selects grounded source IDs only. One native whole-speech authoring call produces the first paragraph and remaining draft together. Listening pre-synthesizes the first paragraph and first body audio for exact reuse. Native Audience feedback runs once on complete input for opening/rebuttal and is skipped for closing. Matching complete-ASR feedback/revision overlaps final tree/planning and is reused. The actual first paragraph passes a stance/source publication review. Approval binds exact text and review inputs; complete ASR changes require endpoint review before playback. Rejection regenerates the entire unpublished first-paragraph/body pair with one repair. Claim ranking, evidence and recall preparation run in the public TreeDebater lifecycle. Every draft inherits writer model, temperature, token limit, system and stage strategy. No preparatory body feedback, separate final body gate or TTS meaning review. Local format/length/source-version checks and immutable published audio are retained.',
        removed_flows=['prefix change review', 'preparatory Audience feedback',
            'final body gate/repair/recheck', 'TTS meaning review',
            'added paragraph quotas', '97-100% word target'],
        first_side='for', generation_model=MODEL, main_temperature=.3, helper_temperature=0,
        max_tokens=4096, planning_max_tokens=1100, two_revision_passes=False, output_config=asdict(CONFIG),
        length_settings=dict(draft=LENGTH_MODE_FOR_DRAFT, statement=TIME_MODE_FOR_STATEMENT),
        authoring_prompt_reuse=dict(draft='Shared stage strategy with role-structured speech history and one JSON output contract',
            revision='Shared native meaning constraints and stage strategy, role-structured history, immutable-prefix continuation',
            system='Configured debater system', feedback='Native Audience criteria, compact output',
            closing_feedback=False, speculative_reuse='Exact source, system, role-structured history, user prompt and options'),
        first_paragraph=dict(local_format_checks=True, max_format_repairs=1, semantic_review=True,
            pre_synthesize=True, immutable_after_publication=True),
        tts_allowance_per_turn_usd=TTS_ALLOWANCE,
        final_input_overlap=dict(enabled=True, complete_asr_feedback_revision=True,
            prefix_requires='Actual text approved against complete ASR and matching ready audio; final tree analysis may overlap playback',
            source_version_checks=True, obsolete_work_blocks_body=False),
        asr=dict(model='whisper-1', slices='Complete played chunks, split locally at sentence boundaries',
            parallel_requests=1, retries=0, ordered_analysis_workers=1,
            analysis_min_text_words=INPUT_CONFIG.min_text_words,
            analysis_max_text_wait_seconds=INPUT_CONFIG.max_text_wait_seconds, final_drain=True),
        retrieval=dict(use_retrieval=True, use_rehearsal_tree=True,
            evidence_source='Saved retrieved_evidence in the existing claim pools; no new web retrieval requests',
            selected_evidence='Native evidence_pool top10; complete high-quality pool only for native supplemental revision selection',
            shared_budget_record=str(D/'v51_combined_budget.json')),
        rehearsal=dict(enabled=True, mode='hybrid', pool_dir=str(REHEARSAL_POOL_DIR),
            index_cache_dir=str(REHEARSAL_POOL_DIR/'retrieval_indexes'),
            model='sentence-transformers/all-MiniLM-L6-v2', api_cost_usd=0),
        prices=dict(gemma_input_per_million=.13, gemma_output_per_million=.40,
            tts_per_million_characters=15, whisper_per_minute=.006),
        prices_checked_utc='2026-10-09', price_sources=['https://aws.amazon.com/bedrock/pricing/',
            'https://developers.openai.com/api/docs/models/tts-1',
            'https://developers.openai.com/api/docs/models/whisper-1'],
        estimate_usd=dict(expected=[1,3], conservative_incremental=min(RUN_CAP,
            CAP-starting['accounted_exposure_usd'], STUDY_CAP-starting['study_exposure_usd'],
            COMBINED_CAP-starting['combined_exposure_usd']),
            run_stop=RUN_CAP, combined_validation_stop=COMBINED_CAP, study_stop=STUDY_CAP, cumulative_cap=CAP,
            basis='Six speeches, 20min nominal audio; 30-100k TTS characters including discarded candidates; <=20min Whisper; approximately200-700 Gemma text requests including repeated preparation/length fitting. Text overhead and speculative requests included. Reserve each TTS bundle at USD1 only when needed, ASR at USD1.2, with 4x per-request audio safety bounds. Existing local host, no rented compute/storage. Dispatch stops if any reservation does not fit.'),
        guard='Atomic durable SQLite reservation transactions enforce global/study/run caps and a combined USD60 validation cap across v48, v48-retrieval, v49-retrieval, v50-retrieval and v51-retrieval, including prior and in-flight charges. Unknown failed charges stay reserved. Meter stop latch prevents new dispatch after budget failure. 30s planning timeout without retry. Stop on operational errors; no automatic rerun.',
        metrics=['first decoded audio from opponent endpoint', 'full duration/error',
            'maximum paced playback gap', 'phase call counts', 'transcripts and cumulative cost'],
        acceptance=dict(first_audio_max_seconds=10, max_gap_seconds=2,
            duration_error_ratio=.15, stop_after_first_failed_turn=False),
        limitations=['One fresh match, not a controlled paired ablation.',
            'Server-paced audio, not browser/network/microphone onset.',
            'First FOR opening is cold. Quality is manually inspected, not independently scored.'],
        input_sources=sources, source_digest=code_digest(), harness_sha256=digest(__file__),
        audio_guard_sha256=digest(D/'audio_probe.py'), starting_ledger=starting)
    atomic_json(MANIFEST, result)
    return result


def new_player(side, motion, meter, players=None):
    from agents import DebaterConfig
    from ouragents import TreeDebater
    pool = REHEARSAL_POOL_DIR / f"{motion.replace(' ', '_').lower()}_pool_{side}.json"
    opposite_pool = REHEARSAL_POOL_DIR / f"{motion.replace(' ', '_').lower()}_pool_{'against' if side == 'for' else 'for'}.json"
    if not pool.is_file() or not opposite_pool.is_file():
        raise ValueError('Both Gemma rehearsal pools must exist before a listening run')
    cfg = DebaterConfig(model=MODEL, helper_model=MODEL, temperature=.3, side=side,
        claim_selection_strategy='saved_scores',
        use_retrieval=True, use_rehearsal_tree=True, add_retrieval_feedback=False,
        pool_file=str(pool), rehearsal_mode='hybrid',
        rehearsal_index_cache_dir=str(REHEARSAL_POOL_DIR/'retrieval_indexes'),
        streaming_listen=True, streaming_tts=True, single_pass_revision=True,
        max_tokens=4096, planning=dict(mode='flat_tree', max_plan_tokens=1100,
            max_gate_tokens=120, max_updates=24, max_wait_chunks=3))
    p = TreeDebater(cfg, motion)
    if players is not None:
        players[side] = p
    p.streaming_output_config, p.debate_first_side = copy.deepcopy(CONFIG), 'for'
    p.speech_budgets = copy.deepcopy(SPEECH_BUDGETS)
    def main_response(this, messages, **kwargs):
        if '_completion' in kwargs:
            from agents import Agent
            return Agent._get_response(this, messages, **kwargs)
        return meter.complete(f'{side}/{p.status}/main', messages, max_tokens=4096,
            temperature=cfg.temperature, json_mode=kwargs.get('response_format', {}).get('type') == 'json_object')
    def helper(prompt, sys=None, response_model=None, max_tokens=4096, json_mode=None,
               history_messages=None, **kwargs):
        if kwargs.get('model', MODEL) != MODEL:
            raise ValueError('This motion run requires Gemma for every text call')
        from utils.model import helper_messages
        if response_model is not None:
            prompt += '\nReturn JSON matching this schema:\n' + json.dumps(response_model.model_json_schema())
        phase = next((key for key in ['LISTENING WHOLE SPEECH FEEDBACK', 'LISTENING FINAL BODY GATE', 'LISTENING BODY DRAFT', 'LISTENING BODY FEEDBACK',
            'LISTENING PREFIX DRAFT', 'LISTENING PREFIX REPAIR', 'ENDPOINT GATE',
            'LISTENING PREFIX REVIEW', 'Prepare compact JSON rebuttal choices'] if key in prompt),
            'helper').lower().replace(' ', '_')
        label_stage = (json.loads(prompt.rsplit('\n', 1)[-1])['stage']
                       if phase == 'listening_whole_speech_feedback' else p.status)
        return [meter.complete(f'{side}/{label_stage}/{phase}',
            helper_messages(prompt, sys=sys, history_messages=history_messages),
            max_tokens=min(max_tokens, 4096), temperature=kwargs.get('temperature', 0), model=kwargs.get('model', MODEL),
            request_timeout=kwargs.get('request_timeout', 120),
            json_mode=response_model is not None or (json_mode if json_mode is not None
                else 'json' in prompt.lower() or (sys is not None and 'json' in sys.lower())))]
    p._get_response, p.helper_client = MethodType(main_response, p), helper
    p.speculative_speech_safe = True
    for audience in p.simulated_audience:
        audience._get_response = MethodType(main_response, audience)
    # Existing files only: the existence check above prevents paid pool generation.
    p.claim_generation(4)
    atomic_json(OUT/f'claims_{side}.json', p.claim_preparation)
    atomic_json(OUT/f'preparation_{side}.json', dict(use_retrieval=p.use_retrieval,
        use_rehearsal_tree=p.use_rehearsal_tree, selected_claims=p.main_claims_content,
        evidence_pool_count=len(p.high_quality_evidence_pool),
        selected_evidence_count=len(p.evidence_pool),
        selected_evidence_ids=[e['id'] for e in p.evidence_pool],
        evidence_sources=[{k: e.get(k) for k in ('id', 'title', 'source', 'reliability')}
                          for e in p.high_quality_evidence_pool]))
    return p


def play_turn(index, speaker, listener, history, budget, endpoint, guard, asr_client, meter,
              *, playback_complete=None, input_completion=None, handoff=None,
              transcript_ready=None, recognized_input=None):
    stage, side, _ = TURNS[index]
    folder = OUT/f'{index:02}_{stage}_{side}'
    folder.mkdir()
    speaker.audio_output_dir = str(folder)
    started = time.perf_counter()
    chunks, playback, heard, analysis_batches = [], [], [], []
    arrivals = queue.Queue()
    record = dict(index=index, stage=stage, side=side, target_seconds=budget, status='running',
        history=copy.deepcopy(history), generation_started_monotonic=started,
        previous_opponent_endpoint_monotonic=endpoint, chunks=chunks, playback=playback, heard=heard,
        analysis_min_text_words=INPUT_CONFIG.min_text_words,
        analysis_max_text_wait_seconds=INPUT_CONFIG.max_text_wait_seconds, analysis_batches=analysis_batches)
    atomic_json(folder/'result.json', record)

    def emit(i, path, text, duration):
        decoded = AudioSegment.from_file(path)
        seconds = len(decoded)/1000
        if not seconds or abs(seconds-duration) > .1:
            raise ValueError('Unplayable or inconsistent audio callback')
        ready = time.perf_counter()
        item = dict(index=i, path=str(Path(path).relative_to(ROOT)), text=text,
                    duration_seconds=seconds, ready_monotonic=ready, ready_seconds=ready-started)
        chunks.append(item)
        atomic_json(folder/'events.json', chunks)
        arrivals.put((item, decoded))
        print(json.dumps(dict(event='audio_ready', turn=index, stage=stage, side=side,
            chunk=i, generation_wait=ready-started, endpoint_wait=ready-endpoint if endpoint else None)), flush=True)

    speaker.tts_chunk_callback = emit
    asr_jobs, analysis_jobs = [], []
    heard_lock = threading.Lock()
    generator = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'speaker-{side}')
    intake = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'asr-{listener.side}')
    analysis = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'analysis-{listener.side}')

    recognized_parts, expected_parts = {}, None
    recognized_lock = threading.Lock()

    def publish_transcript():
        # Called under the lock, from either endpoint or final recognition.
        if (transcript_ready is not None and not transcript_ready.done()
                and expected_parts is not None and len(recognized_parts) == expected_parts):
            record['full_asr_ready_monotonic'] = time.perf_counter()
            transcript_ready.set_result(' '.join(recognized_parts[i] for i in range(expected_parts)))

    def fail_transcript(exc):
        with recognized_lock:
            if transcript_ready is not None and not transcript_ready.done():
                transcript_ready.set_exception(exc)

    def update_heard(item, **fields):
        with heard_lock:
            item.update(fields)
            atomic_json(folder/'heard.json', heard)

    def fail_heard(item, exc):
        # Stop dispatch immediately; queued ASR and analysis tasks check the latch.
        meter.stopped.set()
        fail_transcript(exc)
        update_heard(item, error=f'{type(exc).__name__}: {exc}')

    def update_analysis(batch, members, **fields):
        with heard_lock:
            batch.update(fields)
            for item in members:
                item.update(analysis_batch=batch['index'], **fields)
            atomic_json(folder/'heard.json', heard)
            atomic_json(folder/'analysis_batches.json', analysis_batches)

    def analyze(batch, members):
        try:
            if meter.stopped.is_set():
                raise BudgetExceeded('Stopped before tree analysis')
            update_analysis(batch, members, analysis_start_monotonic=time.perf_counter())
            listener.status = stage
            listener.observe_opponent(batch['text'], side, stage)
            update_analysis(batch, members, analysis_end_monotonic=time.perf_counter())
            print(json.dumps(dict(event='heard', turn=index, batch=batch['index'],
                slices=batch['sequences'], words=batch['words'],
                backlog_seconds=batch['analysis_end_monotonic']-members[-1]['heard_end_monotonic'])), flush=True)
        except BaseException as exc:
            meter.stopped.set()
            fail_transcript(exc)
            update_analysis(batch, members, error=f'{type(exc).__name__}: {exc}')
            raise

    def submit_analysis(members, reason, waited):
        if meter.stopped.is_set():
            raise BudgetExceeded('Stopped before scheduling tree analysis')
        batch = dict(index=len(analysis_batches), sequences=[item['sequence'] for item in members],
            text=' '.join(item['text'] for item in members),
            words=sum(len(item['text'].split()) for item in members),
            final_flush=reason == 'final', flush_reason=reason, buffered_seconds=waited,
            queued_monotonic=time.perf_counter())
        with heard_lock:
            analysis_batches.append(batch)
        update_analysis(batch, members)
        analysis_jobs.append(analysis.submit(analyze, batch, members))

    def fail_batch_timer(exc):
        meter.stopped.set()
        fail_transcript(exc)

    from streaming.text_batching import TimedTextBatcher
    text_batches = TimedTextBatcher(INPUT_CONFIG, submit_analysis, fail_batch_timer)

    def receive(path, ended, sequence):
        item = dict(sequence=sequence, audio=str(path.relative_to(ROOT)), heard_end_monotonic=ended,
                    worker_start_monotonic=time.perf_counter())
        with heard_lock:
            heard.append(item)  # ASR worker preserves playback order.
        try:
            if meter.stopped.is_set():
                raise BudgetExceeded('Stopped before ASR')
            with path.open('rb') as f:
                text = asr_client.audio.transcriptions.create(model='whisper-1', file=f, language='en').text.strip()
            update_heard(item, asr_ready_monotonic=time.perf_counter(), text=text)
            if not text:
                raise ValueError('Empty real ASR result')
            if meter.stopped.is_set():
                raise BudgetExceeded('Stopped while ASR was in flight')
            with recognized_lock:
                recognized_parts[sequence] = text
                publish_transcript()
            # Complete recognized history is available immediately, independently
            # of the text threshold and the slower ordered analysis worker.
            text_batches.append(item)
            return text
        except BaseException as exc:
            fail_heard(item, exc)
            raise

    def generate():
        try:
            options = {}
            if input_completion is not None:
                def complete_input():
                    final_history = input_completion.result()
                    record['history'] = copy.deepcopy(final_history)
                    record['incoming_history_ready_monotonic'] = time.perf_counter()
                    return final_history
                options = dict(listening_input_completion=complete_input, listening_handoff=handoff,
                               listening_recognized_input=recognized_input)
            return getattr(speaker, stage+'_generation')(history,
                max_time=budget, time_control=True, streaming_tts=True, **options)
        finally:
            record['generation_finished_monotonic'] = time.perf_counter()
    future = generator.submit(generate)
    playback_end, sequence = None, 0
    deadline = started + 900
    try:
        while True:
            if meter.stopped.is_set():
                raise RuntimeError('Stop after model-budget or listener failure')
            if time.perf_counter() > deadline:
                raise TimeoutError('Turn exceeded900-second watchdog')
            if future.done() and arrivals.empty():
                future.result()
                break
            try:
                item, decoded = arrivals.get(timeout=.2)
            except queue.Empty:
                if future.done():
                    future.result()
                    break
                continue
            began = time.perf_counter()
            gap = 0 if playback_end is None else max(0, began-playback_end)
            played = dict(index=item['index'], start_monotonic=began, gap_seconds=gap)
            # TTS packs complete sentences. Preserve those acoustic boundaries;
            # hard15s crops can make Whisper lose a whole middle-of-clause slice.
            target = began + len(decoded)/1000
            while time.perf_counter() < target:
                if meter.stopped.wait(min(.2, target-time.perf_counter())):
                    raise RuntimeError('Listener failed during playback')
            ended = time.perf_counter()
            path = folder/f'heard_input_{sequence:03}.wav'
            decoded.export(path, format='wav')
            asr_jobs.append(intake.submit(receive, path, ended, sequence))
            sequence += 1
            playback_end = time.perf_counter()
            played['end_monotonic'] = playback_end
            playback.append(played)
            atomic_json(folder/'playback.json', playback)
        returned = future.result()
        if not chunks:
            raise RuntimeError('Speech returned without any audio')
        record['playback_endpoint_monotonic'] = playback_end
        with recognized_lock:
            expected_parts = sequence
            publish_transcript()
        if playback_complete is not None:
            playback_complete.set_result(playback_end)
        final_analysis = intake.submit(text_batches.close, drain=True)
        transcripts = [job.result() for job in asr_jobs]
        final_analysis.result()
        # ASR completion alone must not transfer ownership of mutable debate state.
        for job in analysis_jobs:
            job.result()
        if guard is not None and guard.artifact['blocked_dispatches']:
            raise BudgetExceeded('TTS guard recorded a blocked dispatch')
        record.update(status='completed', returned_text=returned,
            answer='\n\n'.join(c['text'] for c in chunks),
            listener_transcript=' '.join(transcripts), playback_endpoint_monotonic=playback_end,
            listener_drained_monotonic=time.perf_counter(),
            listener_backlog_seconds=max(0, time.perf_counter()-playback_end),
            generation_to_first_audio_seconds=chunks[0]['ready_seconds'],
            endpoint_to_first_audio_seconds=chunks[0]['ready_monotonic']-endpoint if endpoint else None,
            generation_start_delay_seconds=started-endpoint if endpoint else None,
            audio_seconds=sum(c['duration_seconds'] for c in chunks),
            playback_gap_seconds=sum(p['gap_seconds'] for p in playback),
            playback_span_seconds=playback_end-playback[0]['start_monotonic'])
        record['signed_duration_error_seconds'] = record['audio_seconds']-budget
        trace_path = next(folder.glob('*_chunks/listening_prefix.json'))
        record['delivery_trace'] = json.loads(trace_path.read_text())
        ready = record['delivery_trace'].get('speculative_candidate')
        record['candidate_ready_before_opponent_endpoint'] = (
            ready['ready_monotonic'] <= endpoint if ready and endpoint else None)
        record['original_words'] = len((record['delivery_trace']['fixed_prefix'] + ' ' +
                                       record['delivery_trace']['revised_tail']).split())
        record['answer_words'] = len(record['answer'].split())
        return record
    except BaseException as exc:
        fail_transcript(exc)
        meter.stopped.set()
        record.update(status='failed_partial' if chunks else 'failed_silent',
            error=f'{type(exc).__name__}: {exc}', audio_seconds=sum(c['duration_seconds'] for c in chunks),
            generation_to_first_audio_seconds=chunks[0]['ready_seconds'] if chunks else None,
            endpoint_to_first_audio_seconds=(chunks[0]['ready_monotonic']-endpoint) if chunks and endpoint else None)
        raise
    finally:
        text_batches.close()
        generator.shutdown(wait=True, cancel_futures=True)
        intake.shutdown(wait=True, cancel_futures=True)
        analysis.shutdown(wait=True, cancel_futures=True)
        speaker.tts_chunk_callback = None
        record['finished_monotonic'] = time.perf_counter()
        atomic_json(folder/'result.json', record)


def run_motion_turns(players, views, run, collect, meter):
    """Review against complete ASR; overlap speech with remaining tree analysis."""
    if not CONFIG.listening_prefix_overlap_final_update:
        endpoint = None
        for index, (_, side, _) in enumerate(TURNS):
            result = run(index, copy.deepcopy(views[side]), endpoint)
            collect(index, result)
            endpoint = result['playback_endpoint_monotonic']
        return
    pending, endpoint, waiting_input, previous_transcript = None, None, None, None
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='motion-turn') as executor:
        try:
            for index, (stage, side, budget) in enumerate(TURNS):
                handoff = None
                if pending is not None:
                    preparation = getattr(players[side], '_listening_prefix', None)
                    prior_stage, prior_side, _ = TURNS[index - 1]
                    if preparation is not None:
                        handoff = preparation.handoff(stage, f'{prior_side}:{prior_stage}', budget)
                waiting_input = Future() if pending is not None else None
                playback_complete = Future()
                transcript_ready = Future()
                recognized_input = None
                if previous_transcript is not None:
                    def recognized_input(prior=previous_transcript, base=copy.deepcopy(views[side]),
                                         previous_stage=prior_stage, previous_side=prior_side):
                        return copy.deepcopy(base) + [dict(stage=previous_stage, side=previous_side,
                            content=prior.result(), tree_via_streaming=True)]

                def execute(i=index, h=copy.deepcopy(views[side]), end=endpoint,
                            complete=playback_complete, incoming=waiting_input, ready=handoff,
                            recognized=transcript_ready, recognized_history=recognized_input):
                    try:
                        extra = dict(transcript_ready=recognized, recognized_input=recognized_history)
                        return run(i, h, end, playback_complete=complete,
                                   input_completion=incoming, handoff=ready, **extra)
                    except BaseException as exc:
                        if not complete.done():
                            complete.set_exception(exc)
                        if not recognized.done():
                            recognized.set_exception(exc)
                        raise

                current = executor.submit(execute)
                if pending is not None:
                    collect(index - 1, pending.result())
                    waiting_input.set_result(copy.deepcopy(views[side]))
                endpoint = playback_complete.result()
                pending = current
                previous_transcript = transcript_ready
            collect(len(TURNS) - 1, pending.result())
        except BaseException as exc:
            meter.stopped.set()
            if waiting_input is not None and not waiting_input.done():
                waiting_input.set_exception(exc)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(manifest(), indent=2))
        return
    m = json.loads(MANIFEST.read_text())
    assert m['status'] == 'approved'
    assert m['source_digest'] == code_digest() and m['harness_sha256'] == digest(__file__)
    assert m['audio_guard_sha256'] == digest(D/'audio_probe.py')
    for path, sha in m['input_sources'].items():
        assert digest(ROOT/path) == sha
    OUT.mkdir(exist_ok=False)
    snapshot = OUT/'source_snapshot'
    hashes = {}
    for path in sorted((ROOT/'src').rglob('*.py')):
        dest = snapshot/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = digest(path)
    atomic_json(snapshot/'files.json', hashes)
    (snapshot/'harness.py').write_bytes(Path(__file__).read_bytes())
    (snapshot/'audio_probe.py').write_bytes((D/'audio_probe.py').read_bytes())
    for key, value in json.loads((ROOT/'src/configs/api_key.json').read_text()).items():
        os.environ.setdefault(key, value)
    os.environ['DEBATE_LLM_API_BASE'] = 'http://127.0.0.1:4000/v1'
    import openai
    import litellm
    import tts_streaming
    from debate_tree import Tree
    def blocked(*args, **kwargs):
        raise RuntimeError('Unmetered call blocked')
    litellm.completion = blocked
    Tree.get_most_similar_node = lambda *args, **kwargs: (None, 0.)
    db_client = BudgetedClient(LEDGER, cap=CAP)
    m['combined_guard_trigger'] = arm_combined_guard(db_client.db)
    m['guard_trigger'] = arm_study_guard(db_client.db)
    m['run_guard_trigger'] = arm_study_guard(db_client.db, RUN+'/', RUN_CAP)
    db_client.db.close()
    meter, guards, players, clients = Meter(), [], {}, []
    results, views, history = [], {'for': [], 'against': []}, []
    original_factory = tts_streaming.OpenAI
    original_text_request = tts_streaming._text_request
    tts_streaming._text_request = metered_tts_text_request(meter)
    try:
        tts_guards, reservation_lock = [None] * len(TURNS), threading.Lock()

        def tts_guard(index):
            if CONFIG.tts_backend == 'fastspeech':
                return None
            # Reserve before the first client can dispatch, including preparation
            # for the next turn. Do not tie up budget for untouched future turns.
            with reservation_lock:
                if meter.stopped.is_set():
                    raise BudgetExceeded('Stopped before turn audio reservation')
                if tts_guards[index] is None:
                    try:
                        guard = AudioGuard(LEDGER, f'{RUN}/turn_{index}/tts', allowance=TTS_ALLOWANCE,
                                           approved_cap=CAP, max_requests=128)
                    except (BudgetExceeded, sqlite3.IntegrityError) as exc:
                        meter.stop_for_budget(f'{RUN}/turn_{index}/tts', exc)
                        raise
                    tts_guards[index] = guard
                    guards.append(guard)
                return tts_guards[index]
        asr_guard = AudioGuard(LEDGER, f'{RUN}/asr', allowance=1.2,
                               approved_cap=CAP, max_requests=128)
        guards.append(asr_guard)
        asr_client = openai.OpenAI(http_client=httpx.Client(transport=asr_guard, timeout=60),
            max_retries=0, base_url='https://api.openai.com/v1', timeout=60)
        clients.append(asr_client)
        for side in ('for', 'against'):
            players[side] = new_player(side, m['motion'], meter, players)
        for player in players.values():
            def prepare_audio(text, config, player=player):
                if meter.stopped.is_set():
                    raise BudgetExceeded('Stopped before preparatory TTS')
                if config.tts_backend == 'fastspeech':
                    return tts_streaming.synthesize_audio(None, text, config)
                from streaming.listening_prefix import next_stage
                opponent, heard_stage = player.planner.turn.split(':', 1)
                upcoming = next_stage(opponent, heard_stage, player.debate_first_side)
                turn_index = next(i for i, (s, side, _) in enumerate(TURNS)
                                  if s == upcoming and side == player.side)
                client = openai.OpenAI(http_client=httpx.Client(transport=tts_guard(turn_index), timeout=60),
                    max_retries=0, base_url='https://api.openai.com/v1', timeout=60)
                clients.append(client)
                return tts_streaming.synthesize_audio(client, text, config)
            player.listening_prefix_audio_preparer = prepare_audio
        m.update(status='running', launched_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(MANIFEST, m)
        def run(index, turn_history, endpoint, **options):
            stage, side, budget = TURNS[index]
            other = 'against' if side == 'for' else 'for'
            guard = tts_guard(index)
            def factory(**kwargs):
                if guard is None:
                    raise RuntimeError('OpenAI TTS is disabled for the local FastSpeech run')
                client = openai.OpenAI(http_client=httpx.Client(transport=guard, timeout=60),
                    max_retries=0, base_url='https://api.openai.com/v1', timeout=60)
                clients.append(client)
                return client
            tts_streaming.OpenAI = factory
            print(json.dumps(dict(event='turn_start', index=index, stage=stage, side=side)), flush=True)
            return play_turn(index, players[side], players[other], turn_history,
                budget, endpoint, guard, asr_client, meter, **options)

        def collect(index, r):
            r['debate_thoughts'] = {side: copy.deepcopy(p.debate_thoughts)
                                   for side, p in players.items()}
            stage, side, budget = TURNS[index]
            other = 'against' if side == 'for' else 'for'
            if tts_guards[index] is not None:
                tts_guards[index].finish()
                reconcile_success(tts_guards[index].db, tts_guards[index].request_id, tts_guards[index].path)
            results.append(r)
            exact = dict(stage=stage, side=side, content=r['answer'])
            history.append(exact)
            views[side].append(exact)
            views[other].append(dict(stage=stage, side=side, content=r['listener_transcript'], tree_via_streaming=True))
            atomic_json(OUT/'history.json', history)
            atomic_json(OUT/'listener_views.json', views)
            atomic_json(OUT/'results.json', results)
            print(json.dumps(dict(event='turn_complete', index=index, audio_seconds=r['audio_seconds'],
                first_audio=r['generation_to_first_audio_seconds'], endpoint_wait=r['endpoint_to_first_audio_seconds'],
                gaps=r['playback_gap_seconds'], next_listener_backlog=r['listener_backlog_seconds'])), flush=True)
            first = r['endpoint_to_first_audio_seconds'] if index else r['generation_to_first_audio_seconds']
            gap = max(p['gap_seconds'] for p in r['playback'])
            error = abs(r['signed_duration_error_seconds']) / budget
            r['timing_pass'] = first <= 10 and gap <= 2 and error <= .15
            atomic_json(OUT/'results.json', results)
            if not r['timing_pass']:
                print(json.dumps(dict(event='timing_failure', index=index, first_audio=first,
                    max_gap=gap, duration_error_ratio=error, action='continue_full_motion')), flush=True)
        run_motion_turns(players, views, run, collect, meter)
        m['timing_pass'] = all(r['timing_pass'] for r in results)
        m['status'] = 'completed'
    except BaseException as exc:
        meter.stopped.set()
        m.update(status='stopped', stop_error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        meter.stopped.set()
        try:
            for p in players.values():
                p.discard_listening_prefix()
            for client in clients:
                client.close()
            for guard in guards:
                guard.finish()
        finally:
            tts_streaming.OpenAI = original_factory
            tts_streaming._text_request = original_text_request
            try:
                # Preserve the native TreeDebater record, as src/env.py does.
                m['debate_thoughts'] = {side: copy.deepcopy(p.debate_thoughts)
                                       for side, p in players.items()}
            except BaseException as exc:
                m.update(status='stopped', artifact_error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                m.update(finished_utc=datetime.now(timezone.utc).isoformat(), completed_turns=len(results),
                         ending_ledger=ledger_summary(), budget_stop=meter.budget_stop)
                atomic_json(MANIFEST, m)


if __name__ == '__main__':
    main()
