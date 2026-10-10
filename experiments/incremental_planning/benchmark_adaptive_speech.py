"""Real-model paired speech test. Nothing billable runs without --execute.

Shared prepared Flat Tree state, fresh generation, fixed baseline speak method,
real TTS, exact published transcripts, and failure-inclusive reporting.
"""
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import httpx
from pydub import AudioSegment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from audio_probe import AudioGuard
from scripts.benchmark_incremental_planning import atomic_json, code_digest, judge, make_player
from streaming.config import OutputConfig
from streaming.experiment_accounting import MODEL_RATES, exposure_query
from streaming.experiment_client import BudgetedClient, BudgetExceeded, MODEL

RUN = 'flat-full-adaptive-v1'
DIRECTORY = ROOT / 'experiments/incremental_planning'
LEDGER = DIRECTORY / 'run'
OUTPUT = LEDGER / RUN
LIMIT = 48.0
CASE_IDS = ['focus_bus_display', 'focus_room_booking', 'focus_lecture_recordings',
            'focus_riverside_stall', 'focus_day_lockers', 'focus_hybrid_meetings',
            'focus_food_hub', 'focus_school_news']
CONFIGS = {
    'full_locked12': OutputConfig(budget_mode='audio_duration', adaptive_delivery=True,
        first_chunk_seconds=12, later_chunk_seconds=30, max_refinements=0, early_max_refinements=0,
        max_parallel_tts=1, speed_adjust_min=1, speed_adjust_max=1, refinement_model=MODEL),
    'full_adaptive12': OutputConfig(budget_mode='audio_duration', adaptive_delivery=True,
        first_chunk_seconds=12, later_chunk_seconds=30, max_refinements=2, early_max_refinements=1,
        max_parallel_tts=1, refinement_model=MODEL),
    'full_adaptive8': OutputConfig(budget_mode='audio_duration', adaptive_delivery=True,
        first_chunk_seconds=8, later_chunk_seconds=30, max_refinements=2, early_max_refinements=1,
        max_parallel_tts=1, refinement_model=MODEL),

}


def run_exposure(client):
    return client.db.execute(exposure_query(' WHERE instr(c.label, ?)=1'), (RUN + '/',)).fetchone()[0]


def require_room(client, bound):
    if run_exposure(client) + bound > LIMIT:
        raise BudgetExceeded('Run exposure limit reached; no new dispatch')


class ScopedClient(BudgetedClient):
    # One serial text dispatcher; audio reserves its entire bundle first. No
    # concurrent text worker can race this additional run-level stop check.
    def complete(self, messages, *, max_tokens=700, temperature=0, json_mode=False, model=MODEL):
        body = {'model': model, 'messages': messages, 'max_tokens': max_tokens,
                'temperature': temperature, 'num_retries': 0}
        if model == 'gpt-5.6-sol':
            body['reasoning_effort'] = 'none'
            body.pop('temperature')
        if json_mode:
            body['response_format'] = {'type': 'json_object'}
        i, o = MODEL_RATES[model]
        bound = 4 * ((len(json.dumps(body, ensure_ascii=False).encode()) + 8192) * max(1, i)
                     + max_tokens * max(1, o)) / 1e6
        require_room(self, bound)
        return super().complete(messages, max_tokens=max_tokens, temperature=temperature,
                                json_mode=json_mode, model=model)


class ThreadedClient(ScopedClient):
    """One metered text request at a time, with a connection owned by its thread.

    Audio can overlap text. The whole per-turn audio bundle is already reserved;
    the lock also makes the additional run-prefix guard safe across text workers.
    """
    dispatch_lock = threading.Lock()

    def complete_at(self, label, messages, **kwargs):
        with self.dispatch_lock:
            local = ScopedClient(self.directory, base_url=self.base_url, label=label)
            try:
                return local.complete(messages, **kwargs)
            finally:
                local.db.close()

    def complete(self, messages, **kwargs):
        return self.complete_at(self.label, messages, **kwargs)


def cases():
    data = json.loads((DIRECTORY / 'cases_focus_v1.json').read_text())
    return [next(c for c in data if c['id'] == name) for name in CASE_IDS]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_manifest():
    client = BudgetedClient(LEDGER)
    manifest = {
        'run_id': RUN, 'status': 'prepared', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User requested real-model first-audio latency and quality. Continue the existing '
                         'explicitly approved cumulative USD200 task budget; no increase or reset.',
        'source_digest': code_digest(), 'baseline_commit': '1f55a44',
        'baseline_file_sha256': digest(DIRECTORY / 'flat_speech_baseline.py'),
        'harness_sha256': digest(Path(__file__)),
        'revision': 'Whole Flat speech generation retained. Compare locked TTS, adaptive rewrites, shorter first chunks, and fixed-opening synthesis overlapping remaining-draft revision.',
        'cases_file_sha256': digest(DIRECTORY / 'cases_focus_v1.json'),
        'case_ids': CASE_IDS, 'repeats': 1, 'arms': list(CONFIGS), 'answers': 32,
        'order': 'Rotate four-arm order by case_index%4; one serial speech worker. Internal audio/text overlap remains enabled.',
        'preparation': 'Fresh real-model Flat preparation per case, deepcopy identical state for all four arms; fresh generation for every arm.',
        'model': MODEL, 'main_temperature': .3, 'helper_temperature': 0, 'max_tokens': 1600,
        'judge_model': 'gpt-5.6-sol', 'judge_max_tokens': 1600, 'speech_budget_seconds': 60,
        'output_configs': {arm: asdict(config) for arm, config in CONFIGS.items()}, 'audience_reviewers': 1,
        'tts': 'tts-1 echo real OpenAI audio; no ASR. Adaptive arms enable bounded length rewrites and speed adjustment.',
        'matching': 'Exact target matching in both arms; embedding fallback disabled as in prior evaluations.',
        'refinement': 'Original TTS length-rewrite prompts, metered Gemma helper at temperature0/max1600 through the local proxy, not the default gpt-5-mini. First chunk is not rewritten by TTS.',
        'quality': 'Same frozen per-case rubric/GPT judge on exact published transcript, including partial turns. '
                   'Empty turns recorded separately; do not let absence of falsehood inflate quality.',
        'latency': 'Primary: generation call start to first decoded playable audio callback. '
                   'Separately report simulated 2.3-words/sec listener backlog plus output wait. '
                   'Neither metric includes actual ASR, browser transport or playback.',
        'failures': 'No answer regeneration, no changed verdict, no automatic worker restart. '
                    'Existing in-pipeline repair/retry remains measured. One missing-judge retry only after diagnosis.',
        'estimate_usd': {'expected': [1, 3], 'conservative_usage_planning': 10,
                         'run_exposure_stop': LIMIT, 'cumulative_cap': 200,
                         'basis': '32 speeches, <=32 independent judgments, 8 shared preparations; '
                                  'roughly 200-500 Gemma calls including adaptive rewrites, 32 USD1 TTS bundles with <=24 audio HTTP requests each. No rented compute/storage.'},
        'rates': {**MODEL_RATES, 'tts-1_per_million_characters': 15},
        'price_sources': ['https://aws.amazon.com/bedrock/pricing/',
                          'https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html',
                          'https://developers.openai.com/api/docs/models/tts-1'],
        'starting_ledger': client.summary(),
        'guard': 'Persistent shared SQLite pre-dispatch bounds; text verified receipts settle at 4x cost; '
                 'per-turn USD1 audio bundle reserves before HTTP, max24 audio calls, 4x byte bounds. '
                 'Unknown/failed requests retain bounds; errors latch audio transport closed. '
                 'Reject run exposure above USD48 or cumulative USD200; no automatic restart.',
        'limitations': ['Eight known authored cases, one draw each, not a fresh held-out benchmark.',
                        'Server readiness, not browser audible onset; gaps are replay queue estimates.',
                        'Different generation/review policies and variable length; compare delivery and quality jointly.',
                        'Automatic reviews and judges can miss semantic errors.',
                        'Fresh generation varies between arms; first-chunk shortening and fixed-prefix scheduling are separate contrasts. No claim of isolated model-quality causation.',
                        'Length-rewrite helper is Gemma, not the production default gpt-5-mini; price and quality conclusions apply to this configuration.']}
    atomic_json(DIRECTORY / 'manifest_adaptive_speech_v1.json', manifest)
    return manifest


def load_baseline():
    import ouragents
    namespace = dict(vars(ouragents))
    namespace['TreeDebater'] = ouragents.TreeDebater
    exec(compile((DIRECTORY / 'flat_speech_baseline.py').read_text(), 'flat_speech_baseline.py', 'exec'), namespace)
    return namespace['BaselineTreeDebater']


def prepare_pair(case, repeat, client):
    client.label = f'{RUN}/{case["id"]}/{repeat}/preparation'
    start = time.perf_counter()
    player = make_player(case, 'flat_tree', client)
    setup = time.perf_counter() - start
    arrival = ready = 0.0
    timings = []
    for chunk in case['chunks']:
        arrival += max(3., len(chunk.split()) / 2.3)
        player.status = case.get('stage', 'opening')
        start = time.perf_counter()
        player.observe_opponent(chunk, player.oppo_side, player.status)
        work = time.perf_counter() - start
        ready = max(arrival, ready) + work
        timings.append({'arrival_seconds': arrival, 'work_seconds': work, 'worker_ready_seconds': ready})
    history = ([{'stage': 'opening', 'side': player.side, 'content': case['own_opening']}]
               + case.get('prior_history', [])
               + [{'stage': case.get('stage', 'opening'), 'side': player.oppo_side,
                   'content': ' '.join(case['chunks']), 'tree_via_streaming': True}])
    snapshot = {'our_tree': player.debate_tree.get_tree_info(),
                'opponent_tree': player.oppo_debate_tree.get_tree_info(),
                'state': player.planner.state, 'plan': player.planner.plan,
                'setup_seconds': setup, 'chunks': timings,
                'estimated_listener_backlog_seconds': max(0, ready-arrival),
                'model_usage': client.summary(client.label)}
    atomic_json(OUTPUT / case['id'] / str(repeat) / 'preparation.json', snapshot)
    return player, history, snapshot


def run_arm(prepared, history, preparation, case, repeat, arm, client, baseline):
    import openai
    import tts_streaming
    directory = OUTPUT / case['id'] / str(repeat) / arm
    directory.mkdir(parents=True, exist_ok=False)
    player = copy.deepcopy(prepared)
    player.__class__ = baseline
    player.config.streaming_tts = True
    player.streaming_output_config = CONFIGS[arm]
    player.audio_output_dir = str(directory)
    client.label = f'{RUN}/{case["id"]}/{repeat}/{arm}/generation'
    require_room(client, 1)
    audio_label = f'{RUN}/{case["id"]}/{repeat}/{arm}/audio'
    guard = AudioGuard(LEDGER, audio_label, allowance=1)
    sdk_clients = []
    refinement_label = f'{RUN}/{case["id"]}/{repeat}/{arm}/refinement'
    real_openai = openai.OpenAI
    def factory(**kwargs):
        sdk = real_openai(http_client=httpx.Client(transport=guard, timeout=60), max_retries=0,
                          base_url='https://api.openai.com/v1', timeout=60)
        def rewrite(**request):
            if request['model'] != MODEL:
                raise ValueError('Unpriced refinement model blocked')
            text = client.complete_at(refinement_label, request['messages'], temperature=0,
                                      max_tokens=min(1600, request.get('max_completion_tokens', 1600)), model=MODEL)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        sdk.chat.completions.create = rewrite
        sdk_clients.append(sdk)
        return sdk
    old_factory = tts_streaming.OpenAI
    tts_streaming.OpenAI = factory
    events = []
    started = time.perf_counter()
    def emit(index, path, text, duration):
        called_at = time.perf_counter() - started
        decoded = len(AudioSegment.from_file(path)) / 1000
        if decoded <= 0 or abs(decoded-duration) > .1:
            raise ValueError('Emitted audio is empty or differs from callback duration')
        available_at = time.perf_counter() - started
        previous_end = events[-1]['estimated_playback_end_seconds'] if events else available_at
        play_start = max(previous_end, available_at)
        events.append({'index': index, 'path': str(Path(path).relative_to(ROOT)), 'text': text,
                       'callback_seconds': called_at, 'playable_seconds': available_at,
                       'audio_seconds': decoded, 'estimated_gap_seconds': play_start-previous_end,
                       'estimated_playback_start_seconds': play_start,
                       'estimated_playback_end_seconds': play_start+decoded})
        atomic_json(directory / 'events.json', events)
        print(json.dumps({'event': 'audio_ready', 'case': case['id'], 'repeat': repeat,
                          'arm': arm, 'index': index, 'seconds': available_at}), flush=True)
    player.tts_chunk_callback = emit
    error = None
    result = None
    try:
        result = player.rebuttal_generation(history, max_time=60, time_control=True, streaming_tts=True)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    finally:
        ended = time.perf_counter() - started
        for sdk in sdk_clients:
            sdk.close()
        guard.finish()
        tts_streaming.OpenAI = old_factory
    answer = '\n\n'.join(e['text'] for e in events)
    result_record = {'case': case['id'], 'kind': case['kind'], 'repeat': repeat, 'arm': arm,
                     'answer': answer, 'answer_words': len(answer.split()), 'returned_text': result,
                     'error': error, 'status': 'failed_partial' if error and events else
                     'failed_silent' if error else 'returned',
                     'events': events, 'chunks': len(events), 'audio_seconds': sum(e['audio_seconds'] for e in events),
                     'generation_to_first_audio_seconds': events[0]['playable_seconds'] if events else None,
                     'estimated_endpoint_to_first_audio_seconds':
                     preparation['estimated_listener_backlog_seconds'] + events[0]['playable_seconds'] if events else None,
                     'estimated_total_gap_seconds': sum(e['estimated_gap_seconds'] for e in events),
                     'worker_return_seconds': ended, 'generation_usage': client.summary(client.label),
                     'audio_bundle_id': guard.request_id, 'audio_usage': client.summary(audio_label),
                     'audio_calls': guard.artifact['external_calls'], 'thoughts': player.debate_thoughts,
                     'refinement_usage': client.summary(refinement_label),
                     'output_config': asdict(CONFIGS[arm]),
                     'after_state': player.planner.state, 'after_plan': player.planner.plan}
    speech_files = list(directory.glob('*_chunks/speaking.json'))
    if speech_files:
        result_record['speaking_trace'] = json.loads(speech_files[0].read_text())
    prefix_files = list(directory.glob('*_chunks/overlap_prefix.json'))
    if prefix_files:
        result_record['fixed_prefix_trace'] = json.loads(prefix_files[0].read_text())
    import csv
    profile_files = list(directory.glob('*_chunks/chunk_profile.csv'))
    if profile_files:
        result_record['chunk_profiles'] = list(csv.DictReader(profile_files[0].open()))
    atomic_json(directory / 'result.json', result_record)
    print(json.dumps({'event': 'turn_done', 'case': case['id'], 'repeat': repeat, 'arm': arm,
                      'status': result_record['status'], 'chunks': len(events), 'error': error}), flush=True)
    if error and error['type'] == 'BudgetExceeded':
        raise BudgetExceeded(error['message'])
    return result_record


def main():
    raise SystemExit('Archived comparison includes a retired mode; use its saved source snapshot for reproduction.')


if __name__ == '__main__':
    main()
