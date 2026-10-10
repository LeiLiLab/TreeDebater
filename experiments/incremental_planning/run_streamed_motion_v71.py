"""One six-turn FastSpeech run; prepare without calls, execute only after approval."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import run_streamed_motion_v70 as previous

live = previous.live
live.RUN = previous.accounting.PARENT + '-fastspeech-motion03-v71'
live.OUT = live.LEDGER / live.RUN
live.MANIFEST = live.D / ('manifest_' + live.RUN + '.json')
live.VALIDATION_RUNS = (*live.VALIDATION_RUNS, live.RUN)
live.CONFIG.tts_backend = 'fastspeech'
RECORD = live.D / 'prepared_streamed_motion_v71.json'
AUTHORIZATION = live.D / 'authorization_v71.json'


def prepare():
    if RECORD.exists() or live.OUT.exists():
        raise ValueError('Artifacts already exist; no automatic restart')
    manifest = live.manifest()
    manifest.update(
        status='pending_approval',
        authorization='Pending explicit approval of one six-turn FastSpeech debate with USD10 cap.',
        budget_authorization_record=str(AUTHORIZATION),
        parent_cap_usd=80, run_cap_usd=10,
        prior_costs_retained=True, automatic_retries=False,
        method='Same six-turn motion03 as v70: Gemma 4 26B A4B, streamed body revision, local FastSpeech2 LJSpeech plus HiFi-GAN for all audio, native speed, real Whisper and server-paced playback. Reuse saved evidence pools; no new retrieval or paid grading.',
        first_body_policy='Normal paragraph preference and proportional audio allocation; no short-body duration ceiling.',
        prefix_policy=dict(audio_target_seconds=18, word_target=50, minimum_words=0, length_target_is_soft=True),
        duration_policy='Raw FastSpeech duration estimates without the OpenAI 1.11*x-7 calibration; MP3 decoded duration is authoritative. Native speed, no local tempo adjustment.',
        tts_allowance_per_turn_usd=0,
        guard='Durable SQLite global400/study310/parent80/combined90/run10 pre-dispatch guards including existing/failed/in-flight exposure. No OpenAI TTS allowed. No automatic rerun. 45-minute process timeout.',
        comparison_baselines=[previous.accounting.PARENT + '-streamed-motion03-v70'],
        local_compute=dict(host='existing local host', rented_resources=False,
                           cuda_visible_devices='0', cuda_allocator_fraction=.10,
                           torch_num_threads=2, wall_timeout_seconds=2700),
        prices=dict(gemma_input_per_million=.13, gemma_output_per_million=.40,
                    whisper_per_minute=.006, tts_api_calls=0),
        prices_checked_utc=datetime.now(timezone.utc).date().isoformat(),
        price_sources=['https://aws.amazon.com/bedrock/pricing/',
                       'https://developers.openai.com/api/docs/models/whisper-1'],
        estimate_usd=dict(expected=[.3, 1.], conservative_incremental=10., run_stop=10.,
                          parent_cumulative_stop=80., study_stop=310., cumulative_cap=400.,
                          combined_validation_stop=90.,
                          quantities=dict(llm_input_tokens=[1000000,3000000],
                                          llm_output_tokens=[50000,200000], asr_minutes=20),
                          basis='LLM USD0.15-0.47 plus approximately USD0.12 Whisper, rounded upward for variable preparation and rewrites. Existing local host, no provisioned compute/storage charges. Conservative API reservations include retry/unknown-response headroom; stop before exceeding any cap.'),
        launch_wrapper_sha256=live.digest(__file__),
        parent_wrapper_sha256=live.digest(previous.__file__))
    manifest['retrieval']['selected_evidence'] = 'Reuse eligible listening-prepared evidence or initial evidence; no endpoint supplement.'
    manifest['first_paragraph'].update(semantic_review=True,
        publication='Ready reviewed prefix may publish before complete ASR; the body uses complete recognized input.')
    live.atomic_json(RECORD, manifest)
    live.atomic_json(live.MANIFEST, manifest)
    live.atomic_json(AUTHORIZATION, dict(status='pending_approval',
        user_request='跑一次完整的辩论', configuration=str(RECORD),
        expected_usd=[.3,1.], run_cap_usd=10., parent_cap_usd=80.,
        budget_increase=False, prior_costs_retained=True,
        previous_global_exposure_usd=manifest['starting_ledger']['accounted_exposure_usd'],
        approval=None))
    print(json.dumps(dict(status=manifest['status'], motion=manifest['motion'],
        backend=live.CONFIG.tts_backend, turns=live.TURNS,
        expected_usd=[.3,1.], cap_usd=10., manifest=str(RECORD)), indent=2))


def execute():
    record = json.loads(RECORD.read_text())
    authorization = json.loads(AUTHORIZATION.read_text())
    assert record['status'] == authorization['status'] == 'approved'
    assert authorization['approval']
    assert authorization['run_cap_usd'] == live.RUN_CAP == 10.
    assert record['launch_wrapper_sha256'] == live.digest(__file__)
    assert record['parent_wrapper_sha256'] == live.digest(previous.__file__)
    assert record['output_config'] == asdict(live.CONFIG)
    assert not live.OUT.exists()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.10)
    import tts_streaming
    original_query = tts_streaming._query_time_profiled
    def no_openai_tts(*args, **kwargs):
        raise RuntimeError('OpenAI TTS forbidden in FastSpeech validation')
    tts_streaming._query_time_profiled = no_openai_tts
    live.new_player, live.arm_combined_guard = previous.new_player, previous.arm
    sys.argv = [str(Path(live.__file__)), '--execute']
    try:
        live.main()
    finally:
        tts_streaming._query_time_profiled = original_query
        if live.OUT.exists():
            (live.OUT / 'launch_wrapper.py').write_bytes(Path(__file__).read_bytes())
            (live.OUT / 'parent_wrapper.py').write_bytes(Path(previous.__file__).read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    execute() if args.execute else prepare()


if __name__ == '__main__':
    main()
