"""One additional motion using the current production streaming revision path."""
import argparse
from dataclasses import asdict
import json
import hashlib
import re
from pathlib import Path
import sqlite3
import sys

import benchmark_listening_motion_live as live
import compare_revision_pipelines_v56 as accounting

live.RUN = accounting.PARENT + '-streamed-motion03-v70'
live.OUT = live.LEDGER / live.RUN
live.MANIFEST = live.D / ('manifest_' + live.RUN + '.json')
live.MOTION_NUMBER = 3
live.CAP, live.STUDY_CAP, live.COMBINED_CAP = 400., 310., 90.
accounting.GLOBAL_CAP, accounting.PARENT_CAP = 400., 80.
live.arm_study_guard.__defaults__ = (live.STUDY, live.STUDY_CAP)
live.RUN_CAP = 10.
live.CONFIG.first_chunk_seconds = 18.
live.CONFIG.first_body_chunk_seconds = 0.
live.CONFIG.listening_prefix_min_words = 0  # Length is a soft writing target, not a publication gate.
live.CONFIG.listening_prefix_target_words = 50
live.CONFIG.listening_prefix_initial_review_enabled = True
live.CONFIG.listening_prefix_review_enabled = False
live.CONFIG.min_chunk_words = 50
live.VALIDATION_RUNS = tuple(dict.fromkeys((*live.VALIDATION_RUNS, accounting.PARENT + '-streamed-motion02-v68', accounting.PARENT + '-streamed-motion02-v69', live.RUN)))
RECORD = live.D / 'prepared_streamed_motion_v70.json'
original_player = live.new_player
original_combined_guard = live.arm_combined_guard


def arm(db):
    accounting.arm(db, accounting.PARENT, 80.)
    name = 'validation_v70_' + hashlib.sha256(live.RUN.encode()).hexdigest()[:12]
    exposure = live.exposure_query(' WHERE ' + live.combined_filter('c.label'))
    sql = (f'CREATE TRIGGER {name} BEFORE INSERT ON calls '
        f'WHEN {live.combined_filter("NEW.label")} AND NEW.reserved + ({exposure}) > {live.COMBINED_CAP!r} '
        "BEGIN SELECT RAISE(ABORT, 'Combined validation budget exceeded; no dispatch'); END")
    old = db.execute("select sql from sqlite_master where type='trigger' and name=?", (name,)).fetchone()
    if old and old[0] != sql:
        raise ValueError('Existing combined guard differs')
    if not old:
        db.execute(sql); db.commit()
    return name


def new_player(side, motion, meter, players=None):
    player = original_player(side, motion, meter, players)
    original_helper = player.helper_client

    def helper(prompt, sys=None, response_model=None, max_tokens=4096,
               json_mode=None, history_messages=None, **kwargs):
        on_text = kwargs.get('on_text')
        if any(marker in prompt for marker in ('LISTENING PREFIX DRAFT:', 'LISTENING PREFIX REPAIR:')):
            prompt = re.sub(r'The first paragraph should be about \d+ words, maximum \d+;',
                'Aim for about 50 English words in the first paragraph;', prompt)
            rule = ('\nDELIVERY LENGTH FOR THIS EXPERIMENT: '
                'Aim for about 50 English words in the text field, '
                'developing one substantive point in complete sentences. This is a soft target; do not pad to meet a word quota. '
                'Use the existing total speech word budget; the longer opening redistributes '
                'words and does not add time. Preserve the stage rules, including '
                'no new evidence in closing.\n')
            head, separator, data = prompt.rpartition('\n')
            prompt = head + rule + data if separator else rule + prompt
        if on_text is None:
            return original_helper(prompt, sys=sys, response_model=response_model,
                max_tokens=max_tokens, json_mode=json_mode,
                history_messages=history_messages, **kwargs)
        if response_model is not None or json_mode or kwargs.get('model', live.MODEL) != live.MODEL:
            raise ValueError('Streamed revision must be plain text on the approved model')
        if meter.stopped.is_set():
            raise live.BudgetExceeded('Stopped before streamed revision')
        from utils.model import helper_messages
        label = live.RUN + f'/{side}/{player.status}/whole_revision_stream'
        client = live.BudgetedClient(live.LEDGER, cap=live.CAP, label=label)
        try:
            text, _, _ = accounting.stream_complete(client,
                helper_messages(prompt, sys=sys, history_messages=history_messages),
                on_text, max_tokens=min(max_tokens, 4096),
                temperature=kwargs.get('temperature', 0))
            return [text]
        except (live.BudgetExceeded, sqlite3.IntegrityError) as exc:
            meter.stop_for_budget(label, exc)
            raise
        finally:
            client.db.close()

    player.helper_client = helper
    return player


def prepare():
    if RECORD.exists() or live.OUT.exists():
        raise ValueError('Artifacts already exist; no automatic restart')
    assert live.CONFIG.listening_prefix_min_words == 0
    manifest = live.manifest()
    manifest.update(prices_checked_utc='2026-10-10',
        endpoint_evidence_supplement=False,
        listening_evidence_preparation=True,
        first_body_policy='Normal paragraph preference and proportional audio allocation; no short-body prompt or duration ceiling.',
        prefix_policy=dict(audio_target_seconds=18, word_target=50, prompt_word_range=None, minimum_words=0, length_target_is_soft=True),
        duration_policy='Unchanged: FastSpeech estimate, one streamed body revision with deferred whole-body duration fitting, native per-chunk audio fitting.',
        comparison_baselines=['listening-motion-live-v52-refactor-streamed-motion02-v61',
                              'listening-motion-live-v52-refactor-streamed-motion02-v64'])
    manifest['retrieval']['selected_evidence'] = 'Reuse unchanged eligible listening-prepared evidence, or already-selected initial evidence on cold start; no endpoint supplement request.'
    manifest.update(authorization='User requested 按照目前这个设置，再跑一个 motion after reviewing v69 outcome and cost. One further six-turn run on motion03 with the same model/audio/duration settings within the previously approved cumulative parent USD80 budget (including the explicit additional USD30); retain the previously approved USD10 single-run stop. No cap increase and no automatic rerun.',
        parent_cap_usd=80, run_cap_usd=10,
        prior_costs_retained=True, automatic_retries=False,
        method='Six-turn motion03, Gemma 4 26B A4B, streamed body revision, real TTS and Whisper, server-paced playback. Reuse saved evidence pools. No new retrieval.',
        guard='Durable SQLite global400/study310/parent80/combined90/run10 guards before dispatch. Failed and in-flight reservations retained. No automatic rerun.',
        budget_authorization_record=str(live.D / 'authorization_v70.json'),
        implementation_changes=['Version-bound semantic prefix review, up to three format attempts.',
            'Independent prefix TTS and body preparation; ready approved prefix fallback.',
            'Body task uses latest completed same-turn observer context.',
            'Start complete-ASR feedback before waiting for transferred prefix audio; propagate audio failure to duration waiters.',
            'Recheck transferred body without blocking immediately after complete ASR and before immutable task binding.',
            'Current workspace explicitly labels target_index in listening planning contexts and prompts.'],
        launch_wrapper_sha256=live.digest(__file__))
    manifest['estimate_usd'].update(expected=[1,3], run_stop=10,
        parent_cumulative_stop=80, conservative_incremental=10)
    manifest['first_paragraph'].update(semantic_review=True,
        review_policy='Semantic review bound to exact prefix text and context; repairs and replacements must pass review. At most three attempts on malformed review output; cached exact verdicts reused.',
        publication='Ready opening publishes from the listening snapshot without waiting for complete ASR; body still uses complete recognized input.')
    manifest['method'] += ' Version-bound prefix approval and overlapping handoff; body feedback can start before pending prefix TTS finishes.'
    manifest['preflight_accounting_audit'] = str(live.D / 'authorization_v70.json')
    manifest['comparison_baselines'] = [accounting.PARENT + '-streamed-motion02-v69']
    live.atomic_json(RECORD, manifest)
    live.atomic_json(live.MANIFEST, manifest)
    print(json.dumps(dict(motion=manifest['motion'], turns=len(live.TURNS),
        stream_revision=live.CONFIG.listening_stream_body_revision,
        estimate_usd=[1,3], run_cap=10, parent_cap=80), indent=2))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        prepare(); return
    record = json.loads(RECORD.read_text())
    assert record['launch_wrapper_sha256'] == live.digest(__file__)
    assert record['output_config'] == asdict(live.CONFIG)
    assert record['status'] == 'approved' and not live.OUT.exists()
    live.new_player, live.arm_combined_guard = new_player, arm
    sys.argv = [str(Path(live.__file__)), '--execute']
    try:
        live.main()
    finally:
        if live.OUT.exists():
            (live.OUT / 'launch_wrapper.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
