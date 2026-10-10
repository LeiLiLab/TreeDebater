"""No paid requests: guard transactions and real-time audio-before-ASR ordering."""
import importlib.util
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock
import time

import pytest
from pydub import AudioSegment

from streaming.experiment_client import BudgetedClient
from test_experiment_budget import response

SPEC = importlib.util.spec_from_file_location('listening_live_test',
    Path(__file__).resolve().parents[1]/'experiments/incremental_planning/benchmark_listening_motion_live.py')
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


def test_scope_trigger_sees_other_clients_pending_reservations(tmp_path, monkeypatch):
    one = BudgetedClient(tmp_path, cap=200, label='study/one')
    two = BudgetedClient(tmp_path, cap=200, label='study/two')
    live.arm_study_guard(one.db, 'study/', .06)
    def during(*args, **kwargs):
        with pytest.raises(sqlite3.IntegrityError, match='Study budget'):
            two.text('Hi', 10)
        return response()
    http = Mock(side_effect=during)
    monkeypatch.setattr('streaming.experiment_client.urlopen', http)
    one.text('Hi', 10)
    assert http.call_count == 1
    live.arm_study_guard(two.db, 'study/', .06)
    with pytest.raises(ValueError, match='Cannot change'):
        live.arm_study_guard(two.db, 'study/', .07)
    assert one.summary()['calls'] == 1


def test_speculative_budget_stop_keeps_original_cause_without_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(live, 'OUT', tmp_path)
    client = Mock()
    client.complete.side_effect = live.BudgetExceeded('No dispatch: reservation exceeds $270')
    factory = Mock(return_value=client)
    monkeypatch.setattr(live, 'BudgetedClient', factory)
    meter = live.Meter()
    with pytest.raises(live.BudgetExceeded, match='reservation exceeds'):
        meter.complete('for/closing/listening_body_draft', [], max_tokens=2000)
    saved = json.loads((tmp_path/'budget_stop.json').read_text())
    assert saved == meter.budget_stop and not saved['dispatched']
    assert saved['label'].endswith('/for/closing/listening_body_draft')
    assert 'reservation exceeds $270' in saved['error']
    assert meter.stopped.is_set()
    client.db.close.assert_called_once()
    with pytest.raises(live.BudgetExceeded, match='Run stopped'):
        meter.complete('next', [])
    factory.assert_called_once()
    meter.stop_for_budget('later', live.BudgetExceeded('Later failure'))
    assert json.loads((tmp_path/'budget_stop.json').read_text()) == saved


def test_adaptive_text_edits_and_reviews_use_same_gemma_meter():
    meter = SimpleNamespace(complete=Mock(return_value='Expanded speech.'))
    request = live.metered_tts_text_request(meter)
    messages = [{'role': 'user', 'content': 'Preserve the argument.'}]
    result = request(None, live.MODEL, messages, 4096)
    assert result.choices[0].message.content == 'Expanded speech.'
    meter.complete.assert_called_with('tts/length_rewrite', messages, model=live.MODEL,
                                     max_tokens=4096, temperature=0, json_mode=False)
    request(None, live.MODEL, messages, 400, json_mode=True)
    meter.complete.assert_called_with('tts/meaning_review', messages, model=live.MODEL,
                                     max_tokens=400, temperature=0, json_mode=True)
    with pytest.raises(ValueError, match='requires Gemma'):
        request(None, 'gpt-5.6-sol', messages, 400)
    meter.complete.side_effect = live.BudgetExceeded('No room')
    with pytest.raises(live.BudgetExceeded):
        request(None, live.MODEL, messages, 400)


@pytest.mark.parametrize('fatal', [False, True])
def test_full_motion_records_bad_timing_but_still_stops_on_fatal_error(tmp_path, monkeypatch, fatal):
    from dataclasses import replace
    monkeypatch.setattr(live, 'CONFIG', replace(live.CONFIG, listening_prefix_overlap_final_update=False))
    import sys
    import openai
    from debate_tree import Tree
    root = tmp_path/'repo'
    (root/'src/configs').mkdir(parents=True)
    (root/'src/configs/api_key.json').write_text('{}')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(status='approved', source_digest=live.code_digest(),
        harness_sha256=live.digest(live.__file__), audio_guard_sha256=live.digest(live.D/'audio_probe.py'),
        input_sources={}, motion='Test motion.')))
    monkeypatch.setattr(live, 'ROOT', root)
    monkeypatch.setattr(live, 'OUT', tmp_path/'run')
    monkeypatch.setattr(live, 'MANIFEST', manifest)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--execute'])
    monkeypatch.setattr(live, 'BudgetedClient', Mock(return_value=Mock()))
    monkeypatch.setattr(live, 'arm_study_guard', Mock(return_value='offline'))
    monkeypatch.setattr(live, 'arm_combined_guard', Mock(return_value='offline_combined'))
    monkeypatch.setattr(live, 'AudioGuard', Mock(side_effect=lambda *a, **k: Mock()))
    monkeypatch.setattr(live, 'reconcile_success', Mock())
    monkeypatch.setattr(live, 'ledger_summary', Mock(return_value={'pending_calls': 0}))
    players = {}
    def new_player(side, motion, meter, registry):
        player = SimpleNamespace(side=side, status='opening', debate_thoughts=[],
            _retrieve_on_prepared_tree=Mock(return_value=''), discard_listening_prefix=Mock())
        registry[side] = player
        players[side] = player
        return player
    monkeypatch.setattr(live, 'new_player', new_player)
    monkeypatch.setattr(openai, 'OpenAI', Mock())
    # main installs process-scoped network guards; restore them after this offline test.
    import litellm
    monkeypatch.setattr(litellm, 'completion', litellm.completion)
    monkeypatch.setattr(Tree, 'get_most_similar_node', Tree.get_most_similar_node)
    seen = []
    def play(index, *args):
        seen.append(index)
        players['for'].debate_thoughts.append({'played': index})
        # Only ASR and already needed turns are reserved, never all six upfront.
        labels = [call.args[1] for call in live.AudioGuard.call_args_list]
        assert labels == [f'{live.RUN}/asr'] + [f'{live.RUN}/turn_{i}/tts' for i in seen]
        if fatal and index == 1:
            raise RuntimeError('Body gate failed')
        return dict(answer='Our speech.', listener_transcript='Heard speech.',
            playback_endpoint_monotonic=100.+index, audio_seconds=200.,
            generation_to_first_audio_seconds=12., endpoint_to_first_audio_seconds=12.,
            playback_gap_seconds=3.08, listener_backlog_seconds=0.,
            playback=[dict(gap_seconds=3.08)], signed_duration_error_seconds=-40.)
    monkeypatch.setattr(live, 'play_turn', play)
    if fatal:
        with pytest.raises(RuntimeError, match='Body gate failed'):
            live.main()
    else:
        live.main()
    result = json.loads(manifest.read_text())
    assert seen == ([0, 1] if fatal else list(range(6)))
    assert result['status'] == ('stopped' if fatal else 'completed')
    assert result['completed_turns'] == (1 if fatal else 6)
    thoughts = result['debate_thoughts']
    assert thoughts['for'] == [{'played': i} for i in seen]
    assert thoughts['against'] == []
    rows = json.loads((tmp_path/'run/results.json').read_text())
    assert rows[-1]['debate_thoughts']['for'] == [{'played': i} for i in range(len(rows))]
    assert all(row['timing_pass'] is False for row in rows)
    if not fatal:
        assert result['timing_pass'] is False


def fake_turn(tmp_path, monkeypatch, *, fail_asr=False, min_text_words=1, transcripts=None):
    from dataclasses import replace
    monkeypatch.setattr(live, 'INPUT_CONFIG', replace(live.INPUT_CONFIG, min_text_words=min_text_words))
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    monkeypatch.setattr(live, 'OUT', tmp_path)
    speaker = SimpleNamespace(side='for')
    listener = SimpleNamespace(side='against', observe_opponent=Mock())
    encoded = tmp_path/'encoded.mp3'
    AudioSegment.silent(duration=100).export(encoded, format='mp3')
    audio_bytes = encoded.read_bytes()
    def generate(history, **kwargs):
        folder = Path(speaker.audio_output_dir)/'test_chunks'
        folder.mkdir()
        for i, text in enumerate(['Opening.', 'Tail.'] if transcripts is None else transcripts):
            path = folder/f'chunk_{i:03}.mp3'
            path.write_bytes(audio_bytes)
            speaker.tts_chunk_callback(i, path, text, .1)
        (folder/'listening_prefix.json').write_text(json.dumps(dict(
            fixed_prefix='Opening.', revised_tail='Tail.', speculative_candidate=None)))
        return 'Opening.\n\nTail.'
    speaker.opening_generation = generate
    called = []
    def transcribe(**kwargs):
        called.append(time.perf_counter())
        if fail_asr:
            raise RuntimeError('ASR unavailable')
        return SimpleNamespace(text=f'Heard slice{len(called)}.' if transcripts is None else transcripts[len(called)-1])
    asr = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=transcribe)))
    return speaker, listener, asr, called


def test_playback_precedes_asr_and_full_audio_is_measured(tmp_path, monkeypatch):
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch)
    start = time.perf_counter()
    row = live.play_turn(0, speaker, listener, [], 240, None,
                        SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter())
    assert row['status'] == 'completed' and row['audio_seconds'] == pytest.approx(.2)
    assert row['endpoint_to_first_audio_seconds'] is None
    assert row['signed_duration_error_seconds'] == pytest.approx(-239.8)
    assert len(calls) == 2 and calls[0] - start >= .1 and calls[1] - start >= .2
    assert listener.observe_opponent.call_count == 2
    assert len(AudioSegment.from_file(tmp_path/row['heard'][0]['audio'])) == 100
    for item in row['heard']:
        assert item['worker_start_monotonic'] >= item['heard_end_monotonic']
    assert row['generation_finished_monotonic'] < row['playback_endpoint_monotonic']


def test_failed_asr_preserves_failure_and_latches_dispatch_closed(tmp_path, monkeypatch):
    speaker, listener, asr, calls = fake_turn(tmp_path, monkeypatch, fail_asr=True)
    meter = live.Meter()
    with pytest.raises(RuntimeError):
        live.play_turn(0, speaker, listener, [], 240, None,
                       SimpleNamespace(artifact={'blocked_dispatches': []}), asr, meter)
    assert meter.stopped.is_set() and len(calls) == 1
    row = json.loads((tmp_path/'00_opening_for/result.json').read_text())
    assert row['status'] == 'failed_partial'
    assert row['heard'][0]['error'] == 'RuntimeError: ASR unavailable'


def test_live_helper_respects_explicit_plain_speech_mode(tmp_path, monkeypatch):
    from ouragents import TreeDebater
    player = SimpleNamespace(status='closing', simulated_audience=[], claim_preparation={'claims': []},
                             use_retrieval=True, use_rehearsal_tree=True,
                             main_claims_content=[], high_quality_evidence_pool=[], evidence_pool=[],
                             _add_message=Mock(), build_evidence_pool=Mock(),
                             claim_generation=Mock(), _warm_rehearsal_indexes=Mock(), definition='Motion.',
                             rehearsal_claim_pool=[[dict(claim='Low saved case.', minimax_search_score=1)],
                                                   [dict(claim='High saved case.', minimax_search_score=4)]])
    monkeypatch.setattr('ouragents.TreeDebater', Mock(return_value=player))
    monkeypatch.setattr(live, 'LEDGER', tmp_path)
    monkeypatch.setattr(live, 'OUT', tmp_path)
    monkeypatch.setattr(live, 'REHEARSAL_POOL_DIR', tmp_path)
    for side in ('for', 'against'):
        (tmp_path/f'motion._pool_{side}.json').write_text('[]')
    claims = tmp_path/'flat-motion-overlap-v2'
    claims.mkdir()
    (claims/'motion_01_for_claims.json').write_text(json.dumps(dict(definition='Motion.', claims=[])))
    meter = SimpleNamespace(complete=Mock(return_value='Body.'))
    p = live.new_player('for', 'Motion.', meter)
    meter.complete.assert_not_called()
    p.claim_generation.assert_called_once_with(4)
    saved = json.loads((tmp_path/'claims_for.json').read_text())
    assert saved == p.claim_preparation
    assert p.helper_client(prompt='Return speech, never audit JSON.', json_mode=False) == ['Body.']
    assert meter.complete.call_args.kwargs['json_mode'] is False
    p.helper_client(prompt='Return JSON.')
    assert meter.complete.call_args.kwargs['json_mode'] is True
    history_messages = [dict(role='assistant', content='Our delivered argument.'),
                        dict(role='user', content="**Opponent's closing Statement**\nTheir reply.")]
    p.helper_client(prompt='Write our closing.', sys='Debater system.', history_messages=history_messages,
                    json_mode=False)
    assert meter.complete.call_args.args[1] == [dict(role='system', content='Debater system.'),
        *history_messages, dict(role='user', content='Write our closing.')]

    with pytest.raises(ValueError, match='requires Gemma'):
        p.helper_client(prompt='LISTENING BODY FEEDBACK: Return JSON.', model='gpt-5.6-sol', max_tokens=400)
    p.helper_client(prompt='LISTENING BODY FEEDBACK: Return JSON.', max_tokens=400)
    assert meter.complete.call_args.kwargs['model'] == live.MODEL
    assert meter.complete.call_args.kwargs['max_tokens'] == 400
    assert p.speculative_speech_safe
    p.side, p.config = 'for', SimpleNamespace(model=live.MODEL)
    meter.complete.reset_mock()
    assert TreeDebater._authoring_client(p)(prompt='Write our closing.', json_mode=False,
        history_messages=history_messages, max_tokens=400) == ['Body.']
    meter.complete.assert_called_once()
    assert meter.complete.call_args.kwargs['max_tokens'] == 400
    assert meter.complete.call_args.kwargs['json_mode'] is False
    assert meter.complete.call_args.args[1][:-1] == history_messages
