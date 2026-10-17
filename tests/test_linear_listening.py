"""Offline linear planning -> prewrite -> real audio publication regressions."""
from dataclasses import replace
import json
import threading
import time
from types import MethodType
from unittest.mock import Mock

import pytest

from ouragents import TreeDebater
from streaming.body_plan import parse_body_plan, resolve_body_plan
from streaming.linear_listening import source_targets
from streaming.listening_prefix import material, source_stamp
from streaming.planning import IncrementalPlanner, PlanningConfig
from utils.speech_context import authoring_sources
from test_full_speech import audio as audio
from test_listening_prefix import FRAMEWORK, PREFIX, TAIL, prepared_player, prefix_helper, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')
MODES = ('linear', 'adaptive_linear')
HEARD = 'A free six-week trial, with independent costings first.'
NOTES = 'PRIVATE_LINEAR_NOTES: acknowledge costings and question the funding commitment.'


def plan(notes=NOTES, *, ready=True):
    overview = {k: v for k, v in FRAMEWORK.items() if k != 'position'}
    overview.update(ready=ready, prefix_action='keep' if ready else 'wait')
    return json.dumps(dict(notes=notes, overview=overview))


def linear_player(tmp_path, mode, *, stage='rebuttal', gate='UPDATE', responses=None, max_updates=24):
    p, _, _ = prepared_player(tmp_path)
    p.status = stage
    p.planner = IncrementalPlanner(PlanningConfig(mode=mode, max_updates=max_updates))
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prepare_evidence=False, listening_prefix_pre_synthesize=False)
    p.listen = MethodType(TreeDebater.listen, p)
    p._analyze_statement = Mock(side_effect=AssertionError('Linear planning must not analyze a tree'))
    base = prefix_helper(None)
    responses = iter(responses) if responses is not None else None

    def complete(prompt, **kwargs):
        if prompt.startswith('Prepare concise private rebuttal notes'):
            assert 'Return ONLY JSON {"notes"' in prompt
            assert 'Do not output overview.position' in prompt
            return [next(responses) if responses is not None else plan()]
        if prompt.startswith('Decide whether new opponent speech'):
            return [json.dumps(dict(action=gate, reason='Test decision'))]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=complete)
    return p


def settle(prep):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        # Body and audio workers may be launched by the overview worker.
        threads = [prep._thread, prep._body_thread, *prep._audio_threads]
        if prep.peek() and not any(t is not None and t.is_alive() for t in threads):
            return
        time.sleep(.005)
    raise AssertionError(prep.events)


def planning_calls(p):
    return [c for c in p.helper_client.call_args_list
            if c.args and c.args[0].startswith('Prepare concise private rebuttal notes')]


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_linear_observer_prepares_and_publishes_each_stage(audio, tmp_path, mode, stage):
    p = linear_player(tmp_path, mode, stage=stage)
    p.observe_opponent(HEARD, 'for', stage)
    prep = p._listening_prefix
    try:
        settle(prep)
        assert len(planning_calls(p)) == 1
        assert p.planner.overview['position'] == p.side
        assert p.planner.plan == NOTES and p.planner.state == {}
        context = p._planning_context()
        assert not {'tree_targets', 'clash_records', 'source_targets'} & context.keys()
        assert p._generation_tree_context() == ('', '')
        drafted = [c.kwargs['prompt'] for c in p.helper_client.call_args_list
                   if c.kwargs.get('prompt', '').startswith('LISTENING PREFIX DRAFT:')]
        assert len(drafted) == 1 and NOTES in drafted[0]
        snapshot = material(p, stage)
        assert NOTES not in json.dumps(authoring_sources(snapshot))
        assert json.dumps(authoring_sources(snapshot)['opponent_targets']).count(HEARD) == 1
        assert snapshot['current_targets'][0]['sources'] == [HEARD]
        history = [dict(side='for', stage=stage, content=HEARD)]
        emitted = []
        p.tts_chunk_callback = lambda i, path, text, duration: emitted.append(text)
        answer = getattr(p, stage + '_generation')(history, 60, time_control=True)
        assert answer == '\n\n'.join(emitted) == PREFIX + '\n\n' + TAIL
        assert trace(tmp_path)['status'] == 'completed'
        assert trace(tmp_path)['cold_prefix'] is False
        assert p.planner.finished and len(planning_calls(p)) == 1
        p._analyze_statement.assert_not_called()
        assert p._listening_prefix is None
    finally:
        prep.close()


@pytest.mark.parametrize('mode,cap', [('adaptive_linear', 24), ('linear', 1)])
def test_wait_or_update_cap_preserves_input_and_finalizes_without_stale_notes(audio, tmp_path, mode, cap):
    p = linear_player(tmp_path, mode, gate='WAIT', max_updates=cap,
                      responses=[plan(), plan('UPDATED_PRIVATE_PLAN: preserve the volunteer condition.')])
    p.observe_opponent(HEARD, 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        settle(prep)
        before = p.helper_client.call_count
        p.observe_opponent('Proceed only if volunteers are available.', 'for', 'rebuttal')
        data = material(p, p.status)
        assert data['planning_pending'] and data['framework'] is None and data['linear_plan'] == ''
        assert p.planner.instructions() == ''
        assert p.helper_client.call_count == before + (mode == 'adaptive_linear')
        assert 'volunteers' in prep.body_context(p.status, p.planner.turn)['heard_transcript']
        assert prep._pending is None and prep._body_pending is None
        history = [dict(side='for', stage='rebuttal', content=' '.join(p.planner.chunks))]
        p.rebuttal_generation(history, 60, time_control=True)
        assert len(planning_calls(p)) == 2 and p.planner.finished
        assert p.planner.plan.startswith('UPDATED_PRIVATE_PLAN')
        assert not material(p, p.status, history)['planning_pending']
        assert 'volunteers' in p._get_revision_suggestion.call_args.kwargs['history'][-1]['content']
        p._analyze_statement.assert_not_called()
    finally:
        prep.close()


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('response', ['not JSON', '{"notes":"invented"}', plan(ready=False)])
def test_unready_or_malformed_plan_waits_then_uses_endpoint_fallback(audio, tmp_path, mode, response):
    p = linear_player(tmp_path, mode, responses=[response])
    p.observe_opponent(HEARD, 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        if prep._thread is not None:
            prep._thread.join(3)
        assert prep.peek() is None
        history = [dict(side='for', stage='rebuttal', content=HEARD)]
        assert p.rebuttal_generation(history, 60, time_control=True).startswith(PREFIX)
        assert trace(tmp_path)['cold_prefix'] is True
        assert len(planning_calls(p)) == 1  # No extra planner repair request.
    finally:
        prep.close()


@pytest.mark.parametrize('mode', MODES)
def test_final_asr_correction_invalidates_source_bindings_and_old_plan(tmp_path, mode):
    p = linear_player(tmp_path, mode, responses=[plan(), plan('Corrected final plan.')])
    p.observe_opponent(HEARD, 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        settle(prep)
        old = material(p, p.status)
        bound = parse_body_plan([dict(axis=0, target=0, issue='Funding',
            action='challenge_support', point='Ask how it is funded.', weight=2)], FRAMEWORK, old['current_targets'])
        corrected = 'No free trial is proposed; costings must precede any decision.'
        # Recognition can finish before the planner catches up. Old notes must
        # already be absent from that complete-input authoring snapshot.
        history = [dict(side='for', stage='rebuttal', content=corrected)]
        intermediate = material(p, p.status, history)
        assert intermediate['linear_plan'] == '' and intermediate['planning_pending']
        p.finalize_opponent(corrected, 'for', 'rebuttal')
        final = material(p, p.status, history)
        assert final['linear_plan'] == 'Corrected final plan.'
        assert source_stamp(old) != source_stamp(final)
        assert old['current_targets'][0]['node_id'] == final['current_targets'][0]['node_id']
        assert old['current_targets'][0]['version'] != final['current_targets'][0]['version']
        assert all(item['target'] is None for item in resolve_body_plan(dict(final, body_plan=bound), FRAMEWORK))
        assert p.planner.chunks == [corrected] and len(planning_calls(p)) == 2
        p._analyze_statement.assert_not_called()
    finally:
        prep.close()


@pytest.mark.parametrize('mode', MODES)
def test_later_ready_plan_starts_preparation_without_tree_analysis(tmp_path, mode):
    p = linear_player(tmp_path, mode, responses=[plan(ready=False), plan()])
    p.observe_opponent('A trial is under consideration.', 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        if prep._thread is not None:
            prep._thread.join(3)
        assert prep.peek() is None
        p.observe_opponent(HEARD, 'for', 'rebuttal')
        settle(prep)
        assert prep.peek()['text'] == PREFIX
        assert len(planning_calls(p)) == 2
        p._analyze_statement.assert_not_called()
    finally:
        prep.close()


@pytest.mark.parametrize('mode', MODES)
def test_bad_update_cannot_republish_old_plan_as_current(tmp_path, mode):
    p = linear_player(tmp_path, mode, responses=[plan(), '{}'])
    p.observe_opponent(HEARD, 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        settle(prep)
        old = material(p, p.status)
        # Private notes affect authoring/reuse even when source text is identical.
        assert source_stamp(old) != source_stamp(dict(old, linear_plan='Different writing plan.'))
        before = p._get_response.call_count
        p.observe_opponent('The trial has been withdrawn.', 'for', 'rebuttal')
        data = material(p, p.status)
        assert data['planning_pending'] and data['framework'] is None and data['linear_plan'] == ''
        assert p.planner.overview is None and p.planner.plan == ''
        assert p._get_response.call_count == before
        assert 'withdrawn' in prep.body_context(p.status, p.planner.turn)['heard_transcript']
        assert any(e['action'] == 'INVALID_STATE' for e in p.planner.events)
    finally:
        prep.close()


@pytest.mark.parametrize('mode', MODES)
def test_linear_handoff_plays_before_final_input_and_finishes_body(audio, tmp_path, mode):
    _, encoded = audio
    p = linear_player(tmp_path, mode)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prefix_overlap_final_update=True, listening_prefix_pre_synthesize=True)
    p.listening_prefix_audio_preparer = lambda *args: encoded(2)
    p.observe_opponent(HEARD, 'for', 'rebuttal')
    prep = p._listening_prefix
    try:
        settle(prep)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff
        published = threading.Event()
        p.tts_chunk_callback = lambda i, *args: published.set() if i == 0 else None
        history = [dict(side='for', stage='rebuttal', content=HEARD)]
        final_history = [dict(history[0], content=HEARD + ' FINAL_INPUT_QUALIFICATION.')]

        def complete():
            assert published.wait(3), 'Prefix waited for final input'
            return final_history

        answer = p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=lambda: final_history)
        assert answer.startswith(PREFIX) and TAIL in answer
        saved = trace(tmp_path)
        assert saved['status'] == 'completed' and saved['overlaps_final_update']
        assert saved['chunks'][0]['ready_seconds'] < saved['final_input_ready_seconds']
        assert 'FINAL_INPUT_QUALIFICATION' in saved['final_transcript']
        assert p.planner.finished and p.planner.chunks == [final_history[0]['content']]
        p._analyze_statement.assert_not_called()
    finally:
        prep.close()


@pytest.mark.parametrize('side,opponent', [('against', 'for'), ('for', 'against')])
def test_source_references_exclude_private_and_own_text_and_keep_newest_spans(side, opponent):
    history = [dict(role='user', content='PRIVATE PREPARATION'),
               dict(role='assistant', content='Our assertion is not opponent evidence.'),
               dict(role='user', content="**Opponent's opening Statement**\nAn earlier claim.")]
    chunks = ['First heard batch.', 'Latest qualification.']
    targets = source_targets(history, side, f'{opponent}:rebuttal', chunks, 2)
    assert [t['sources'][0] for t in targets] == chunks
    assert all(t['side'] == opponent and t['source_kind'] == 'transcript_span' for t in targets)
    more = source_targets(history, side, f'{opponent}:rebuttal', chunks + ['A new point.'], 3)
    assert more[:2] == targets


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('speech_mode', ['incremental', 'overlap_prefix'])
def test_other_speaking_modes_keep_their_existing_restriction(tmp_path, mode, speech_mode):
    p = linear_player(tmp_path, mode)
    p.streaming_output_config = replace(p.streaming_output_config, speech_mode=speech_mode)
    with pytest.raises(ValueError, match='requires planning mode flat_tree'):
        p._listening_prefix_enabled()
