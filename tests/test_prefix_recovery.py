"""Regression coverage for nonsemantic failures on the first-audio path."""
import copy
import json
import threading
from unittest.mock import Mock

import pytest

from streaming.branch_planning import parse_branch_state, recover_listening_state
from streaming.listening_prefix import (InitialPrefixReview, PrefixPreparation,
    PrefixReviewPending, material, prepare)
from test_listening_prefix import prepared_player, PREFIX, TAIL, FRAMEWORK, trace
from test_overview_review_gate import passing
from test_full_speech import audio as audio
from dataclasses import replace

pytestmark = pytest.mark.usefixtures('word_length_modes')


def test_presentation_is_normalized_without_author_repair(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    bad = '  ' + PREFIX[:-1].replace(' ', '\n', 1) + '  '
    draft = dict(text=bad, draft=TAIL, framework=FRAMEWORK)
    helper = Mock(side_effect=[[json.dumps(draft)], [json.dumps(passing(PREFIX, 'against'))]])
    result = prepare(material(p, p.status, history), helper, p.streaming_output_config)
    assert result['text'] == PREFIX
    assert helper.call_count == 2
    assert result['audits'][0]['normalizations']


def test_review_format_failure_preserves_and_rechecks_same_draft(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    data = material(p, p.status, history)
    draft = dict(text=PREFIX, draft=TAIL, framework=FRAMEWORK)
    helper = Mock(side_effect=[[json.dumps(draft)], ['{}'], ['{}'], ['{}'],
                              [json.dumps(passing(PREFIX, 'against'))]])
    review = InitialPrefixReview()
    with pytest.raises(PrefixReviewPending) as error:
        prepare(data, helper, p.streaming_output_config, initial_review=review)
    assert error.value.candidate['text'] == PREFIX
    assert error.value.candidate['draft'] == TAIL
    assert not review.invalidated(draft)
    result = prepare(data, helper, p.streaming_output_config, initial_review=review,
                     candidate_override=error.value.candidate)
    assert result['text'] == PREFIX
    assert helper.call_count == 5
    assert sum('LISTENING PREFIX DRAFT:' in c.kwargs['prompt'] for c in helper.call_args_list) == 1
    assert not any('LISTENING PREFIX REPAIR:' in c.kwargs['prompt'] for c in helper.call_args_list)


def test_selection_and_overview_recover_independently(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    ctx = p._planning_context()
    ctx['position_limits'] = ['Only during a pilot.']
    overview = {k:v for k,v in FRAMEWORK.items() if k != 'position'}
    raw = dict(claims=[dict(target=0), dict(target=999)], limits=[0,999], overview=overview)
    state = recover_listening_state(json.dumps(raw), ' '.join(p.planner.chunks), ctx)
    assert state['overview']['ready']
    assert len(state['claims']) == len(state['limits']) == 1
    raw['overview'] = dict(overview, reason=None, response_axes=['Valid issue', 42], extra='ignored')
    recovered = recover_listening_state(json.dumps(raw), '', ctx)
    assert recovered['overview']['ready']
    assert recovered['overview']['response_axes'] == ['Valid issue']
    raw['overview'] = {'ready': 'invalid'}
    state = recover_listening_state(json.dumps(raw), ' '.join(p.planner.chunks), ctx)
    assert not state['overview']['ready']
    assert len(state['claims']) == len(state['limits']) == 1
    # Old bindings survive malformed output only while their sources are still active.
    prior = state
    recovered = recover_listening_state('bad json', '', ctx, prior)
    assert len(recovered['claims']) == 1
    ctx['tree_targets'] = []
    recovered = recover_listening_state('bad json', '', ctx, prior)
    assert recovered['claims'] == [] and not recovered['overview']['ready']


def test_many_claims_and_axes_are_valid(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    ctx = p._planning_context()
    node = ctx['tree_targets'][0]
    ctx['tree_targets'] = [dict(copy.deepcopy(node), node_id=f'node-{i}') for i in range(5)]
    overview = {k:v for k,v in FRAMEWORK.items() if k != 'position'}
    overview['response_axes'] = [f'Issue {i}' for i in range(6)]
    state = parse_branch_state(json.dumps(dict(claims=[dict(target=i) for i in range(5)],
        limits=[], overview=overview)), ' '.join(p.planner.chunks), ctx)
    assert len(state['claims']) == 5 and len(state['overview']['response_axes']) == 6


def test_cold_snapshot_first_audio_precedes_final_analysis(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prefix_overlap_final_update=True)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    p._listening_prefix = prep
    data = material(p, p.status, history)
    data['framework'] = None
    prep.offer(data)
    first_audio = threading.Event()
    p.tts_chunk_callback = lambda index,*args: first_audio.set() if index == 0 else None
    def complete():
        assert first_audio.wait(3), 'Cold prefix waited for final analysis'
        return history
    result = p.rebuttal_generation(history, 60, time_control=True,
        listening_input_completion=complete, listening_recognized_input=lambda: history)
    assert result.startswith(PREFIX)
    saved = trace(tmp_path)
    assert saved['cold_prefix'] and saved['overlaps_final_update']
    assert saved['chunks'][0]['ready_seconds'] < saved['final_input_ready_seconds']
