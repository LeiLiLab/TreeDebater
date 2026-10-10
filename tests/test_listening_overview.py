"""Structural preparation, late updates, framing review and concurrent body work."""
import json
import threading
import time
from types import MethodType
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("word_length_modes")

from streaming.branch_planning import branch_prompt, parse_branch_state
from streaming.config import OutputConfig, from_mapping
from streaming.listening_prefix import PrefixPreparation, material, prepare
from streaming.overview_review import review
from streaming.flat_speaking import SegmentRejected
from streaming.overview_planning import parse_overview
from test_claim_constraints import indexed
from test_listening_prefix import (FRAMEWORK, PREFIX, TAIL, prepared_player,
                                   prefix_helper, offer_and_wait, trace, conflict)
from test_full_speech import audio as audio


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def event_count(prep, kind='overview'):
    return sum(e['kind'] == kind for e in prep.events)


def test_framework_not_full_plan_triggers_first_overview(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    data = material(p, p.status, history)
    data['framework'] = dict(FRAMEWORK, ready=False, prefix_action='wait', reason='Position unfinished.')
    prep.offer(data)
    wait_for(lambda: event_count(prep) == 1)
    p.helper_client.assert_not_called()
    data['framework'] = dict(FRAMEWORK, response_axes=[])
    data['current_plan'] = ''
    prep.offer(data)
    wait_for(lambda: prep.peek() is not None)
    prep.close()
    assert prep.freeze()['text'] == PREFIX
    assert not p._get_response.called and not p._prepare_stage_prompt.called
    assert p.conversation == []


def test_many_elaborations_update_body_keep_overview_and_allow_late_rewrite(tmp_path):
    p, history, node = prepared_player(tmp_path)
    p.streaming_output_config.listening_prefix_max_calls = 12
    base = prefix_helper(node)
    changed = 'With external funding required, we oppose the trial on affordability and practical delivery.'
    draft_calls = []
    def helper(*, prompt, **kwargs):
        data = json.loads(prompt.rsplit('\n', 1)[-1])
        if 'LISTENING PREFIX DRAFT:' in prompt:
            draft_calls.append(data)
            current = changed if 'external funding' in kwargs['history_messages'][-1]['content'] else PREFIX
            return prefix_helper(node, prefix=current)(prompt=prompt, **kwargs)
        result = json.loads(base(prompt=prompt, **kwargs)[0])
        if ('LISTENING PREFIX REVIEW:' in prompt and data['draft'] == PREFIX
                and 'external funding' in data['context']['heard_transcript']):
            conflict(result, data)
        return [json.dumps(result)]
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    data = material(p, p.status, history)
    for i in range(7):
        data['heard_transcript'] += f' An additional example number {i}.'
        data['final_transcript'] = data['heard_transcript']
        prep.offer(data)
        wait_for(lambda: event_count(prep) == i + 1)
    assert len(draft_calls) == 1 and prep.peek()['text'] == PREFIX
    data['heard_transcript'] += ' The trial requires external funding.'
    data['final_transcript'] = data['heard_transcript']
    data['framework'] = dict(FRAMEWORK, core_dispute='Conditional external funding and access.',
                             prefix_action='replace', reason='The prerequisite changes our framing.')
    prep.offer(data)
    wait_for(lambda: prep.peek()['text'] == changed)
    prep.close()
    assert len(draft_calls) == 2 and prep._rewrites == 1
    assert prep._calls <= prep.config.listening_prefix_max_calls


def test_planner_change_without_actual_overview_defect_keeps_text(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = offer_and_wait(p, history)
    data = material(p, p.status, history)
    data['framework'] = dict(FRAMEWORK, prefix_action='replace', reason='A different example arrived.')
    prep.offer(data)
    wait_for(lambda: event_count(prep) == 2)
    prep.close()
    assert prep.peek()['text'] == PREFIX and prep._rewrites == 0
    assert sum('LISTENING PREFIX DRAFT:' in c.kwargs['prompt'] for c in p.helper_client.call_args_list) == 1


@pytest.mark.parametrize('new_input', [False, True])
def test_central_opponent_point_updates_valid_prefix_only_with_new_input(tmp_path, audio, new_input):
    """A planner change alone is insufficient; updated text/audio stay bound together."""
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    cfg = p.streaming_output_config
    cfg.listening_prefix_overlap_final_update = True
    cfg.listening_prefix_pre_synthesize = True
    cfg.listening_prefix_max_calls = 24
    changed = 'They propose external funding; we question whether it can sustain the trial.'
    drafts = []
    base = prefix_helper(None)

    def helper(*, prompt, **kwargs):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        if 'LISTENING PREFIX DRAFT:' in prompt:
            drafts.append(payload['context'])
            text = changed if 'previous_speech' in payload['context'] else PREFIX
            return prefix_helper(None, prefix=text)(prompt=prompt, **kwargs)
        if 'LISTENING PREFIX REVIEW:' in prompt:
            assert payload['prefix_handoff'] and not payload['endpoint']
        return base(prompt=prompt, **kwargs)

    prep = PrefixPreparation(p.planner.turn, helper, cfg, lambda *args: encoded(2))
    try:
        data = material(p, p.status, history)
        prep.offer(data)
        wait_for(lambda: prep.prepared_audio(PREFIX, cfg) is not None)
        data['framework'] = dict(FRAMEWORK, core_dispute='Whether external funding can sustain access.',
            prefix_action='replace', reason='New central funding proposal grounds the response.')
        if new_input:
            data['heard_transcript'] += ' We propose external funding for the trial.'
            data['opponent_sources'].append('We propose external funding for the trial.')
        prep.offer(data)
        wait_for(lambda: event_count(prep) == 2)
        expected = changed if new_input else PREFIX
        wait_for(lambda: prep.prepared_audio(expected, cfg) is not None)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff['candidate']['text'] == handoff['audio']['text'] == expected
        assert prep._rewrites == int(new_input)
        assert len(drafts) == 1 + int(new_input)
        if new_input:
            assert drafts[-1]['previous_speech'] == PREFIX + '\n\n' + TAIL
            assert handoff['data']['opponent_sources'][-1] == 'We propose external funding for the trial.'
        calls = prep._calls
        prep.offer(dict(data, heard_transcript='More input after freeze.'))
        assert prep._calls == calls and prep.peek()['text'] == expected
    finally:
        prep.close()


def test_failed_optional_prefix_update_retains_candidate(tmp_path):
    (p, history, _) = prepared_player(tmp_path)
    cfg = p.streaming_output_config
    cfg.listening_prefix_overlap_final_update = True
    cfg.listening_prefix_max_calls = 24
    cfg.listening_prefix_max_rewrites = 1
    base = prefix_helper(None)

    def helper(*, prompt, **kwargs):
        payload = json.loads(prompt.rsplit('\n', 1)[-1])
        if 'LISTENING PREFIX DRAFT:' in prompt and 'previous_speech' in payload['context']:
            raise RuntimeError('Replacement provider unavailable')
        result = json.loads(base(prompt=prompt, **kwargs)[0])
        return [json.dumps(result)]
    prep = PrefixPreparation(p.planner.turn, helper, cfg)
    try:
        data = material(p, p.status, history)
        prep.offer(data)
        wait_for(lambda : event_count(prep) == 1)
        data['heard_transcript'] += ' The trial requires external funding.'
        data['framework'] = dict(FRAMEWORK, core_dispute='Conditional external funding and access.', prefix_action='replace', reason='The funding proposal changes the central clash.')
        prep.offer(data)
        wait_for(lambda : event_count(prep) == 2)
        assert prep._rewrites == 1
        candidate = prep.freeze()
        assert candidate['text'] == PREFIX
    finally:
        prep.close()


def test_unchanged_failed_framework_does_not_repeat_drafting(tmp_path):
    p, history, node = prepared_player(tmp_path)
    def reject(result, data):
        conflict(result, data, 'ready_to_speak', reason='Overview misstates the scope.')
    helper = Mock(return_value=['invalid JSON'])
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    data = material(p, p.status, history)
    prep.offer(data)
    wait_for(lambda: event_count(prep) == 1)
    calls = helper.call_count
    data['final_transcript'] += ' Private formatting change.'
    prep.offer(data)
    wait_for(lambda: event_count(prep) == 2)
    prep.close()
    assert helper.call_count == calls and prep.peek() is None






def test_ordinary_overview_does_not_require_a_condition_checklist(tmp_path):
    p, history, node = prepared_player(tmp_path)
    data = material(p, p.status, history)
    data['condition_candidates'] *= 50
    helper = prefix_helper(node)
    candidate = dict(text=PREFIX, target_ids=[node.node_id], framework=FRAMEWORK)
    assert review(candidate, data, helper, endpoint=True)['accepted']
    payload = json.loads(helper.call_args.kwargs['prompt'].rsplit('\n', 1)[-1])
    assert 'checklist' not in payload and 'final_input_units' not in payload
    assert helper.call_count == 1
    def misleading(result, payload):
        conflict(result, payload, 'ready_to_speak', reason='Draft misrepresents the proposed prerequisite.')
    assert not review(candidate, data, prefix_helper(node, review_edit=misleading), endpoint=True)['accepted']


def test_abstract_overview_can_have_no_specific_opponent_target(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    candidate = prepare(material(p, p.status, history), prefix_helper(None),
                        p.streaming_output_config, endpoint=True)
    assert 'target_ids' not in candidate and candidate['audits'][-1]['accepted']


def test_ready_body_is_reconciled_against_final_input_in_revision(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = offer_and_wait(p, history)
    wait_for(lambda: any(e['status'] == 'body_saved_unreviewed' for e in prep.events))
    assert not p._get_response.called and p.conversation == []
    final = history[-1]['content'] + ' Additional independent evidence is now supplied.'
    history[-1]['content'] = final
    p.planner.chunks = [final]
    updates = event_count(prep)
    prep.offer(material(p, p.status, history))
    wait_for(lambda: event_count(prep) == updates + 1 and prep._body_thread is None)
    assert any(e['status'] == 'waiting_for_words' for e in prep.events)
    p.rebuttal_generation(history, 60, time_control=True)
    p._get_response.assert_not_called()
    assert trace(tmp_path)['body_preparation']['draft'] == TAIL
    assert p._get_revision_suggestion.call_args.kwargs['history'][-1]['content'] == final
    assert p._length_adjust.call_args.kwargs['history'][-1]['content'] == final
    assert p._length_adjust.call_args.args[0] == TAIL
    assert 'Affordability' in p._length_adjust.call_args.args[3]


def test_blocked_body_worker_does_not_delay_first_audio_and_closes(audio, tmp_path):
    p, history, node = prepared_player(tmp_path)
    base = prefix_helper(node)
    body_started, published = threading.Event(), threading.Event()
    def helper(*, prompt, **kwargs):
        if 'LISTENING BODY DRAFT:' in prompt:
            body_started.set()
            assert published.wait(3), 'Endpoint waited for incomplete body preparation'
        return base(prompt=prompt, **kwargs)
    p.helper_client = Mock(side_effect=helper)
    prep = offer_and_wait(p, history)
    from test_listening_body_cadence import add_words
    wait_for(lambda: prep._body_thread is None)
    data = material(p, p.status, history)
    add_words(data, 100)
    prep.offer(data)
    assert body_started.wait(3)
    p.tts_chunk_callback = lambda *args: published.set()
    p.rebuttal_generation(history, 60, time_control=True)
    assert trace(tmp_path)['body_preparation']['draft'] == TAIL
    assert prep._thread is None and prep._body_thread is None
    assert not any('LISTENING BODY FEEDBACK:' in c.kwargs['prompt'] for c in p.helper_client.call_args_list)


def test_all_speculative_requests_share_call_cap_and_endpoint_keeps_working(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_prefix_max_calls = 2
    prep = offer_and_wait(p, history)
    wait_for(lambda: prep._body_thread is None)
    assert prep._calls == 2 and prep.peek()['text'] == PREFIX
    assert not any(c.kwargs['prompt'].startswith('LISTENING BODY DRAFT:') for c in p.helper_client.call_args_list)
    p.rebuttal_generation(history, 60, time_control=True)
    assert trace(tmp_path)['status'] == 'completed'


def test_planning_call_emits_framework_without_needing_complete_points(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    context = p._planning_context()
    assert context['overview_preparation']['stage'] == 'opening'
    raw = json.loads(indexed())
    raw.pop('rebuttals')
    raw['overview'] = {k: v for k, v in dict(FRAMEWORK, response_axes=[]).items() if k != 'position'}
    prompt = branch_prompt(context, p.planner.chunks, p.planner.state)
    assert 'NOT a complete speech plan' in prompt
    p.helper_client = Mock(return_value=[json.dumps(raw)])
    result = p._planning_llm(prompt, 1100)
    assert p.helper_client.call_args.kwargs['max_tokens'] == 1100
    assert p.helper_client.call_args.kwargs['response_model'].__name__ == 'ListeningSelectionResponse'
    state = parse_branch_state(result, ' '.join(p.planner.chunks), context)
    assert state['overview']['ready'] and state['rebuttals'] == []
    with pytest.raises(ValueError):
        parse_overview(dict(FRAMEWORK, ready=False, prefix_action='keep'))


def test_grounded_tail_revision_preserves_overview_structure_and_allocation(tmp_path):
    from ouragents import TreeDebater
    p, history, _ = prepared_player(tmp_path)
    p.helper_client = Mock(return_value=[TAIL])
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p._length_adjust(TAIL, 'Keep the condition.', [], 'First affordability, then delivery.',
                     3.7, max_retry=1, frozen_prefix=PREFIX)
    prompt = p.helper_client.call_args.kwargs['prompt']
    assert 'Only the already spoken prefix is immutable.' in prompt
    assert 'Do not output that prefix or any introduction' in prompt
    assert 'First affordability, then delivery.' in prompt
    assert 'If there is no overview' not in prompt  # The existing prefix is already the introduction.
    assert 'Open with a substantive overview' not in prompt
    p.helper_client.return_value = [json.dumps(dict(points=[], other_corrections=[]))]
    p._get_feedback_from_audience(PREFIX + '\n\n' + TAIL, history, frozen_prefix=PREFIX)
    prompt = p.helper_client.call_args.kwargs['prompt']
    assert 'Review the complete supplied speech' in prompt and 'fixed for audio' in prompt


def test_replaced_overview_discards_body_bound_to_old_promises(audio, tmp_path):
    p, history, node = prepared_player(tmp_path)
    prep = offer_and_wait(p, history)
    wait_for(lambda: any(e['status'] == 'body_saved_unreviewed' for e in prep.events))
    changed = 'With external funding required, we oppose the trial on affordability and delivery.'
    prep._latest['text'] = changed
    p.rebuttal_generation(history, 60, time_control=True)
    assert trace(tmp_path)['fixed_prefix'] == changed
    assert trace(tmp_path)['body_preparation'] is None


def test_new_limits_validate_and_old_config_maps_to_semantic_rewrites():
    assert OutputConfig(listening_prefix_max_rewrites=0).listening_prefix_max_rewrites == 0
    assert from_mapping(OutputConfig, {'listening_prefix_max_updates': 4}).listening_prefix_max_rewrites == 3
    with pytest.raises(ValueError):
        OutputConfig(listening_prefix_max_calls=0)
    with pytest.raises(ValueError):
        OutputConfig(listening_prefix_max_updates=True)




def test_invalid_source_selection_cannot_publish_overview_and_resets_on_next_turn(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    context = p._planning_context()
    raw = dict(claims=[{'target': 9999}], limits=[], rebuttals=[], overview=FRAMEWORK)
    p.planner.observe('Further explanation.', llm=lambda *args: json.dumps(raw),
                      analyze=lambda *args: None, context=lambda: context)
    assert not p.planner.state
    assert p.planner.overview is None
    assert material(p, 'opening')['framework'] is None
    p.planner.start('for:rebuttal')
    assert p.planner.overview is None


@pytest.mark.parametrize('words,budget', [(140, 60), (530, 227), (476, 221)])
def test_no_change_feedback_and_measured_length_skip_rewrite(audio, tmp_path, words, budget):
    p, history, node = prepared_player(tmp_path)
    tail = ' '.join(['reason'] * words) + '.'
    p.config.single_pass_revision = True
    p.helper_client = prefix_helper(node, draft=tail)
    published = threading.Event()
    p.tts_chunk_callback = lambda *args: published.set()
    def feedback(**kwargs):
        assert published.wait(3)
        return 'Revision Guidance:\nNo changes', [], '', tail
    p._get_revision_suggestion.side_effect = feedback
    p.rebuttal_generation(history, budget, time_control=True)
    p._length_adjust.assert_not_called()
    assert trace(tmp_path)['revised_tail'] == tail




def test_concrete_attribution_correction_is_preserved_in_feedback(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    correction = dict(action='qualify', draft_quote='must store documents', opponent_quote='verify identity',
        added_assumption='retaining raw documents is required',
        correction='Describe document storage as a possible implementation risk, not a mandate.')
    raw = '[Critical Issues and Minimal Revision Suggestions]\n' + correction['correction']
    p.helper_client = Mock(return_value=[raw])
    feedback, rows = p._get_feedback_from_audience(PREFIX + '\n\n' + TAIL, history, frozen_prefix=PREFIX)
    assert correction['correction'] in feedback and rows == [raw]
    prompt = p.helper_client.call_args.kwargs['prompt']
    assert 'Core Message Clarity' in prompt and 'Evidence Presentation' in prompt







def test_listening_planning_history_contains_only_spoken_turns(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    spoken = [dict(role='assistant', content='Our delivered opening.'),
              dict(role='user', content="**Opponent's Opening Statement**\nTheir actual speech.")]
    p.conversation = [dict(role='system', content='Long instructions.'),
                      dict(role='user', content='Private draft and allocation.')] + spoken
    context = p._planning_context()
    assert context['prior_debate'] == spoken
    assert len(p.conversation) == 4
    prompt = branch_prompt(context, p.planner.chunks, p.planner.state)
    assert 'Do not generate rebuttals, body_plan' in prompt
    assert p.planner.chunks[0] in prompt
    p.streaming_output_config.speech_mode = 'full_script'
    assert p._planning_context()['prior_debate'] == p.conversation


def test_whole_review_accepts_four_valid_defects_without_speech_point_limit(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    points = [dict(action='qualify', draft_quote=f'Claim{i}', opponent_quote='Actual words',
                   added_assumption='Unstated premise', correction=f'Qualify claim{i}') for i in range(4)]
    raw = '[Critical Issues and Minimal Revision Suggestions]\n' + '\n'.join(p['correction'] for p in points)
    p.helper_client = Mock(return_value=[raw])
    feedback, audit = p._get_feedback_from_audience(PREFIX+'\n\n'+TAIL, history, frozen_prefix=PREFIX)
    assert all(p['correction'] in feedback for p in points)
    assert audit == [raw]
    assert 'Evaluate all dimensions thoroughly' in p.helper_client.call_args.kwargs['prompt']


def test_prepared_audio_is_reused_only_after_final_gate(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_prefix_pre_synthesize = True
    query, encoded = audio
    renderer = Mock(side_effect=lambda text, config: encoded())
    p.listening_prefix_audio_preparer = renderer
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config,
                             p.listening_prefix_audio_preparer)
    p._listening_prefix = prep
    prep.offer(material(p, p.status, history))
    wait_for(lambda: prep.prepared_audio(PREFIX, p.streaming_output_config) is not None)
    prep.close()
    assert renderer.call_count == 1
    p.rebuttal_generation(history, 60, time_control=True)
    t = trace(tmp_path)
    assert t['endpoint_reviews'] == []
    assert t['prepared_audio']['text'] == PREFIX
    assert 'tts_out' not in t['prepared_audio']
    assert all(c.args[1] != PREFIX for c in query.call_args_list)
    assert prep.prepared_audio('Different opening.', p.streaming_output_config) is None
    config = OutputConfig(voice='alloy')
    assert prep.prepared_audio(PREFIX, config) is None




def test_planning_timeout_retains_final_words_and_invalidates_private_notes(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_planning_timeout_seconds = 3
    p.helper_client = Mock(side_effect=TimeoutError('Slow private plan'))
    final_words = 'The proposed trial is now withdrawn; private access remains permitted.'
    p.planner.observe(final_words, llm=p._planning_llm, analyze=Mock(), context=p._planning_context)
    assert final_words in p.planner.instructions()
    assert not p.planner.state
    assert p.planner.processed == len(p.planner.chunks)
    assert any(e['action'] == 'PLANNING_TIMEOUT' for e in p.planner.events)
    assert p.helper_client.call_args.kwargs['request_timeout'] == 3
    assert p.helper_client.call_args.kwargs['use_instructor'] is False
    p.streaming_output_config.listening_planning_timeout_seconds = 0
    with pytest.raises(TimeoutError):
        p._planning_llm('Private plan', 100)




def test_preparatory_review_sees_prior_spoken_concessions(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    previous = dict(role='user', content="**Opponent's Opening Statement**\nWe concede disclosure risk.")
    p.conversation = [dict(role='system', content='Private instructions.'), previous]
    data = material(p, 'closing')
    assert data['debate_history'] == [previous]
    assert data['final_transcript'] == ' '.join(p.planner.chunks)
