"""Offline endpoint gates, speculative isolation and real audio/thread ordering."""
import copy
import json
import threading
import time
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("word_length_modes")

from streaming.config import OutputConfig
from streaming.flat_speaking import SegmentRejected
from streaming.listening_prefix import PrefixPreparation, material, next_stage, prepare
from test_flat_speaking import speaker
from test_full_speech import audio as audio


PREFIX = 'With costings first, we oppose this trial on affordability and practical delivery.'
TAIL = 'The remaining argument needs clear funding commitments before launch.'
FRAMEWORK = dict(ready=True, position='Oppose the trial.', core_dispute='Access versus affordability.',
                 response_axes=['Affordability', 'Practical delivery'], prefix_action='keep',
                 reason='The proposal and our response directions are clear.')


@pytest.mark.parametrize('stage,custom,budget', [
    ('opening', False, 522), ('rebuttal', False, 522), ('closing', False, 261),
    ('opening', True, 100), ('rebuttal', True, 200), ('closing', True, 150)])
def test_initial_prewrite_uses_whole_stage_duration(tmp_path, stage, custom, budget):
    p, history, _ = prepared_player(tmp_path)
    if custom:
        p.speech_budgets = dict(opening=46, rebuttal=92, closing=69)
    p.streaming_output_config.listening_body_words = 1  # Old configs cannot override seconds.
    data = material(p, stage, history)
    result = prepare(data, p.helper_client, p.streaming_output_config)
    prompt = p.helper_client.call_args_list[0].kwargs['prompt']
    assert f'length target: approximately {budget} words' in prompt
    assert 'Both fields belong to ONE speech' in prompt
    assert result['text'] == PREFIX and result['body_preparation']['draft'] == TAIL
    assert not result['body_preparation']['needs_fit']


def test_endpoint_override_and_format_repair_keep_the_same_total_budget(tmp_path):
    p, history, node = prepared_player(tmp_path)
    p.speech_budgets = dict(rebuttal=92)
    base = p.helper_client
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            return ['invalid json']
        return base(prompt=prompt, **kwargs)
    checked = Mock(side_effect=helper)
    result = prepare(material(p, 'rebuttal', history), checked,
                     p.streaming_output_config, endpoint=True, max_time=23)
    drafts = [call.kwargs['prompt'] for call in checked.call_args_list
              if call.kwargs['prompt'].startswith(('LISTENING PREFIX DRAFT:', 'LISTENING PREFIX REPAIR:'))]
    assert len(drafts) == 2
    assert all('length target: approximately 50 words' in prompt for prompt in drafts)
    assert result['body_preparation']['draft'] == TAIL


def test_overlong_prefix_gets_one_local_format_repair(tmp_path):
    p, history, node = prepared_player(tmp_path)
    long_text = ' '.join(['word'] * 32) + ' end.'
    base = prefix_helper(node)
    calls = []
    def complete(*, prompt, **kwargs):
        calls.append(prompt)
        if 'LISTENING PREFIX DRAFT:' in prompt:
            return [json.dumps(dict(text=long_text, draft=TAIL, target_ids=[node.node_id], framework=FRAMEWORK))]
        if 'LISTENING PREFIX REPAIR:' in prompt:
            assert prompt_data(prompt)['maximum_word_equivalents'] == 26
            assert 'text exceeds 26 word equivalents' in prompt_data(prompt)['issues']
        return base(prompt=prompt, **kwargs)
    result = prepare(material(p, p.status, history), complete, p.streaming_output_config, endpoint=True)
    assert result['text'] == PREFIX
    assert len(calls) == 3 and 'LISTENING PREFIX REPAIR:' in calls[1]
    assert not result['audits'][0]['accepted'] and result['audits'][-1]['accepted']


def test_underlength_prefix_is_repaired_before_publication(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_prefix_min_words = 20
    p.streaming_output_config.listening_prefix_initial_review_enabled = False
    p.streaming_output_config.listening_prefix_review_enabled = False
    repaired = PREFIX + ' The published cost estimate leaves this proposal without a credible funding path.'
    calls = []
    def complete(*, prompt, **kwargs):
        calls.append(prompt)
        text = PREFIX if len(calls) == 1 else repaired
        if len(calls) == 2:
            assert 'text must contain at least 20 words' in prompt
        return [json.dumps(dict(text=text, draft=TAIL, framework=FRAMEWORK))]
    result = prepare(material(p, p.status, history), complete, p.streaming_output_config, endpoint=True)
    assert result['text'] == repaired and len(calls) == 2
    assert not result['audits'][0]['accepted'] and result['audits'][-1]['accepted']




def test_review_excludes_private_preparation_from_evidence(tmp_path):
    from streaming.overview_review import review
    p, history, node = prepared_player(tmp_path)
    data = material(p, p.status, history)
    for key in ('prior_debate', 'private_claim_options', 'our_main_claims', 'current_plan'):
        data[key] = 'UNVERIFIED_PRIVATE_PREPARATION'
    helper = prefix_helper(node)
    result = review(dict(text=PREFIX, target_ids=[node.node_id]), data, helper, endpoint=True)
    prompt = helper.call_args.kwargs['prompt']
    assert 'UNVERIFIED_PRIVATE_PREPARATION' not in prompt
    assert history[-1]['content'] in prompt and result['accepted']


def conflict(result, data, kind='latest_input', *, source='', reason='Latest proposal invalidates this overview.'):
    result[kind + '_ok'] = False
    result['conflicts'] = [dict(kind=kind,
        basis='invalidated_premise' if kind == 'latest_input' else 'misstated_commitment',
        draft_quote=data['draft'], source_quote=source or data['sources'][0], reason=reason)]


def prompt_data(prompt):
    if 'LISTENING PREFIX REPAIR:' in prompt:
        return json.loads(prompt[prompt.index('{"context":'):])
    return json.loads(prompt.rsplit('\n', 1)[-1])


def prefix_helper(node, *, review_edit=None, prefix=PREFIX, draft=TAIL):
    def complete(*, prompt, **kwargs):
        if 'LISTENING FINAL BODY GATE:' in prompt:
            return [json.dumps(dict(stance_ok=True, attribution_ok=True, conditions_ok=True, issues=[]))]
        if 'LISTENING BODY DRAFT:' in prompt:
            return [json.dumps(dict(draft=draft))]
        if 'LISTENING BODY FEEDBACK:' in prompt or 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            return ['[Critical Issues and Minimal Revision Suggestions]\nRetain the funding qualification.']
        if 'IMMUTABLE SPOKEN PREFIX' in prompt or 'Context and material (data):' in prompt:
            return [draft]
        data = prompt_data(prompt)
        if 'LISTENING PREFIX DRAFT:' in prompt or 'LISTENING PREFIX REPAIR:' in prompt:
            return [json.dumps(dict(text=prefix, draft=draft, target_ids=[node.node_id] if node else [],
                                    framework=data['context'].get('framework') or FRAMEWORK))]
        result = dict(stance_ok=True,
                      stance_assessment=dict(expressed_side=data['context']['our_side'],
                          quote=data['draft'], reason='The supplied opening defends its assigned side.'),
                      ready_to_speak_ok=True,
                      latest_input_ok=True, conflicts=[])
        if review_edit:
            review_edit(result, data)
        return [json.dumps(result)]
    return Mock(side_effect=complete)


def prepared_player(tmp_path):
    p, history, node, _ = speaker()
    p.speculative_speech_safe = True
    p.config.claim_selection_strategy = 'saved_scores'
    p.claim_preparation = {}  # This fixture starts after the common preparation lifecycle.
    # Historical gate regressions explicitly enable the retained review path.
    p.streaming_output_config = OutputConfig(speech_mode='listening_prefix',
        listening_prefix_review_enabled=True,
        first_chunk_seconds=8, budget_mode='audio_duration', adaptive_delivery=True,
        audience_feedback_mode='compact',
        max_refinements=0, early_max_refinements=0, normalize_seams=False,
        speed_adjust_min=1, speed_adjust_max=1)
    p.audio_output_dir = str(tmp_path)
    p.planner.state['overview'] = copy.deepcopy(FRAMEWORK)
    p.helper_client = prefix_helper(node)
    from agents import Audience, AudienceConfig
    p.simulated_audience = [Audience(AudienceConfig())]
    p.use_debate_flow_tree = False
    from agents import Agent
    from ouragents import TreeDebater
    from types import MethodType
    p._prepare_stage_prompt = Mock(wraps=MethodType(TreeDebater._prepare_stage_prompt, p))
    p._get_response = Mock(wraps=MethodType(Agent._get_response, p))
    p._get_revision_suggestion = Mock(return_value=('Keep conditions.', [], '', PREFIX + '\n\n' + TAIL))
    p._length_adjust = Mock(return_value=TAIL)
    p.listen = Mock()
    return p, history, node


def offer_and_wait(p, history):
    preparation = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    p._listening_prefix = preparation
    preparation.offer(material(p, p.status, history))
    deadline = time.monotonic() + 3
    while not any(e['status'] == 'reviewed' for e in preparation.events) and time.monotonic() < deadline:
        time.sleep(.005)
    assert any(e['status'] == 'reviewed' for e in preparation.events)
    return preparation


def trace(tmp_path):
    return json.loads(next(tmp_path.glob('*_chunks/listening_prefix.json')).read_text())


def test_prepared_opening_publishes_while_whole_feedback_is_blocked(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    preparation = offer_and_wait(p, history)
    p._get_response.assert_not_called()
    p._prepare_stage_prompt.assert_not_called()
    assert p.conversation == []
    feedback_started, published = threading.Event(), threading.Event()
    def feedback(*args, **kwargs):
        assert kwargs['frozen_prefix'] == PREFIX
        assert kwargs['statement'] == PREFIX + '\n\n' + TAIL
        feedback_started.set()
        assert published.wait(3), 'Whole feedback blocked the first audio'
        return 'Keep conditions.', [], '', kwargs['statement']
    p._get_revision_suggestion.side_effect = feedback
    query, encoded = audio
    def synthesize(*args, **kwargs):
        assert feedback_started.wait(3)
        return encoded()
    query.side_effect = synthesize
    seen = []
    def emit(index, path, text, duration):
        seen.append(text)
        if index == 0:
            assert text == PREFIX
            assert p._length_adjust.call_count == 0
            published.set()
    p.tts_chunk_callback = emit
    answer = p.rebuttal_generation(history, 60, time_control=True)
    assert answer == '\n\n'.join(seen) and seen == [PREFIX, TAIL]
    assert [m['content'] for m in p.conversation if m['role'] == 'assistant'] == [answer]
    t = trace(tmp_path)
    assert t['status'] == 'completed' and t['cold_prefix'] is False
    assert t['candidate_ready_seconds'] < 0
    assert t['gate_ready_seconds'] < t['chunks'][0]['ready_seconds'] < t['tail_work']['end_seconds']
    assert preparation._thread is None and p._listening_prefix is None
    assert p._length_adjust.call_args.kwargs['frozen_prefix'] == PREFIX










def test_source_change_during_synthesis_blocks_publication(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    query, encoded = audio
    def synthesize(*args, **kwargs):
        p.planner.chunks.append('The earlier proposal has been withdrawn.')
        return encoded()
    query.side_effect = synthesize
    p.tts_chunk_callback = Mock()
    with pytest.raises(SegmentRejected, match='Input changed'):
        p.rebuttal_generation(history, 60, time_control=True)
    p.tts_chunk_callback.assert_not_called()
    assert not list(tmp_path.glob('*_chunks/chunk_*.mp3'))


def test_failed_background_tail_preserves_prefix_without_replaying(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    p._length_adjust.side_effect = RuntimeError('Tail provider failed')
    seen = []
    p.tts_chunk_callback = lambda i, path, text, duration: seen.append(text)
    with pytest.raises(RuntimeError, match='Tail provider failed'):
        p.rebuttal_generation(history, 60, time_control=True)
    assert seen == [PREFIX]
    assert p.conversation[-1] == dict(role='assistant', content=PREFIX)
    assert trace(tmp_path)['committed_text'] == PREFIX
    from pydub import AudioSegment
    assert len(AudioSegment.from_file(next(tmp_path.glob('*.mp3')))) > 0


def test_speculation_coalesces_updates_without_redrafting_stable_overview(tmp_path, monkeypatch):
    p, history, _ = prepared_player(tmp_path)
    started, release = threading.Event(), threading.Event()
    seen = []
    def work(data, helper, config, *, system_prompt, writing_options, author, prompt_builder, initial_review):
        from utils.prompts.authoring import DEFAULT_DEBATER_SYSTEM
        assert system_prompt == DEFAULT_DEBATER_SYSTEM
        seen.append(copy.deepcopy(data))
        started.set()
        assert release.wait(3)
        return dict(text=PREFIX, target_ids=[], framework=FRAMEWORK, stage=data['stage'], turn=data['turn'])
    monkeypatch.setattr('streaming.listening_prefix.prepare', work)
    config = OutputConfig(speech_mode='listening_prefix', listening_prefix_max_updates=2)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, config)
    data = material(p, p.status, history)
    prep.offer(data)
    assert started.wait(3)
    for text in ['Second update.', 'Third update.', 'Latest update.']:
        data['final_transcript'] = text
        prep.offer(data)
    data['final_transcript'] = 'Mutable caller changed again.'
    release.set()
    deadline = time.monotonic() + 3
    while len(prep.events) < 2 and time.monotonic() < deadline:
        time.sleep(.005)
    prep.close()
    assert len(seen) == 1
    assert seen[0]['final_transcript'] == history[-1]['content']
    assert any(e['status'] == 'kept' for e in prep.events)
    assert p.conversation == []


def test_endpoint_does_not_wait_for_unfinished_speculation(tmp_path, monkeypatch):
    p, history, _ = prepared_player(tmp_path)
    started, release = threading.Event(), threading.Event()
    def work(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return dict(text=PREFIX)
    monkeypatch.setattr('streaming.listening_prefix.prepare', work)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    prep.offer(material(p, p.status, history))
    assert started.wait(3)
    assert prep.freeze() is None
    release.set()
    prep.close()
    assert prep._thread is None


def test_observe_schedules_preparation_for_next_stage_without_changing_listener_status(tmp_path):
    p, history, node = prepared_player(tmp_path)
    p.status = 'opening'
    p.planner.observe = Mock()
    p._planning_llm = Mock()
    p._analyze_statement = Mock()
    p.observe_opponent('New point.', 'for', 'opening')
    prep = p._listening_prefix
    deadline = time.monotonic() + 3
    while not prep.events and time.monotonic() < deadline:
        time.sleep(.005)
    prep.close()
    assert any(e['status'] == 'reviewed' for e in prep.events)
    assert prep.freeze()['stage'] == 'opening'
    assert p.status == 'opening' and p.conversation == []
    assert p._get_response.call_count == 1


@pytest.mark.parametrize('first', ['for', 'against'])
def test_stage_schedule_respects_speaking_order(first):
    other = 'against' if first == 'for' else 'for'
    assert next_stage(first, 'opening', first) == 'opening'
    assert next_stage(other, 'opening', first) == 'rebuttal'
    assert next_stage(first, 'rebuttal', first) == 'rebuttal'
    assert next_stage(other, 'rebuttal', first) == 'closing'
    assert next_stage(other, 'closing', first) is None


def test_tail_cannot_repeat_prefix(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    p._length_adjust.return_value = TAIL + ' ' + PREFIX
    seen = []
    p.tts_chunk_callback = lambda i, path, text, duration: seen.append(text)
    with pytest.raises(ValueError, match='repeats'):
        p.rebuttal_generation(history, 60, time_control=True)
    assert seen == [PREFIX]


def test_wrong_stage_candidate_is_not_reused(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    p._listening_prefix._latest['stage'] = 'opening'
    p.rebuttal_generation(history, 60, time_control=True)
    assert trace(tmp_path)['cold_prefix'] is True
    assert 'Wrong stage' in trace(tmp_path)['discard_reason']


def test_two_pass_tail_reviews_see_frozen_prefix_after_statement_header_parsing(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    p.config.single_pass_revision = False
    p._get_response.return_value = '**Plan**\nUnspoken outline.\n**Statement:**\n' + TAIL
    p.rebuttal_generation(history, 60, time_control=True)
    assert p._get_revision_suggestion.call_count == p._length_adjust.call_count == 2
    for call in p._get_revision_suggestion.call_args_list:
        assert call.kwargs['statement'] == PREFIX + '\n\n' + TAIL
        assert call.kwargs['frozen_prefix'] == PREFIX
    assert trace(tmp_path)['tail_work']['whole_feedback_passes'] == 2


def test_real_listen_finalizes_input_before_endpoint_gate(audio, tmp_path):
    from ouragents import TreeDebater
    from types import MethodType
    p, history, node = prepared_player(tmp_path)
    p.status = 'opening'
    offer_and_wait(p, history)
    p.planner.processed = len(p.planner.chunks)
    p.listen = MethodType(TreeDebater.listen, p)
    def edit(result, data):
        if data['endpoint']:
            assert p.planner.finished
            assert data['context']['final_transcript'] == history[-1]['content']
    p.helper_client = prefix_helper(node, review_edit=edit)
    p.opening_generation(history, 60, time_control=True)
    assert trace(tmp_path)['cold_prefix'] is False


def test_first_speaker_cold_start_requires_review_without_opponent_targets(audio, tmp_path):
    p, _, _ = prepared_player(tmp_path)
    p.debate_tree.root.children = []
    p.oppo_debate_tree.root.children = []
    p.planner.chunks, p.planner.state, p.planner.plan, p.planner.turn = [], {}, '', None
    p.helper_client = prefix_helper(None)
    p.opening_generation([], 60, time_control=True)
    t = trace(tmp_path)
    assert t['cold_prefix'] and not t['speculative_candidate_available']
    assert t['final_transcript'] == ''
    assert t['endpoint_reviews'][-1]['accepted']


def test_callback_failure_preserves_opening_and_joins_background(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = offer_and_wait(p, history)
    p.tts_chunk_callback = Mock(side_effect=OSError('Playback disconnected'))
    with pytest.raises(OSError, match='Playback disconnected'):
        p.rebuttal_generation(history, 60, time_control=True)
    assert p.tts_chunk_callback.call_count == 1
    assert p.conversation[-1]['content'] == PREFIX
    assert prep._thread is None
    assert trace(tmp_path)['status'] == 'failed'


def test_listening_delivery_does_not_run_unused_legacy_stage_selection(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    offer_and_wait(p, history)
    p._prepare_stage_prompt.side_effect = AssertionError('Unused legacy prompt construction')
    p.claim_selection = Mock(side_effect=AssertionError('Duplicate claim selection'))
    p.rebuttal_generation(history, 60, time_control=True)
    p._prepare_stage_prompt.assert_not_called()
    p.claim_selection.assert_not_called()
    assert trace(tmp_path)['status'] == 'completed'


@pytest.mark.parametrize('prepared', [False, True])
def test_disabled_prefix_review_skips_model_gate_and_delivers_audio(audio, tmp_path, prepared):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_prefix_review_enabled = OutputConfig().listening_prefix_review_enabled
    assert p.streaming_output_config.listening_prefix_review_enabled is False
    p.streaming_output_config.listening_prefix_initial_review_enabled = False
    base = p.helper_client

    def helper(*, prompt, **kwargs):
        assert 'LISTENING PREFIX REVIEW:' not in prompt
        assert 'ENDPOINT GATE:' not in prompt
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    if prepared:
        offer_and_wait(p, history)
    result = p.rebuttal_generation(history, 60, time_control=True)
    saved = trace(tmp_path)
    assert saved['status'] == 'completed'
    assert saved['prefix_review_enabled'] is False
    assert saved['chunks'][0]['text'] == PREFIX
    assert PREFIX in result and TAIL in result
    assert all(not row.get('semantic_review') for row in saved['endpoint_reviews'])
