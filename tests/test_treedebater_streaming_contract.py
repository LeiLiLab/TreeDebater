"""Native extension points, prompt modes, and immutable streaming requests."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from agents import Audience, AudienceConfig, DebaterConfig
from ouragents import TreeDebater
from streaming.body_task import BodyTask
from streaming.config import OutputConfig
from streaming.listening_prefix import material, prepare
from utils.model import CompletionResults
from utils.prompts.others import audience_feedback_rules
from utils.prompts.speech_revision import revision_prompt
from test_listening_prefix import prepared_player, prefix_helper, PREFIX, TAIL, FRAMEWORK

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.mark.parametrize('mode', ['full', 'compact'])
def test_feedback_modes_preserve_enabled_retrieval_and_audience_rules(tmp_path, mode):
    player, history, _ = prepared_player(tmp_path)
    player.streaming_output_config = replace(player.streaming_output_config, audience_feedback_mode=mode)
    player.add_retrieval_feedback = player.use_debate_flow_tree = True
    player._get_retrieval_debate_tree = Mock(return_value=('tree', 'NATIVE_RETRIEVAL_SENTINEL'))
    player.helper_client = Mock(return_value=['No changes'])
    context = material(player, player.status, history)['feedback_context']
    assert context['mode'] == mode and context['retrieval'] == 'NATIVE_RETRIEVAL_SENTINEL'
    result = TreeDebater._get_feedback_from_audience(player, PREFIX + '\n\n' + TAIL,
                                                    history, frozen_prefix=PREFIX)
    player._get_retrieval_debate_tree.assert_called_once_with(include_points=False)
    assert result[0] == 'No changes'
    prompt = player.helper_client.call_args.kwargs['prompt']
    assert 'NATIVE_RETRIEVAL_SENTINEL' in prompt
    assert ('at most THREE' in prompt) is (mode == 'compact')
    assert ('[Comprehensive Analysis]' in prompt) is (mode == 'full')
    for criterion in ('Core Message Clarity', 'Engagement Impact', 'Evidence Presentation', 'Persuasive Elements'):
        assert criterion in audience_feedback_rules and criterion in prompt
    assert 'remaining body only' in prompt
    player.add_retrieval_feedback = False
    assert player._feedback_context(player.status)['retrieval'] == ''


def test_feedback_snapshot_invalidates_reuse_on_policy_and_retrieval_changes(tmp_path):
    player, history, _ = prepared_player(tmp_path)
    arguments = dict(motion=player.motion, side=player.side, stage=player.status,
                     history=history, prefix=PREFIX, framework=FRAMEWORK, draft=TAIL)
    context = player._feedback_context(player.status)
    task = BodyTask.create(**arguments, feedback_context=context)
    context['retrieval'] = 'Newly retrieved human debate.'
    assert not task.same_input(BodyTask.create(**arguments, feedback_context=context))
    context = copy.deepcopy(json.loads(task.context)['feedback_context'])
    context['mode'] = 'full'
    assert not task.same_input(BodyTask.create(**arguments, feedback_context=context))
    assert json.loads(task.context)['feedback_context']['mode'] == 'compact'


def test_feedback_cache_tracks_query_even_when_flat_generation_omits_tree_text(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    player.add_retrieval_feedback = player.use_debate_flow_tree = True
    player._get_retrieval_debate_tree = Mock(return_value=('tree', 'Example'))
    player.debate_tree.get_all_nodes = Mock(return_value=[object()])
    player._retrieval_tree_text = Mock(return_value='Selected argument version one')
    assert player._generation_tree_context() == ('', '')
    player._feedback_context(player.status)
    player._feedback_context(player.status)
    assert player._get_retrieval_debate_tree.call_count == 1
    player._retrieval_tree_text.return_value = 'Selected argument with corrected qualification'
    player._feedback_context(player.status)
    assert player._get_retrieval_debate_tree.call_count == 2


def test_prefix_generation_uses_native_prompt_and_execution_hooks_without_committing(tmp_path):
    player, history, node = prepared_player(tmp_path)
    data = material(player, player.status, history)
    data.update(our_tree='OUR_TREE_SENTINEL', opponent_tree='OPPONENT_TREE_SENTINEL')
    original_builder = player._prepare_stage_prompt
    player._prepare_stage_prompt = Mock(side_effect=lambda *args, **kwargs:
        (original_builder(*args, **kwargs)[0] + '\nCUSTOM_STAGE_RULE', []))
    before = copy.deepcopy((player.conversation, player.debate_thoughts))
    result = prepare(data, player.helper_client, player.streaming_output_config,
                     author=player._authoring_client(), prompt_builder=player._prepare_stage_prompt)
    assert result['text'] == PREFIX and result['body_preparation']['draft'] == TAIL
    assert player._get_response.call_count == player._prepare_stage_prompt.call_count == 1
    prompt = player._get_response.call_args.args[0][-1]['content']
    assert all(value in prompt for value in ('CUSTOM_STAGE_RULE', 'OUR_TREE_SENTINEL', 'OPPONENT_TREE_SENTINEL'))
    assert (player.conversation, player.debate_thoughts) == before


def test_authoring_cost_is_counted_once_even_for_discarded_concurrent_results(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    def complete(**kwargs):
        result = CompletionResults(['Uncommitted text.'])
        result.response_cost = .125
        return result
    player.helper_client = complete
    author = player._authoring_client()
    with ThreadPoolExecutor(4) as executor:
        results = list(executor.map(lambda index: author(prompt=f'Draft {index}', json_mode=False), range(16)))
    assert results == [['Uncommitted text.']] * 16
    assert player.client_cost == 2
    assert player._get_response.call_count == 16
    assert player.conversation == []


def test_budget_rejection_is_not_retried_by_authoring_adapter(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    player.helper_client = Mock(side_effect=RuntimeError('Budget stop before dispatch'))
    with pytest.raises(RuntimeError, match='Budget stop'):
        player._authoring_client()(prompt='Draft', json_mode=False)
    assert player.helper_client.call_count == 1
    assert player.client_cost == 0


def test_isolated_audience_keeps_native_execution_hook_and_accounts_cost():
    calls = []
    class Reviewer(Audience):
        def _get_response(self, messages, **kwargs):
            calls.append(copy.deepcopy(messages))
            return super()._get_response(messages, **kwargs)
    audience = Reviewer(AudienceConfig())
    before = copy.deepcopy(audience.conversation)
    result = CompletionResults(['No changes'])
    result.response_cost = .25
    completion = Mock(return_value=result)
    assert audience.feedback('Evaluate this speech.', isolated=True, completion=completion) == 'No changes'
    assert len(calls) == completion.call_count == 1
    assert audience.conversation == before
    assert audience.client_cost == .25


def test_unsafe_custom_extensions_use_serial_native_path(tmp_path):
    player, history, _ = prepared_player(tmp_path)
    player.speculative_speech_safe = False
    player._prepare_stage_prompt = Mock(return_value=('Custom native prompt.', []))
    player.speak = Mock(return_value='Custom delivered speech.')
    assert not player._listening_prefix_enabled()
    assert player.rebuttal_generation(history, 60, time_control=True) == 'Custom delivered speech.'
    player._prepare_stage_prompt.assert_called_once()
    assert player.speak.call_args.args[0] == 'Custom native prompt.'


@pytest.mark.parametrize('speech_mode', ['full_script', 'listening_prefix'])
@pytest.mark.parametrize('strategy', ['native', 'saved_scores'])
def test_claim_strategy_is_explicit_and_independent_of_streaming(tmp_path, monkeypatch, speech_mode, strategy):
    player, history, _ = prepared_player(tmp_path)
    player.streaming_output_config.speech_mode = speech_mode
    player.config.claim_selection_strategy = strategy
    player.config.claim_pool_limit = 10
    player.claim_preparation = None
    player.definition = 'Existing motion definition.'
    player.use_rehearsal_tree = False
    player.claim_pool = [[dict(claim='First', minimax_search_score=1)],
                         [dict(claim='Second', minimax_search_score=5)]]
    player.build_evidence_pool = Mock()
    native = Mock(return_value=(['First'], [0], dict(mode='choose_main_claims', framework='Native framework.')))
    monkeypatch.setattr('ouragents.build_logic_claims', native)
    TreeDebater.claim_selection(player, history)
    if strategy == 'native':
        assert player.main_claims_content == ['First']
        assert native.call_args.kwargs['context'] == history[-1]['content']
    else:
        assert player.main_claims_content == ['Second', 'First']
        native.assert_not_called()
    player.build_evidence_pool.assert_called_once()


def test_shared_revision_rules_have_one_body_output_contract():
    prompt = revision_prompt(motion='Motion', side='for', stage='rebuttal', statement=TAIL,
        feedback='Clarify funding.', allocation_plan='Funding first.', evidence=[], prefix=PREFIX, n_words=100)
    material = json.loads(prompt.split('Context and material (data):\n', 1)[1])
    assert material['unpublished_draft'] == TAIL and material['already_spoken_prefix'] == PREFIX
    assert material['remaining_word_target'] == 100
    assert 'If there is no overview' not in prompt
    assert 'List ALL your references' not in prompt
    assert prompt.count('Return only natural spoken body text') == 1
    assert 'no headings, bibliography, explanation or separate plan.' in prompt


def test_new_prompt_and_policy_options_validate():
    assert DebaterConfig().claim_selection_strategy == 'native'
    assert OutputConfig().audience_feedback_mode == 'full'
    with pytest.raises(ValueError, match='claim_selection_strategy'):
        DebaterConfig(claim_selection_strategy='implicit')
    with pytest.raises(ValueError, match='audience_feedback_mode'):
        OutputConfig(audience_feedback_mode='shortish')


def test_native_opening_preparation_reconciles_final_history_once(tmp_path):
    player, history, _ = prepared_player(tmp_path)
    player.config.claim_selection_strategy = 'native'
    player.claim_preparation = dict(strategy='native', history=[])
    def selected(final_history):
        player.claim_preparation = dict(strategy='native', history=copy.deepcopy(final_history))
    player.claim_selection = Mock(side_effect=selected)
    player._prepare_speech_claims('opening', history)
    player._prepare_speech_claims('opening', history)
    player.claim_selection.assert_called_once_with(history)
    player._prepare_speech_claims('rebuttal', history + [dict(content='Later argument')])
    assert player.claim_selection.call_count == 1
