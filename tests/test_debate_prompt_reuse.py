"""Inspect model-bound messages, including speculative and repaired speech drafts."""
import copy
import json
from functools import partial
from types import MethodType, SimpleNamespace

import pytest

from agents import DebaterConfig
from ouragents import TreeDebater
from streaming.listening_prefix import PrefixPreparation, material, prepare
from utils import model
from utils.prompts.authoring import DEFAULT_DEBATER_SYSTEM, authoring_options, debater_system, stage_strategy
from utils.prompts.opening import expert_opening_prompt_2, opening_strategy_workflow, opening_strategy_notes
from utils.prompts.rebuttal import (expert_rebuttal_prompt_2, rebuttal_strategy_knowledge,
                                   rebuttal_strategy_workflow, rebuttal_strategy_notes)
from utils.prompts.closing import (expert_closing_prompt_2, closing_strategy_rules,
                                  closing_strategy_workflow, closing_strategy_allocation, closing_strategy_notes)
from test_listening_overview import wait_for
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, prepared_player, prefix_helper, prompt_data

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.fixture
def wire(monkeypatch):
    """Use the real HelperClient message assembly, replacing only the transport."""
    monkeypatch.delenv('DEBATE_LLM_API_BASE', raising=False)
    sent = []
    responder = [None]

    def complete(**kwargs):
        sent.append(copy.deepcopy(kwargs))
        messages = kwargs['messages']
        text = responder[0](prompt=messages[-1]['content'])[0]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    monkeypatch.setattr(model, '_completion_text', complete)
    return sent, responder, partial(model.HelperClient, model='gpt-test', temperature=0)


def assert_native_draft(prompt, stage, *, continuation=True):
    blocks = dict(opening=[opening_strategy_workflow, opening_strategy_notes],
        rebuttal=[rebuttal_strategy_knowledge, rebuttal_strategy_workflow, rebuttal_strategy_notes],
        closing=[closing_strategy_rules, closing_strategy_workflow, closing_strategy_allocation, closing_strategy_notes])
    for block in blocks[stage]:
        assert block.split('{', 1)[0] in prompt
    assert ('IMMUTABLE SPOKEN PREFIX' in prompt) is continuation
    assert '{n_words}' not in prompt


def assert_authoring(request, stage, system):
    messages = request['messages']
    assert messages[0]['role'] == 'system' and messages[-1]['role'] == 'user'
    assert all(m['role'] in ('user', 'assistant') for m in messages[1:-1])
    assert messages[0]['content'] == system
    prompt = messages[-1]['content']
    if prompt.startswith('LISTENING BODY DRAFT:'):
        assert_native_draft(prompt, stage)
    elif 'Context and material (data):' in prompt:
        assert json.loads(prompt.split('Context and material (data):\n', 1)[1])['stage'] == stage
    elif prompt.startswith('LISTENING PREFIX REPAIR:'):
        assert prompt_data(prompt)['context']['stage'] == stage
        assert_native_draft(prompt, stage, continuation=False)
    elif prompt.startswith('LISTENING PREFIX DRAFT:'):
        assert_native_draft(prompt, stage, continuation=False)
    else:
        assert stage_strategy(stage) in prompt
    assert '{{n_words}}' not in prompt
    if len(messages) > 2:
        assert '**Opening Plan**' not in prompt and '**Closing Plan**' not in prompt
        assert '**Rebuttal Plan**' not in prompt and '**Statement**' not in prompt


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_prepared_overview_and_body_send_native_system_and_stage_strategy(tmp_path, wire, stage):
    sent, responder, helper = wire
    p, history, node = prepared_player(tmp_path)
    p.system_prompt = DebaterConfig().system_prompt + '\nCUSTOM_STYLE'
    p.config.model, p.config.temperature = 'gpt-writer', .65
    writing_options = authoring_options(p)
    definition = p.definition = 'Our proposed scope covers private verification, not public names.'
    responder[0] = prefix_helper(node)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config,
                             system_prompt=debater_system(p), writing_options=writing_options)
    writing_options['temperature'] = .95  # The worker retains its own snapshot.
    try:
        prep.offer(material(p, stage, history))
        p.definition = 'A later mutable definition must not alter the prepared snapshot.'
        wait_for(lambda: any(e.get('status') == 'body_saved_unreviewed' for e in prep.events))
    finally:
        prep.close()
    authors = [r for r in sent if r['messages'][-1]['content'].startswith(
        ('LISTENING PREFIX DRAFT:', 'LISTENING BODY DRAFT:'))]
    assert len(authors) == 1
    for request in authors:
        assert_authoring(request, stage, p.system_prompt)
        assert request['wants_json']
        is_prefix = request['messages'][-1]['content'].startswith('LISTENING PREFIX DRAFT:')
        assert request['model_name'] == 'gpt-writer'
        assert request['temperature'] == .65
        if is_prefix:
            assert prompt_data(request['messages'][-1]['content'])['context']['our_definition'] == definition
            assert 'opening and body together' in request['messages'][-1]['content']
    for request in sent:
        if request not in authors:
            assert [m['role'] for m in request['messages']] == ['user']
            if request['messages'][-1]['content'].startswith('LISTENING PREFIX REVIEW:'):
                assert request['model_name'] == 'gpt-test' and request['temperature'] == 0


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
@pytest.mark.parametrize('temperature', [0, .65])
def test_endpoint_overview_repair_keeps_system_and_strategy(tmp_path, wire, stage, temperature):
    sent, responder, helper = wire
    p, history, node = prepared_player(tmp_path)
    p.definition = 'Our definition allows non-document credentials.'
    p.config.model, p.config.temperature = 'gpt-writer', temperature
    base = prefix_helper(node)

    def reply(*, prompt):
        result = base(prompt=prompt)
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            draft = json.loads(result[0]); draft['text'] = 'word ' * 40 + 'end.'
            return [json.dumps(draft)]
        return result

    responder[0] = reply
    prepare(material(p, stage, history), helper, p.streaming_output_config,
            endpoint=True, system_prompt=DEFAULT_DEBATER_SYSTEM, writing_options=authoring_options(p))
    assert len(sent) == 3
    for request in sent[:2]:
        assert_authoring(request, stage, DEFAULT_DEBATER_SYSTEM)
        payload = prompt_data(request['messages'][-1]['content'])
        assert payload['context']['our_definition'] == p.definition
        assert request['model_name'] == 'gpt-writer'
        assert request['temperature'] == temperature
    assert sent[1]['messages'][-1]['content'].startswith('LISTENING PREFIX REPAIR:')


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
@pytest.mark.parametrize('continuation', [False, True])
def test_committed_revision_sends_actual_configured_system_and_stage(tmp_path, wire, stage, continuation):
    sent, responder, helper = wire
    p, history, _ = prepared_player(tmp_path)
    p.status = stage
    p.system_prompt = DEFAULT_DEBATER_SYSTEM + '\nCUSTOM_REVISION_STYLE'
    p.helper_client = helper
    responder[0] = lambda **kw: [TAIL]
    MethodType(TreeDebater._length_adjust, p)(TAIL, 'Preserve the qualification.', [], '', 60,
        max_retry=1, defer_duration_fit=True, frozen_prefix=PREFIX if continuation else '', history=history)
    assert len(sent) == 1
    assert_authoring(sent[0], stage, p.system_prompt)
    assert not sent[0]['wants_json']
    if continuation:
        prompt = sent[0]['messages'][-1]['content']
        assert json.loads(prompt.split('Context and material (data):\n', 1)[1])['already_spoken_prefix'] == PREFIX
        assert ('Do not output, repeat or contradict it, or add another introduction'
                if stage == 'closing' else 'Do not output that prefix or any introduction') in prompt


@pytest.mark.parametrize('stage,legacy,blocks', [
    ('opening', expert_opening_prompt_2, [opening_strategy_workflow, opening_strategy_notes]),
    ('rebuttal', expert_rebuttal_prompt_2, [rebuttal_strategy_knowledge, rebuttal_strategy_workflow,
                                         rebuttal_strategy_notes]),
    ('closing', expert_closing_prompt_2, [closing_strategy_rules, closing_strategy_workflow,
                                        closing_strategy_allocation, closing_strategy_notes]),
])
def test_legacy_and_streaming_share_complete_strategy_blocks(stage, legacy, blocks):
    for block in blocks:
        assert block in legacy and block in stage_strategy(stage)
    assert DEFAULT_DEBATER_SYSTEM == DebaterConfig().system_prompt


def test_explicit_empty_or_custom_system_is_not_overwritten():
    player = SimpleNamespace(config=SimpleNamespace(system_prompt='configured'))
    assert debater_system(player) == 'configured'
    player.system_prompt = ''
    assert debater_system(player) == ''


def test_observation_creates_worker_with_instance_system(tmp_path, wire, monkeypatch):
    from unittest.mock import Mock
    from streaming.listening_prefix import next_stage
    sent, responder, helper = wire
    p, history, node = prepared_player(tmp_path)
    p.system_prompt = DEFAULT_DEBATER_SYSTEM + '\nOBSERVATION_STYLE'
    p.config.model, p.config.temperature = 'gpt-writer', .55
    p.helper_client = helper
    responder[0] = prefix_helper(node)
    monkeypatch.setattr(p, '_start_planning_turn', Mock())
    monkeypatch.setattr(p.planner, 'observe', Mock())
    p.observe_opponent('New input.', history[-1]['side'], history[-1]['stage'])
    prep = p._listening_prefix
    try:
        wait_for(lambda: any(e.get('status') == 'body_saved_unreviewed' for e in prep.events))
    finally:
        prep.close()
    expected_stage = next_stage(history[-1]['side'], history[-1]['stage'])
    authors = [r for r in sent if r['messages'][-1]['content'].startswith(
        ('LISTENING PREFIX DRAFT:', 'LISTENING BODY DRAFT:'))]
    assert len(authors) == 1
    for request in authors:
        assert_authoring(request, expected_stage, p.system_prompt)
        if request['messages'][-1]['content'].startswith('LISTENING PREFIX DRAFT:'):
            assert request['model_name'] == 'gpt-writer' and request['temperature'] == .55
    for request in sent:
        if request['messages'][-1]['content'].startswith('LISTENING PREFIX REVIEW:'):
            assert request['model_name'] == 'gpt-test' and request['temperature'] == 0


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_production_endpoint_and_cold_body_use_system_and_stage(audio, tmp_path, wire, stage):
    sent, responder, helper = wire
    p, history, node = prepared_player(tmp_path)
    p.system_prompt = DEFAULT_DEBATER_SYSTEM + '\nENDPOINT_STYLE'
    p.config.model, p.config.temperature = 'gpt-writer', .45
    p.conversation = [{'role': 'system', 'content': p.system_prompt}]
    p.helper_client = helper
    responder[0] = prefix_helper(node)
    assert TAIL in getattr(p, stage + '_generation')(history, 60, time_control=True)
    authors = [r for r in sent if r['messages'][-1]['content'].startswith('LISTENING PREFIX DRAFT:')]
    assert len(authors) == 1
    assert_authoring(authors[0], stage, p.system_prompt)
    assert authors[0]['model_name'] == 'gpt-writer' and authors[0]['temperature'] == .45
    for request in sent:
        if request['messages'][-1]['content'].startswith('LISTENING PREFIX REVIEW:'):
            assert request['model_name'] == 'gpt-test' and request['temperature'] == 0
    assert p._get_response.call_count == 1
    assert p._prepare_stage_prompt.call_count == 1
    assert p._prepare_stage_prompt.call_args.kwargs['speech_snapshot']['stage'] == stage


def test_listener_body_refresh_inherits_writer_model_sampling_limit_and_system(tmp_path, wire):
    from test_listening_body_cadence import add_words, offer_and_drain
    sent, responder, helper = wire
    p, history, node = prepared_player(tmp_path)
    p.config.model, p.config.temperature, p.config.max_tokens = 'gpt-writer', .7, 64
    p.system_prompt = 'Configured speech style.'
    responder[0] = prefix_helper(node)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config,
        system_prompt=debater_system(p), writing_options=authoring_options(p))
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)
        add_words(data, 100)
        offer_and_drain(prep, data)
    finally:
        prep.close()
    authors = [r for r in sent if r['messages'][-1]['content'].startswith(
        ('LISTENING PREFIX DRAFT:', 'LISTENING BODY DRAFT:'))]
    assert len(authors) == 2
    for request in authors:
        assert request['model_name'] == 'gpt-writer'
        assert request['temperature'] == .7 and request['max_tokens'] == 64
        assert request['messages'][0]['content'] == p.system_prompt
    reviews = [r for r in sent if r['messages'][-1]['content'].startswith('LISTENING PREFIX REVIEW:')]
    assert len(reviews) == 1 and reviews[0]['model_name'] == 'gpt-test'


def test_incremental_speech_keeps_native_style_strategy_and_explicit_writer_options():
    from streaming.flat_speaking import FlatSpeechProducer
    from test_flat_speaking import speaker
    p, history, _, _ = speaker()
    p.system_prompt = 'Custom concise debate style.'
    p.config.model, p.config.temperature, p.config.max_tokens = 'writer', .6, 512
    options = authoring_options(p, temperature=0, max_tokens=128)
    producer = FlatSpeechProducer(p, history, writing_options=options)
    producer.prepare(12)
    request = p._get_response.call_args
    assert request.args[0][0]['content'].startswith(p.system_prompt)
    assert stage_strategy(p.status) in request.args[0][1]['content']
    assert request.kwargs == dict(model='writer', temperature=0, max_tokens=128,
                                  response_format={'type': 'json_object'})
