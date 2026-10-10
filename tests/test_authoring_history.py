"""Offline ownership and request regressions for the v50 copied-opponent closing."""
import copy
import json
from pathlib import Path
from types import MethodType

import pytest

from ouragents import TreeDebater
from streaming.body_revision import authoring_history, speech_history
from streaming.body_task import BodyTask
from streaming.listening_prefix import material, prepare
from utils.model import helper_messages
from utils.prompts.authoring import authoring_options, debater_system
from test_debate_prompt_reuse import wire
from test_listening_prefix import PREFIX, TAIL, FRAMEWORK, prepared_player, prefix_helper

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.mark.parametrize('side,opponent', [('for', 'against'), ('against', 'for')])
def test_roles_belong_to_speakers_and_current_asr_replaces_only_its_turn(side, opponent):
    history = [dict(side=opponent, stage='opening', content='Repeated opponent words.'),
               dict(side=side, stage='opening', content='Our delivered case.'),
               dict(side=opponent, stage='rebuttal', content='Repeated opponent words.')]
    original = copy.deepcopy(history)
    messages = speech_history(history, side, opponent_statement='Corrected latest words.',
                              opponent_stage='rebuttal')
    assert [m['role'] for m in messages] == ['user', 'assistant', 'user']
    assert messages[0]['content'].endswith('Repeated opponent words.')
    assert messages[1]['content'] == 'Our delivered case.'
    assert messages[2]['content'] == "**Opponent's rebuttal Statement**\nCorrected latest words."
    assert history == original
    # Equal wording in a genuinely later speech must not erase spoken history.
    later = speech_history(history, side, opponent_statement='Repeated opponent words.',
                           opponent_stage='closing')
    assert len(later) == 4 and "closing Statement" in later[-1]['content']


def test_debater_chat_history_keeps_our_speech_and_excludes_private_prompts():
    history = [dict(role='system', content='Private system'),
               dict(role='user', content='Private writing plan'),
               dict(role='assistant', content='Our delivered opening.'),
               dict(role='user', content="**Opponent's Rebuttal Statement**\nOld ASR.")]
    messages = speech_history(history, 'against', opponent_statement='New ASR.', opponent_stage='rebuttal')
    assert messages == [dict(role='assistant', content='Our delivered opening.'),
                        dict(role='user', content="**Opponent's rebuttal Statement**\nNew ASR.")]


def test_endpoint_uses_final_transcript_instead_of_stale_heard_prefix():
    data = dict(our_side='against', turn='for:closing', endpoint=True,
        heard_transcript='Partial ASR.', final_transcript='Corrected full ASR.',
        debate_history=[dict(side='for', stage='closing', content='Partial ASR.')])
    assert authoring_history(data) == [dict(role='user',
        content="**Opponent's closing Statement**\nCorrected full ASR.")]


def test_helper_owns_history_and_rejects_embedded_system_roles():
    history = [dict(role='assistant', content='Our case.')]
    request = helper_messages('Continue.', sys='System.', history_messages=history)
    history[0]['content'] = 'Later mutation.'
    assert request[1]['content'] == 'Our case.'
    with pytest.raises(ValueError, match='user/assistant'):
        helper_messages('Continue.', history_messages=[dict(role='system', content='Other system')])


@pytest.mark.parametrize('repair', [False, True])
def test_recorded_v50_history_reaches_provider_once_with_our_assistant_turns(tmp_path, wire, repair):
    sent, responder, helper = wire
    player, _, node = prepared_player(tmp_path)
    record = json.loads((Path(__file__).parent/'fixtures/listening_v50_authoring_history.json').read_text())
    data = dict(material(player, 'closing'), **record)
    data.update(current_targets=[], supplied_evidence=[], prepared_rehearsal_materials=[])
    base = prefix_helper(node)
    def reply(*, prompt):
        if repair and prompt.startswith('LISTENING PREFIX DRAFT:'):
            return ['invalid JSON']
        return base(prompt=prompt)
    responder[0] = reply
    prepare(data, helper, player.streaming_output_config, endpoint=True,
            system_prompt=debater_system(player))
    writing = [r for r in sent if r['messages'][-1]['content'].startswith(
        ('LISTENING PREFIX DRAFT:', 'LISTENING PREFIX REPAIR:'))]
    assert len(writing) == (2 if repair else 1)
    for request in writing:
        messages = request['messages']
        assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'user']
        for entry, message in zip(record['debate_history'], messages[1:-1]):
            assert message['content'].endswith(entry['content'])
            assert message['role'] == ('assistant' if entry['side'] == 'against' else 'user')
        assert messages[2]['content'] == record['debate_history'][1]['content']
        latest = record['heard_transcript']
        assert sum(m['content'].count(latest) for m in messages) == 1
        prompt = messages[-1]['content']
        assert 'Your assigned side is against: oppose this exact motion.' in prompt
        assert '**Closing Plan**' not in prompt and '**Statement**' not in prompt
        assert 'Present only the final text' not in prompt
        assert 'debate_history' not in prompt and latest not in prompt
        assert request['wants_json']


@pytest.mark.parametrize('changed', [False, True])
def test_revision_reuse_compares_role_history_even_when_last_opponent_and_prompt_match(tmp_path, wire, changed):
    sent, responder, helper = wire
    player, history, _ = prepared_player(tmp_path)
    history.insert(0, dict(side=player.side, stage='opening', content='Our original position.'))
    task = BodyTask.create(motion=player.motion, side=player.side, stage=player.status,
        history=history, prefix=PREFIX, framework=FRAMEWORK, draft=TAIL)
    feedback = 'Clarify the qualification.'
    offer = dict(prompt=task.revision_prompt(feedback, 60),
        system_prompt=debater_system(player), history_messages=task.authoring_history(),
        writing_options=authoring_options(player), raw='The speculative revised body.', reused=False)
    if changed:
        history[0]['content'] = 'Our corrected position.'
    player.helper_client = helper
    responder[0] = lambda **kwargs: [TAIL]
    result = MethodType(TreeDebater._length_adjust, player)(TAIL, task.guidance(feedback), [],
        task.allocation, 60, max_retry=1, defer_duration_fit=True, frozen_prefix=PREFIX,
        history=history, speculative_revision=offer)
    assert offer['reused'] is (not changed)
    if changed:
        assert result == TAIL and len(sent) == 1
        assert sent[0]['messages'][-1]['content'] == offer['prompt']
        assert sent[0]['messages'][1] == dict(role='assistant', content='Our corrected position.')
        assert history[-1]['content'] not in sent[0]['messages'][-1]['content']
        assert not sent[0]['wants_json']
    else:
        assert result == 'The speculative revised body.' and sent == []
