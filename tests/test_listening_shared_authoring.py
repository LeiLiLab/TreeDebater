"""Regression for v46's reversed plan -> independent prefix -> opposite body."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from streaming.body_revision import authoring_sources, authoring_history, draft_prompt
from streaming.branch_planning import branch_prompt, parse_branch_state
from streaming.listening_prefix import PrefixFormatRejected, material, prepare
from utils.llm_schemas import ListeningSelectionResponse
from test_listening_prefix import prepared_player, PREFIX, TAIL, FRAMEWORK

pytestmark = pytest.mark.usefixtures('word_length_modes')
RECORD = json.loads((Path(__file__).parent / 'fixtures/listening_v46_stance_reversal.json').read_text())


def selection():
    return dict(claims=[dict(target=0)], limits=[], overview={k:v for k,v in FRAMEWORK.items() if k != 'position'})


def test_recorded_reversed_tactics_cannot_enter_listening_selection(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    context = player._planning_context()
    bad = RECORD['planning_response']
    assert json.loads(bad)['overview']['position'] == 'against'
    # Correct metadata never authorizes the old free-text argument channels.
    with pytest.raises(ValidationError):
        ListeningSelectionResponse.model_validate_json(bad)
    with pytest.raises(ValueError, match='must not generate speech arguments'):
        parse_branch_state(bad, ' '.join(player.planner.chunks), context)


def test_selection_is_source_bound_and_side_is_owned_by_the_server(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    context = player._planning_context()
    raw = selection()
    ListeningSelectionResponse.model_validate(raw)
    state = parse_branch_state(json.dumps(raw), ' '.join(player.planner.chunks), context)
    assert state['overview']['position'] == player.side
    assert state['rebuttals'] == [] and state['body_plan'] == []
    target = context['tree_targets'][0]
    assert state['claims'][0]['node_id'] == target['node_id']
    assert state['claims'][0]['quote'] == target['sources'][-1]
    assert state['constraints'] == context['constraints']
    for field, value in [('position', 'for'), ('position', 'against')]:
        bad = copy.deepcopy(raw)
        bad['overview'][field] = value
        with pytest.raises(ValueError, match='assigned side'):
            parse_branch_state(json.dumps(bad), ' '.join(player.planner.chunks), context)


def test_previous_reversed_tactics_are_not_recirculated(tmp_path):
    player, _, _ = prepared_player(tmp_path)
    context = player._planning_context()
    previous = json.loads(RECORD['planning_response'])
    request = branch_prompt(context, player.planner.chunks, previous)
    payload = json.loads(request.rsplit('\n', 1)[-1])
    assert 'rebuttals' not in payload['previous_state']
    assert 'body_plan' not in payload['previous_state']
    assert all(t['side'] == player.oppo_side for t in payload['context']['tree_targets'])
    assert payload['heard_prefix'] == player.planner.chunks


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
@pytest.mark.parametrize('side', ['for', 'against'])
def test_recorded_planner_arguments_never_become_authoring_instructions(stage, side):
    data = copy.deepcopy(RECORD['authoring_input'])
    data.update(stage=stage, our_side=side)
    sources = authoring_sources(data)
    assert 'current_plan' not in sources and 'body_plan' not in sources and 'framework' not in sources
    assert sources['our_main_claims'] == data['our_main_claims']
    assert 'content' not in sources['opponent_statement'] and 'debate_history' not in sources
    messages = authoring_history(data)
    assert messages[-1]['role'] == 'user' and messages[-1]['content'].endswith(data['heard_transcript'])
    assert sources['opponent_targets'][0]['sources'] == data['current_targets'][0]['sources']
    for prefix in ('', PREFIX):
        prompt = draft_prompt(data, prefix, 400)
        assert 'Prompt engineering lacks the cognitive resistance necessary for deep mental architecture.' not in prompt
        assert 'Manual writing preserves unique identity against automated mediocrity.' not in prompt
        assert data['heard_transcript'] not in prompt
        assert ('support' if side == 'for' else 'oppose') in prompt


def test_repair_replaces_opening_and_body_as_one_unpublished_draft(tmp_path):
    player, history, _ = prepared_player(tmp_path)
    bad = dict(text='word ' * 90 + 'end.', draft='Obsolete body.', target_ids=[], framework=FRAMEWORK)
    good = dict(text=PREFIX, draft=TAIL, target_ids=[], framework=FRAMEWORK)
    from test_overview_review_gate import passing
    helper = Mock(side_effect=[[json.dumps(bad)], [json.dumps(good)], [json.dumps(passing(PREFIX, 'against'))]])
    candidate = prepare(material(player, player.status, history), helper, player.streaming_output_config)
    assert helper.call_count == 3
    assert candidate['text'] == PREFIX and candidate['body_preparation']['draft'] == TAIL
    assert candidate['body_preparation']['source'] == 'shared_speech_draft'
    repair = helper.call_args_list[1].kwargs['prompt']
    assert 'Obsolete body.' in repair and bad['text'] in repair


def test_incomplete_shared_draft_cannot_publish_a_standalone_prefix(tmp_path):
    player, history, _ = prepared_player(tmp_path)
    helper = Mock(return_value=[json.dumps(dict(text=PREFIX, target_ids=[], framework=FRAMEWORK))])
    with pytest.raises(PrefixFormatRejected):
        prepare(material(player, player.status, history), helper, player.streaming_output_config)
    assert helper.call_count == 2


def test_opponent_target_cannot_be_relabelled_from_our_side():
    data = copy.deepcopy(RECORD['authoring_input'])
    data['current_targets'][0]['side'] = data['our_side']
    with pytest.raises(ValueError, match='owned by our side'):
        authoring_sources(data)
