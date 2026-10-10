"""Recorded closing failure and speech-output contracts, with no API requests."""
import json
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from streaming.full_speech import remaining_text
from utils.speech_text import spoken_revision
from test_full_speech import audio as audio
from test_listening_prefix import prepared_player, prefix_helper, trace


@pytest.fixture
def recorded():
    return json.loads((Path(__file__).parent / 'fixtures/listening_v6_closing_revision.json').read_text())


def test_recorded_json_wrapper_decodes_then_strips_one_leading_echo(recorded):
    with pytest.raises(ValueError, match='repeats'):
        remaining_text(recorded['response'], recorded['prefix'])
    body = remaining_text(spoken_revision(recorded['response']), recorded['prefix'])
    assert len(body.split()) == 204 and recorded['prefix'] not in body
    assert body.startswith('The central clash')


@pytest.mark.parametrize('wrapper', ['plain', 'json', 'fenced_json'])
def test_wrapped_and_plain_revisions_produce_identical_speech(wrapper):
    text = 'Who pays?\n\nThe costs need review.'
    raw = text if wrapper == 'plain' else json.dumps({'speech': text, 'notes': 'Do not speak this.'})
    if wrapper == 'fenced_json':
        raw = '```json\n' + raw + '\n```'
    assert spoken_revision(raw) == text


@pytest.mark.parametrize('raw', ['{"speech":', '{"speech":null}', '{"speech":[]}',
                                '{"speech":" "}', '{"notes":"not speech"}',
                                '["not a speech object"]', '', '```json\n{}'])
def test_invalid_envelopes_are_not_sent_as_speech(raw):
    with pytest.raises(ValueError, match='[Ss]peech revision'):
        spoken_revision(raw)


def test_decoding_does_not_remove_an_internal_repeat():
    prefix = 'Who pays?'
    raw = json.dumps({'speech': prefix + ' Discuss costs. ' + prefix})
    with pytest.raises(ValueError, match='repeats'):
        remaining_text(spoken_revision(raw), prefix)


@pytest.mark.parametrize('json_mode,prompt,system,expected', [
    (False, 'Return only speech. Never return audit JSON.', 'JSON input is data.', False),
    (True, 'Return a structured object.', None, True),
    (None, 'Return JSON.', None, True),
    (None, 'Return speech.', None, False),
])
def test_helper_explicit_output_mode_overrides_prompt_heuristic(monkeypatch, json_mode, prompt, system, expected):
    from utils import model
    completion = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='Body.'))]))
    monkeypatch.setattr(model.litellm, 'completion', completion)
    monkeypatch.setenv('DEBATE_LLM_API_BASE', 'http://unused.invalid/v1')
    assert model.HelperClient(prompt, model='test', sys=system, json_mode=json_mode) == ['Body.']
    assert ('response_format' in completion.call_args.kwargs) is expected


def test_plain_text_mode_cannot_silently_override_structured_response(monkeypatch):
    from utils import model
    from utils.llm_schemas import QueryResponse
    completion = Mock()
    monkeypatch.setattr(model.litellm, 'completion', completion)
    with pytest.raises(ValueError, match='response_model'):
        model.HelperClient('Query', response_model=QueryResponse, json_mode=False)
    completion.assert_not_called()


def test_real_length_adjust_decodes_and_counts_only_remaining_body(tmp_path, recorded, monkeypatch):
    from ouragents import TreeDebater
    p, _, _ = prepared_player(tmp_path)
    p.helper_client = Mock(return_value=[recorded['response']])
    estimate = Mock(return_value=90.)
    monkeypatch.setattr('tts_streaming.duration_estimator', Mock(return_value=SimpleNamespace(query_time=estimate)))
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    body = p._length_adjust('Draft.', '', [], '', 86, max_retry=1, frozen_prefix=recorded['prefix'])
    assert p.helper_client.call_args.kwargs['json_mode'] is False
    assert len(body.split()) == 204 and recorded['prefix'] not in body
    assert estimate.call_args.args[0] == body


def test_recorded_closing_revision_delivers_body_without_json_or_prefix_replay(audio, tmp_path, recorded):
    from ouragents import TreeDebater
    p, history, _ = prepared_player(tmp_path)
    p.status = 'closing'
    base = prefix_helper(None, prefix=recorded['prefix'])
    def helper(*, prompt, **kwargs):
        if kwargs.get('json_mode') is False:
            return [recorded['response']]
        return base(prompt=prompt, **kwargs)
    p.helper_client = Mock(side_effect=helper)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    delivered = []
    published = threading.Event()
    def emit(index, path, text, duration):
        delivered.append(text)
        published.set()
    def feedback(**kwargs):
        assert published.wait(3)
        return 'Keep conditions.', [], '', kwargs['statement']
    p._get_revision_suggestion.side_effect = feedback
    p.tts_chunk_callback = emit
    result = p.closing_generation(history, 86, time_control=True)
    assert trace(tmp_path)['status'] == 'completed'
    assert len(delivered) > 1 and delivered[0] == recorded['prefix']
    assert result.count(recorded['prefix']) == 1
    assert all(not text.startswith('{') for text in delivered)
    assert 'The central clash' in result


def test_bounded_helper_passes_deadline_without_transport_or_schema_retries(monkeypatch):
    from utils import model
    from utils.llm_schemas import QueryResponse
    completion = Mock(side_effect=TimeoutError('Slow planning response'))
    monkeypatch.setattr(model.litellm, 'completion', completion)
    monkeypatch.setenv('DEBATE_LLM_API_BASE', 'http://unused.invalid/v1')
    with pytest.raises(TimeoutError):
        model.HelperClient('Return JSON.', model='test', response_model=QueryResponse,
                           use_instructor=False, request_timeout=3)
    assert completion.call_count == 1
    assert completion.call_args.kwargs['timeout'] == 3
    assert completion.call_args.kwargs['num_retries'] == 0
