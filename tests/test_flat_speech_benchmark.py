"""Preflight the real speech harness without any provider requests."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location('flat_speech_benchmark', Path(__file__).resolve().parents[1]
                                             / 'experiments/incremental_planning/benchmark_flat_speech.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_run_guard_blocks_text_dispatch_and_keeps_shared_cap(tmp_path, monkeypatch):
    client = benchmark.ScopedClient(tmp_path)
    client.db.execute("INSERT INTO calls(label,reserved,state) VALUES(?,48,'error')", (benchmark.RUN+'/prior',))
    client.db.commit()
    transport = Mock()
    monkeypatch.setattr(benchmark.BudgetedClient, 'complete', transport)
    with pytest.raises(benchmark.BudgetExceeded):
        client.complete([{'role': 'user', 'content': 'Do not dispatch.'}])
    transport.assert_not_called()
    assert client.summary()['cap_usd'] == 200


def test_shared_preparation_can_be_cloned_without_copying_budget_connection(tmp_path):
    from types import MethodType
    from test_flat_speaking import speaker
    p, _, _, _ = speaker()
    client = benchmark.ScopedClient(tmp_path)
    def response(self, messages, **kwargs):
        return client.label
    p._get_response = MethodType(response, p)
    sibling = copy.deepcopy(p)
    sibling.__class__ = benchmark.load_baseline()
    assert sibling.debate_tree is not p.debate_tree
    assert sibling.planner.state == p.planner.state
    assert sibling.conversation == p.conversation
    assert sibling._get_response.__self__ is sibling
    client.label = 'shared-client'
    assert sibling._get_response([]) == p._get_response([]) == 'shared-client'


def test_frozen_baseline_routes_to_original_full_script_speak(monkeypatch):
    from agents import Debater
    from test_flat_speaking import speaker
    p, history, _, _ = speaker()
    p.__class__ = benchmark.load_baseline()
    p._speak_flat_streaming = Mock(side_effect=AssertionError('Baseline entered incremental speaking'))
    p._get_response.return_value = 'Whole draft.'
    p._get_revision_suggestion = Mock(return_value=('', [], '', 'Whole draft.'))
    p._length_adjust = Mock(return_value='Whole revised speech.')
    post = Mock(return_value='Whole revised speech.')
    monkeypatch.setattr(Debater, 'post_process', post)
    assert p.speak('Speak.', 60, time_control=True, history=history, streaming_tts=True) == 'Whole revised speech.'
    assert post.call_args.args == ('Whole revised speech.', 60, True)
    assert post.call_args.kwargs['streaming_tts'] is True
    p._speak_flat_streaming.assert_not_called()


@pytest.mark.parametrize('format_arg,json_mode', [({}, False), ({'response_format': {'type': 'json_object'}}, True)])
def test_metered_main_client_preserves_requested_json_protocol(monkeypatch, format_arg, json_mode):
    monkeypatch.setenv('DEBATE_LLM_API_BASE', 'http://unused.invalid/v1')
    case = dict(benchmark.cases()[0], prior_history=[])
    client = Mock()
    client.complete.return_value = '{}'
    p = benchmark.make_player(case, 'flat_tree', client)
    p._get_response([{'role': 'user', 'content': 'A test request.'}], **format_arg)
    assert client.complete.call_args.kwargs['json_mode'] is json_mode
