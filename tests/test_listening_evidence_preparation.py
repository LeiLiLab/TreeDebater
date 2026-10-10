"""No-network evidence preparation, incremental selection and handoff races."""
import copy
import threading
from unittest.mock import Mock

import pytest
from test_full_speech import audio as audio

pytestmark = pytest.mark.usefixtures("word_length_modes")

from streaming.config import OutputConfig
from streaming.listening_evidence import EvidenceState, ListeningEvidence


def pool():
    return [dict(id='cost', content='Funding affordability budget costs.'),
            dict(id='health', content='Hospitals infection mortality.'),
            dict(id='energy', content='Electricity storage batteries.')]


def data(text):
    return dict(stage='rebuttal', heard_transcript=text, framework={}, body_plan=[])


def test_incremental_selection_never_reoffers_selected_or_used_ids():
    candidates = pool()
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:1])
    state = EvidenceState()
    assert state.select(selector, 'Costs', '', candidates, stage='rebuttal',
                        initial_query='Funding affordability budget costs.') == candidates[:1]
    # Unchanged input and feedback about an already covered need cost no call.
    state.select(selector, 'Costs', 'Explain funding costs.', candidates, stage='rebuttal')
    assert selector.call_count == 1
    result = state.select(selector, 'Costs and health', 'Add hospital infection mortality evidence.',
                          candidates, stage='rebuttal')
    assert [e['id'] for e in result] == ['cost', 'health']
    assert [e['id'] for e in selector.call_args.args[2]] == ['health']
    assert 'Already selected' in selector.call_args.args[1]
    # Committed elsewhere: exclude from both retained and new choices.
    result = state.select(selector, 'Health', 'No changes', candidates[1:], stage='rebuttal')
    assert [e['id'] for e in result] == ['health']
    assert selector.call_count == 2


def test_repeated_no_change_feedback_and_empty_pool_do_not_call():
    state = EvidenceState()
    selector = Mock(return_value=pool()[:1])
    state.select(selector, 'Costs', '', pool(), stage='opening', initial_query='Funding costs')
    for _ in range(3):
        state.select(selector, 'Costs', 'Revision Guidance:\nNo changes.', pool(), stage='opening')
    assert selector.call_count == 1
    assert state.select(selector, '', '', [], stage='opening') == []
    assert state.select(selector, '', '', pool(), stage='closing') == []
    assert selector.call_count == 1


def test_same_id_changed_content_invalidates_old_selection():
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:1])
    state = EvidenceState()
    state.select(selector, 'Costs', '', pool(), stage='opening', initial_query='Funding costs')
    changed = pool()
    changed[0]['content'] = 'Corrected funding costs.'
    result = state.select(selector, 'Costs', 'Check corrected funding costs.', changed, stage='opening')
    assert result[0]['content'] == 'Corrected funding costs.'
    assert selector.call_count == 2


def test_selection_ignores_unknown_and_duplicate_ids_without_trusting_model_content():
    selector = Mock(return_value=[dict(id='cost', content='INVENTED'), dict(id='cost'), dict(id='unknown')])
    state = EvidenceState()
    assert state.select(selector, 'Costs', '', pool(), stage='opening') == pool()[:1]


def test_shortlist_bounds_candidates_without_limiting_selection_rounds():
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:1])
    state = EvidenceState()
    state.select(selector, 'Costs', '', pool(), stage='opening', initial_query='Funding costs')
    many = pool() + [dict(id=f'h{i}', content='infection mortality') for i in range(100)]
    state.select(selector, 'Health', 'infection mortality', many, stage='opening', candidate_limit=12)
    assert len(selector.call_args.args[2]) == 12
    assert 'cost' not in {e['id'] for e in selector.call_args.args[2]}


def test_failed_selection_does_not_approve_need_or_mutate_caller():
    import pytest
    candidates = pool()
    original = copy.deepcopy(candidates)
    def fail(statement, guidance, offered, **kwargs):
        offered[0]['content'] = 'MUTATION'
        raise RuntimeError('service unavailable')
    state = EvidenceState()
    with pytest.raises(RuntimeError):
        state.select(fail, 'Costs', 'funding', candidates, stage='opening')
    assert not state.selected and not state.covered and not state.initialized
    assert candidates == original and state.calls == 1


def test_freeze_does_not_wait_and_cannot_publish_late_result():
    entered, release = threading.Event(), threading.Event()
    def select(statement, guidance, candidates, **kwargs):
        entered.set()
        assert release.wait(3)
        return candidates[:1]
    preparation = ListeningEvidence(select, OutputConfig())
    preparation.offer(data('Funding costs'), pool())
    assert entered.wait(3)
    snapshot = preparation.freeze()
    assert snapshot.selected == [] and snapshot.calls == 1
    assert not preparation.reserve(), 'Frozen listening work must not dispatch'
    assert preparation.reserve(endpoint=True), 'Final-input work remains allowed after freeze'
    release.set()
    preparation.close()
    assert preparation.snapshot().selected == []


def test_queued_input_is_coalesced_and_cache_results_are_detached():
    from test_listening_overview import wait_for
    entered, release = threading.Event(), threading.Event()
    requests = []
    def select(statement, guidance, candidates, **kwargs):
        requests.append(statement)
        if len(requests) == 1:
            entered.set()
            assert release.wait(3)
        return candidates[:1]
    preparation = ListeningEvidence(select, OutputConfig())
    preparation.offer(data('Funding costs'), pool())
    assert entered.wait(3)
    preparation.offer(data('Funding costs infection'), pool())
    preparation.offer(data('Funding costs infection mortality electricity storage'), pool())
    release.set()
    wait_for(lambda: preparation._thread is None)
    snapshot = preparation.snapshot()
    assert len(requests) == 2 and 'electricity' in requests[-1]
    snapshot.selected[0]['content'] = 'MUTATED'
    assert preparation.snapshot().selected[0]['content'] != 'MUTATED'
    preparation.close()


def test_observe_opponent_starts_native_selection_before_endpoint(tmp_path):
    from test_listening_prefix import prepared_player
    from test_listening_overview import wait_for
    p, history, _ = prepared_player(tmp_path)
    p.high_quality_evidence_pool, p.used_evidence = pool(), {'health'}
    p.planner.observe = Mock(return_value={})
    p._start_planning_turn = Mock()
    p.observe_opponent(history[-1]['content'], p.oppo_side, 'rebuttal')
    preparation = p._listening_prefix
    assert preparation.evidence is not None
    wait_for(lambda: preparation.evidence.snapshot().initialized)
    assert {e['id'] for e in preparation.evidence.snapshot().selected} == {'cost', 'energy'}
    assert p.used_evidence == {'health'}
    p.discard_listening_prefix()


def test_final_revision_reuses_listening_selection_without_another_request(tmp_path, monkeypatch, audio):
    import json
    from types import MethodType
    from ouragents import TreeDebater
    from streaming.listening_prefix import material
    from test_listening_prefix import prepared_player, PREFIX, TAIL, trace
    from test_final_input_overlap import ready_handoff
    from test_listening_overview import wait_for
    import tts_streaming
    monkeypatch.setattr(tts_streaming, 'estimate_statement_seconds', lambda *a, **kw: 58.)
    p, history, _ = prepared_player(tmp_path)
    preparation, handoff = ready_handoff(p, history, audio[1])
    p.high_quality_evidence_pool = [dict(id=f'e{i}', content=f'Funding costs source {i}.') for i in range(12)]
    p.used_evidence = {'e0'}
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p.config.temperature, p.config.max_tokens = .3, 2048
    original = p.helper_client
    selections = []
    def helper(**kw):
        prompt = kw['prompt']
        if prompt.startswith('From the provided list of evidence dictionaries'):
            assert '"id": "e0"' not in prompt
            selections.append(prompt)
            return [json.dumps(dict(selected_ids=['e2']))]
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            return ['No changes']
        if kw.get('json_mode') is False:
            assert 'Funding costs source 2.' in prompt
            return [TAIL]
        return original(**kw)
    p.helper_client = Mock(side_effect=helper)
    preparation.evidence = ListeningEvidence(p._select_revision_evidence, p.streaming_output_config)
    preparation.evidence.offer(material(p, p.status, history), p.high_quality_evidence_pool[1:])
    wait_for(lambda: preparation.evidence.snapshot().initialized)
    preparation.evidence.freeze()
    assert len(selections) == 1 and p.used_evidence == {'e0'}
    result = p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
        listening_input_completion=lambda: copy.deepcopy(history),
        listening_recognized_input=lambda: copy.deepcopy(history))
    assert result == PREFIX + '\n\n' + TAIL
    assert len(selections) == 1
    assert p.used_evidence == {'e0', 'e2'}
    assert trace(tmp_path)['prepared_evidence']['selected_ids'] == ['e2']
    preparation.close()


def test_feedback_can_request_an_unmet_need_that_was_already_heard():
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:1])
    state = EvidenceState()
    state.select(selector, 'Costs and health', '', pool(), stage='opening',
                 initial_query='Funding costs infection mortality')
    assert [e['id'] for e in state.selected] == ['cost']
    state.select(selector, 'Costs and health', 'Need infection mortality evidence', pool(), stage='opening')
    assert [e['id'] for e in state.selected] == ['cost', 'health']
    assert [e['id'] for e in selector.call_args.args[2]] == ['health']


def test_corrected_listening_topic_retires_only_affected_choices():
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:2])
    state = EvidenceState()
    state.select(selector, '', '', pool(), stage='opening', initial_query='Funding costs infection mortality')
    assert [e['id'] for e in state.selected] == ['cost', 'health']
    state.select(selector, '', '', pool(), stage='opening', initial_query='infection mortality')
    assert [e['id'] for e in state.selected] == ['health']
    assert selector.call_count == 1


def test_disable_preparation_keeps_original_listening_path(tmp_path):
    from test_listening_prefix import prepared_player
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_prepare_evidence = False
    p.high_quality_evidence_pool, p.used_evidence = pool(), set()
    p.planner.observe = Mock(return_value={})
    p._start_planning_turn = Mock()
    p.observe_opponent(history[-1]['content'], p.oppo_side, 'rebuttal')
    assert p._listening_prefix.evidence is None
    p.discard_listening_prefix()


def test_removed_evidence_is_not_exposed_while_new_selection_is_in_flight():
    from test_listening_overview import wait_for
    entered, release = threading.Event(), threading.Event()
    calls = []
    def select(statement, guidance, candidates, **kwargs):
        calls.append(statement)
        if len(calls) == 2:
            entered.set()
            assert release.wait(3)
        return candidates[:1]
    preparation = ListeningEvidence(select, OutputConfig())
    preparation.offer(data('Funding costs'), pool())
    wait_for(lambda: preparation.snapshot().initialized)
    preparation.offer(data('infection mortality'), pool()[1:])
    assert entered.wait(3)
    assert preparation.snapshot().selected == []
    release.set()
    preparation.close()


def test_cache_cannot_be_reused_across_stage_or_turn():
    preparation = ListeningEvidence(Mock(return_value=[]), OutputConfig())
    preparation.offer(dict(data('cost'), turn='for:opening'), [])
    with pytest.raises(ValueError, match='one stage and listening turn'):
        preparation.offer(dict(data('cost'), turn='for:rebuttal'), [])
    preparation.close()


def test_more_than_three_rounds_and_final_feedback_can_select_new_evidence():
    topics = ['alphatopic', 'betatopic', 'gammatopic', 'deltatopic', 'epsilontopic']
    candidates = [dict(id=str(i), content=topic) for i, topic in enumerate(topics)]
    selector = Mock(side_effect=lambda statement, guidance, offered, **kw: offered[:1])
    preparation = ListeningEvidence(selector, OutputConfig())
    state = EvidenceState()
    for i, topic in enumerate(topics):
        if i == 3:
            preparation.freeze()
        result = state.select(selector, topic, topic, candidates, stage='rebuttal',
                              reserve=lambda: preparation.reserve(endpoint=i >= 3))
        assert [e['id'] for e in result] == [str(j) for j in range(i + 1)]
        assert not ({e['id'] for e in selector.call_args.args[2]} & {str(j) for j in range(i)})
    assert selector.call_count == state.calls == preparation.snapshot().calls == 5
    assert all(event['status'] == 'selected' for event in state.events)
    preparation.close()
