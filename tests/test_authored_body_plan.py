"""Writer-owned plans survive existing authoring and publication without extra calls."""
import copy
import json
from unittest.mock import Mock

import pytest

from streaming.body_plan import read_author_plan, prepared_body_plan
from streaming.listening_prefix import PrefixPreparation, material, prepare
from streaming.listening_evidence import query
from test_listening_prefix import prepared_player, FRAMEWORK, TAIL, trace
from test_full_speech import audio as audio
from test_listening_body_cadence import offer_and_drain, add_words

pytestmark = pytest.mark.usefixtures('word_length_modes')
POINT = 'Acknowledge the safeguard; compare the remaining funding gap.'


def items():
    return [dict(axis=0, target=0, issue='Funding gap', action='concede_then_distinguish',
                 point=POINT, weight=4),
            dict(axis=1, target=None, issue='Net benefit', action='weigh',
                 point='Compare surviving costs against benefits.', weight=2)]


def author(p, value=None):
    base = p.helper_client
    def complete(*, prompt, **kw):
        raw = base(prompt=prompt, **kw)
        if prompt.startswith(('LISTENING PREFIX DRAFT:', 'LISTENING PREFIX REPAIR:', 'LISTENING BODY DRAFT:')):
            parsed = json.loads(raw[0])
            parsed['body_plan'] = items() if value is None else value
            return [json.dumps(parsed)]
        return raw
    p.helper_client = Mock(side_effect=complete)
    return p.helper_client


def test_one_author_call_preserves_bound_plan_and_keeps_metadata_unspoken(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    client = author(p)
    data = material(p, p.status, history)
    candidate = prepare(data, client, p.streaming_output_config)
    body = candidate['body_preparation']
    assert body['body_plan_audit']['status'] == 'authored'
    assert body['body_plan'][0]['target']['node_id'] == data['current_targets'][0]['node_id']
    assert body['body_plan'][0]['point'] == POINT
    assert body['draft'] == TAIL and POINT not in candidate['text']
    assert sum(c.kwargs['prompt'].startswith('LISTENING PREFIX DRAFT:') for c in client.call_args_list) == 1
    assert not any('REPAIR:' in c.kwargs['prompt'] for c in client.call_args_list)


def test_plan_reaches_review_and_final_publication(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    author(p)
    result = p.rebuttal_generation(history, 60, time_control=True)
    saved = trace(tmp_path)
    assert saved['body_plan_audit']['status'] == 'authored'
    assert saved['body_plan'][0]['target'] is not None
    assert saved['body_task_binding']['context']['body_plan'][0]['point'] == POINT
    assert POINT not in result and 'body_plan' not in saved['committed_text']


def test_updated_body_plan_survives_worker_and_enters_existing_evidence_query(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    author(p)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    p._listening_prefix = prep
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)
        first_calls = p.helper_client.call_count
        add_words(data, 110)
        offer_and_drain(prep, data)
        assert p.helper_client.call_count == first_calls + 1
        assert prep._body_latest['body_plan'][0]['point'] == POINT
        assert POINT in query(material(p, p.status, history))
    finally:
        prep.close()


def test_stale_node_binding_is_removed_before_reuse(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    data = material(p, p.status, history)
    plan, audit = read_author_plan(items(), data, FRAMEWORK)
    assert audit['status'] == 'authored'
    changed = copy.deepcopy(data)
    changed['current_targets'][0]['version'] = 'changed-version'
    resolved = prepared_body_plan(changed, FRAMEWORK, dict(body_plan=plan))
    assert POINT not in json.dumps(resolved)
    assert all(x['target'] is None for x in resolved)


@pytest.mark.parametrize('bad', [[], [dict(items()[0], target=999)], [dict(items()[0], action='invent')]])
def test_invalid_plan_is_audited_without_extra_repair(tmp_path, bad):
    p, history, _ = prepared_player(tmp_path)
    client = author(p, bad)
    candidate = prepare(material(p, p.status, history), client, p.streaming_output_config)
    assert candidate['body_preparation']['body_plan_audit']['status'] == 'fallback'
    assert candidate['body_preparation']['draft'] == TAIL
    assert not any('REPAIR:' in c.kwargs['prompt'] for c in client.call_args_list)


@pytest.mark.parametrize('snapshot', [False, True])
def test_handoff_preserves_authored_plan_in_cold_and_snapshot_paths(audio, tmp_path, snapshot):
    from dataclasses import replace
    from test_final_input_overlap import ready_handoff
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    prep, handoff = ready_handoff(p, history, encoded, snapshot_delivery=snapshot)
    assert handoff
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    client = author(p)
    if snapshot:
        candidate = prepare(material(p, p.status, history), client, p.streaming_output_config)
        handoff['candidate']['body_preparation'] = candidate['body_preparation']
        p.streaming_output_config = replace(p.streaming_output_config,
            listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    else:
        handoff['candidate'].pop('body_preparation', None)
    handoff['body_future'] = None
    try:
        result = p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=lambda: history,
            listening_recognized_input=lambda: history)
        saved = trace(tmp_path)
        assert saved['body_plan_audit']['status'] == 'authored'
        assert saved['body_plan'][0]['point'] == POINT
        assert POINT not in result
        if snapshot:
            assert saved['body_publication_snapshot']['context']['body_plan'][0]['target'] is not None
        else:
            assert saved['body_preparation_source'] == 'cold_draft'
    finally:
        prep.close()
