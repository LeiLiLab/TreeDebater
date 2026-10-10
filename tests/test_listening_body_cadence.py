"""Offline scheduling checks for word-batched provisional body updates."""
import copy
import json
import threading

import pytest

from streaming.config import OutputConfig
from streaming.body_task import BodyTask
from streaming.listening_prefix import PrefixPreparation, material
from test_listening_prefix import prepared_player, prefix_helper
from test_listening_overview import event_count, wait_for

pytestmark = pytest.mark.usefixtures('word_length_modes')


def offer_and_drain(prep, data):
    count = event_count(prep)
    prep.offer(data)
    wait_for(lambda: event_count(prep) == count + 1 and prep._body_thread is None)


def add_words(data, count):
    data['heard_transcript'] += ' ' + ' '.join(f'word{i}' for i in range(count))
    data['final_transcript'] = data['heard_transcript']


def test_small_batches_accumulate_without_rewriting_body(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)
        original = copy.deepcopy(prep._body_latest)
        calls = p.helper_client.call_count
        for increment in (40, 59):
            add_words(data, increment)
            offer_and_drain(prep, data)
            assert prep._body_latest == original
            assert p.helper_client.call_count == calls
        # Punctuation alone must not satisfy the hundred-word threshold.
        data['heard_transcript'] += ' ... !'
        offer_and_drain(prep, data)
        assert p.helper_client.call_count == calls
        add_words(data, 1)
        offer_and_drain(prep, data)
        assert p.helper_client.call_count == calls + 1  # draft only; feedback already used
        assert prep._body_latest['heard_transcript'] == data['heard_transcript']
        assert prep._body_latest['feedback'] is None
        assert prep._body_latest['feedback_status'] == 'deferred_to_final_input'
        data['current_targets'] = []
        offer_and_drain(prep, data)
        assert p.helper_client.call_count == calls + 2
    finally:
        prep.close()


def test_pending_body_uses_latest_input_and_counts_from_actual_start(tmp_path):
    p, history, node = prepared_player(tmp_path)
    base = prefix_helper(node)
    started, release = threading.Event(), threading.Event()
    bodies = []

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:'):
            bodies.append(kwargs['history_messages'][-1]['content'].split('\n', 1)[1])
            if len(bodies) == 1:
                started.set()
                assert release.wait(3)
        return base(prompt=prompt, **kwargs)

    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)  # Initial body now comes from the shared speech call.
        add_words(data, 100)
        prep.offer(data)
        assert started.wait(3)
        for increment in (100, 40):
            add_words(data, increment)
            count = event_count(prep)
            prep.offer(data)
            wait_for(lambda: event_count(prep) == count + 1)
        release.set()
        wait_for(lambda: prep._body_thread is None)
        assert len(bodies) == 2 and bodies[-1] == data['heard_transcript']
        add_words(data, 99)
        offer_and_drain(prep, data)
        assert len(bodies) == 2
        add_words(data, 1)
        offer_and_drain(prep, data)
        assert len(bodies) == 3 and bodies[-1] == data['heard_transcript']
        assert sum(e.get('feedback_status') == 'completed' for e in prep.events) == 0
        assert sum(e.get('feedback_status') == 'deferred_to_final_input' for e in prep.events) == 4
    finally:
        release.set()
        prep.close()


@pytest.mark.parametrize('change', ['opening', 'transcript', 'evidence', 'budget'])
def test_invalidated_body_input_bypasses_word_threshold(tmp_path, change):
    p, history, _ = prepared_player(tmp_path)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)
        candidate = prep.peek()
        if change == 'opening':
            candidate['text'] += ' We will consider implementation.'
        elif change == 'transcript':
            data['heard_transcript'] = 'Correction: ' + data['heard_transcript']
            data['final_transcript'] = data['heard_transcript']
        elif change == 'evidence':
            data['supplied_evidence'].append(dict(title='Updated evidence', content='A new qualification.'))
        else:
            data['max_time'] += 20
        calls = p.helper_client.call_count
        prep._offer_body(data, candidate)
        wait_for(lambda: prep._body_thread is None)
        assert p.helper_client.call_count == calls + 1
        transferred = json.loads(prep.pending_body(candidate).result(timeout=1))
        assert transferred['feedback_status'] == 'deferred_to_final_input'
        assert transferred['feedback'] is None
    finally:
        prep.close()


def test_zero_threshold_restores_unbatched_updates(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config.listening_body_update_words = 0
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    data = material(p, p.status, history)
    try:
        offer_and_drain(prep, data)
        calls = p.helper_client.call_count
        add_words(data, 1)
        offer_and_drain(prep, data)
        assert p.helper_client.call_count == calls + 1
    finally:
        prep.close()
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            OutputConfig(listening_body_update_words=invalid)


def test_overlong_body_is_retained_for_final_revision(tmp_path):
    p, history, node = prepared_player(tmp_path)
    p.speech_budgets = dict(rebuttal=10)
    base = prefix_helper(node)
    draft = ' '.join(['reason'] * 20) + '.'
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:'):
            return [json.dumps({'draft': draft})]
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            value = json.loads(base(prompt=prompt, **kwargs)[0])
            return [json.dumps(dict(value, draft=draft))]
        return base(prompt=prompt, **kwargs)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    try:
        offer_and_drain(prep, material(p, p.status, history))
        assert prep._body_latest['draft'] == draft
        assert prep._body_latest['needs_fit'] is True
        assert prep._body_latest['feedback'] is None
        assert any(e['status'] == 'body_saved_unreviewed' and e['needs_fit'] for e in prep.events)
    finally:
        prep.close()


def test_no_preparatory_feedback_and_one_complete_input_review(tmp_path):
    (p, history, node) = prepared_player(tmp_path)
    base = prefix_helper(node)
    (provisional, complete) = ([], [])

    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY FEEDBACK:'):
            provisional.append(prompt)
        if prompt.startswith('LISTENING WHOLE SPEECH FEEDBACK:'):
            complete.append(prompt)
            return ['No changes']
        return base(prompt=prompt, **kwargs)
    data = material(p, p.status, history)
    prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    try:
        offer_and_drain(prep, data)
        for _ in range(2):
            add_words(data, 100)
            offer_and_drain(prep, data)
        assert len(provisional) == 0
        latest = copy.deepcopy(prep._body_latest)
        assert latest['feedback'] is None and latest['feedback_status'] == 'deferred_to_final_input'
        assert latest['heard_transcript'] == data['heard_transcript']
        transferred = json.loads(prep.pending_body(prep.peek()).result(timeout=1))
        assert transferred['feedback_status'] == 'deferred_to_final_input'
        assert transferred['feedback'] is None
        final_history = copy.deepcopy(history)
        final_history[-1]['content'] = data['heard_transcript'] + ' FINAL_NEW_CONDITION.'
        task = BodyTask.create(motion=data['motion'], side=data['our_side'], stage=data['stage'], history=final_history, prefix=latest['prefix_text'], framework=latest['framework'], draft=latest['draft'], preparation=latest, body_plan=latest['body_plan'])
        assert task.review(helper, p.simulated_audience)[0] == 'No changes'
        assert len(complete) == 1 and 'FINAL_NEW_CONDITION' in complete[0]
    finally:
        prep.close()
    next_prep = PrefixPreparation(p.planner.turn, helper, p.streaming_output_config)
    try:
        offer_and_drain(next_prep, data)
        assert len(provisional) == 0
    finally:
        next_prep.close()
