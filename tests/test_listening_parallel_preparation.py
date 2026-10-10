"""Event-gated preparation, approved-version fallback and context ownership."""
import copy
import json
import threading
import time
from dataclasses import replace
from unittest.mock import Mock

import pytest

from streaming.listening_prefix import PrefixPreparation, material, source_stamp
from test_full_speech import audio as audio
from test_initial_prefix_review import once_player
from test_listening_prefix import PREFIX, TAIL, trace
from test_final_input_overlap import ready_handoff

pytestmark = pytest.mark.usefixtures('word_length_modes')
NEW_PREFIX = 'We oppose this trial on delivery and the newly raised affordability issue.'


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def changed(data, *, replace_prefix=False):
    data = copy.deepcopy(data)
    data['heard_transcript'] += ' A newly heard central issue.'
    if replace_prefix:
        data['framework'].update(prefix_action='replace', core_dispute='New substantive clash.')
    return data


def test_body_update_finishes_while_initial_tts_is_blocked(audio, tmp_path):
    p, history = once_player(tmp_path)
    _, encoded = audio
    started, release, body = (threading.Event() for _ in range(3))
    base = p.helper_client
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:'):
            body.set()
        return base(prompt=prompt, **kwargs)
    def synthesize(*args):
        started.set()
        assert release.wait(3)
        return encoded(2)
    config = replace(p.streaming_output_config, listening_body_update_words=0)
    prep = PrefixPreparation(p.planner.turn, helper, config, synthesize)
    data = material(p, p.status, history)
    try:
        prep.offer(data)
        assert started.wait(3)
        prep.offer(changed(data))
        assert body.wait(3), 'Body update waited for initial TTS'
        wait_for(lambda: prep._body_latest is not None and prep._body_latest['source_stamp'] == source_stamp(changed(data)))
        assert not prep._audio_futures[(PREFIX, config.voice, config.model)].done()
    finally:
        release.set()
        prep.close()
    assert all(not thread.is_alive() for thread in prep._audio_threads)


@pytest.mark.parametrize('blocked', ['review', 'tts', 'body'])
def test_handoff_uses_ready_approved_version_without_waiting_for_replacement(audio, tmp_path, blocked):
    p, history = once_player(tmp_path)
    _, encoded = audio
    entered, release, replacement_audio = (threading.Event() for _ in range(3))
    base = p.helper_client
    drafts = []
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING BODY DRAFT:') and blocked == 'body':
            entered.set()
            assert release.wait(3)
        if prompt.startswith('LISTENING PREFIX REVIEW:') and NEW_PREFIX in prompt and blocked == 'review':
            entered.set()
            assert release.wait(3)
        result = base(prompt=prompt, **kwargs)
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            value = json.loads(result[0])
            drafts.append(value)
            if len(drafts) > 1:
                value['text'] = NEW_PREFIX
                return [json.dumps(value)]
        return result
    def synthesize(text, config):
        if text == NEW_PREFIX:
            replacement_audio.set()
            if blocked == 'tts':
                entered.set()
                assert release.wait(3)
        return encoded(2)
    config = replace(p.streaming_output_config, listening_body_update_words=0)
    prep = PrefixPreparation(p.planner.turn, helper, config, synthesize)
    data = material(p, p.status, history)
    try:
        prep.offer(data)
        wait_for(lambda: prep.prepared_audio(PREFIX, config) is not None)
        if blocked == 'body':
            prep.offer(changed(data))
            assert entered.wait(3)
        prep.offer(changed(data, replace_prefix=True))
        if blocked == 'body':
            assert replacement_audio.wait(3), 'New prefix TTS waited for obsolete body update'
            wait_for(lambda: prep.prepared_audio(NEW_PREFIX, config) is not None)
        else:
            assert entered.wait(3)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff and handoff['audio'] is not None
        assert handoff['candidate']['text'] == (NEW_PREFIX if blocked == 'body' else PREFIX)
        assert handoff['candidate']['audits'][-1]['semantic_review']
        assert handoff['candidate']['audits'][-1]['accepted']
    finally:
        release.set()
        prep.close()


def test_body_uses_latest_completed_context_after_handoff(audio, tmp_path):
    p, history = once_player(tmp_path)
    _, encoded = audio
    prep, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    latest = material(p, p.status, history)
    latest['clash_records'] = [dict(issue='NEW_COMPLETED_TREE', entries=[])]
    latest['feedback_context']['retrieval'] = 'NEW_COMPLETED_TREE feedback context'
    expected = copy.deepcopy(latest)
    body = threading.Event()
    base = p.helper_client
    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert 'NEW_COMPLETED_TREE' in prompt
            return [json.dumps(dict(points=[], other_corrections=['Keep the qualification.']))]
        if kwargs.get('json_mode') is False:
            assert 'NEW_COMPLETED_TREE' in prompt
            return [TAIL]
        return base(prompt=prompt, **kwargs)
    p.helper_client = Mock(side_effect=helper)
    def recognized():
        prep.offer(expected)
        # Caller mutation and a different turn must never alter the owned snapshot.
        wrong = copy.deepcopy(latest)
        wrong['turn'] = 'wrong:opening'
        prep.offer(wrong)
        return copy.deepcopy(history)
    def complete():
        assert body.wait(3)
        return copy.deepcopy(history)
    p.tts_chunk_callback = lambda index, *args: body.set() if index else None
    try:
        p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
            listening_recognized_input=recognized, listening_input_completion=complete)
        saved = trace(tmp_path)
        assert saved['body_context_snapshot']['source_stamp'] == source_stamp(expected)
        assert saved['body_publication_snapshot']['context']['clash_records'] == expected['clash_records']
        assert handoff['data']['clash_records'] != expected['clash_records']
        latest['clash_records'].clear()
        assert prep.body_context(p.status, p.planner.turn)['clash_records'] == expected['clash_records']
    finally:
        prep.close()


def test_explicitly_invalidated_ready_opening_cannot_be_handoff_fallback(audio, tmp_path):
    from test_listening_prefix import prefix_helper, conflict
    p, history = once_player(tmp_path)
    _, encoded = audio
    prep, handoff = ready_handoff(p, history, encoded)
    candidate = handoff['candidate']
    newer = changed(handoff['data'])
    rejecting = prefix_helper(None, review_edit=lambda result, data: conflict(result, data))
    try:
        verdict = prep.initial_review.check(candidate, newer, rejecting)
        assert verdict['review_format_valid'] and not verdict['accepted']
        assert prep.handoff(p.status, p.planner.turn, 60) is None
    finally:
        prep.close()
