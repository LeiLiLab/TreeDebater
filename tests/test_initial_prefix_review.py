"""Version-bound reviews with format retry, before TTS; providers are mocked."""
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from streaming.listening_prefix import InitialPrefixReview, PrefixPreparation, material, prepare
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, conflict, prefix_helper, prepared_player, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')


def once_player(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prefix_review_enabled=False,
        listening_prefix_initial_review_enabled=True,
        listening_prefix_overlap_final_update=True,
        listening_prefix_pre_synthesize=True)
    return p, history


def test_initial_rejection_repairs_once_and_reviews_repair(tmp_path):
    p, history = once_player(tmp_path)
    base = prefix_helper(None, review_edit=lambda result, data: conflict(result, data) if data['draft'] == PREFIX else None)
    repaired = 'We oppose this trial while recognizing the updated funding commitment.'
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX REPAIR:'):
            value = json.loads(base(prompt=prompt, **kwargs)[0])
            value['text'] = repaired
            return [json.dumps(value)]
        return base(prompt=prompt, **kwargs)
    checked = Mock(side_effect=helper)
    state = InitialPrefixReview()
    candidate = prepare(material(p, p.status, history), checked, p.streaming_output_config,
                        initial_review=state)
    assert candidate['text'] == repaired
    assert [a['semantic_review'] for a in candidate['audits']] == [True, True]
    assert not candidate['audits'][0]['accepted'] and candidate['audits'][1]['accepted']
    assert sum('LISTENING PREFIX REVIEW:' in c.kwargs['prompt'] for c in checked.call_args_list) == 2
    assert state.snapshot()['checked_text'] == PREFIX
    # Endpoint fallback reuses each exact text/context verdict.
    prepare(material(p, p.status, history), checked, p.streaming_output_config,
            initial_review=state, endpoint=True)
    assert sum('LISTENING PREFIX REVIEW:' in c.kwargs['prompt'] for c in checked.call_args_list) == 2


def test_invalid_review_response_stops_after_three_attempts(tmp_path):
    p, history = once_player(tmp_path)
    base = p.helper_client
    def helper(*, prompt, **kwargs):
        if 'LISTENING PREFIX REVIEW:' in prompt:
            return ['{}']
        return base(prompt=prompt, **kwargs)
    checked = Mock(side_effect=helper)
    state = InitialPrefixReview()
    data = material(p, p.status, history)
    candidate = dict(text=PREFIX, framework=data['framework'])
    verdict = state.check(candidate, data, checked)
    assert len(verdict['format_attempts']) == 3
    assert not verdict['review_format_valid'] and not verdict['accepted']
    assert state.check(candidate, data, checked) == verdict
    assert checked.call_count == 3


@pytest.mark.parametrize('success_attempt', [2, 3])
@pytest.mark.parametrize('malformed', ['json', 'empty_quote'])
def test_review_format_retry_is_independent_of_framework_repair(tmp_path, malformed, success_attempt):
    p, history = once_player(tmp_path)
    base = p.helper_client
    review_prompts = []
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING FRAMEWORK REPAIR:'):
            return [json.dumps(dict(framework=material(p, p.status, history)['framework']))]
        result = base(prompt=prompt, **kwargs)
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            value = json.loads(result[0])
            value.pop('framework')
            return [json.dumps(value)]
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            review_prompts.append(prompt)
            if len(review_prompts) < success_attempt:
                if malformed == 'json':
                    return ['not valid JSON']
                value = json.loads(result[0])
                value['stance_assessment']['quote'] = ''
                return [json.dumps(value)]
        return result
    checked = Mock(side_effect=helper)
    state = InitialPrefixReview()
    candidate = prepare(material(p, p.status, history), checked, p.streaming_output_config,
                        initial_review=state)
    assert candidate['text'] == PREFIX
    assert candidate['audits'][-1]['accepted']
    assert len(review_prompts) == success_attempt
    for retry in review_prompts[1:]:
        assert 'REVIEW FORMAT REPAIR:' in retry
        assert json.loads(review_prompts[0].splitlines()[-1]) == json.loads(retry.splitlines()[-1])
    prompts = [c.kwargs['prompt'] for c in checked.call_args_list]
    assert sum(x.startswith('LISTENING FRAMEWORK REPAIR:') for x in prompts) == 1
    assert not any(x.startswith('LISTENING PREFIX REPAIR:') for x in prompts)
    attempts = state.snapshot()['verdict']['format_attempts']
    assert len(attempts) == success_attempt and attempts[-1]['accepted']
    assert all(not attempt['review_format_valid'] for attempt in attempts[:-1])
    assert state.check(candidate, candidate['reviewed_context'], checked, endpoint=True)['accepted']
    assert len(review_prompts) == success_attempt


def test_review_precedes_tts_and_covers_later_rewrite_without_endpoint_recheck(audio, tmp_path):
    p, history = once_player(tmp_path)
    entered, release, first_audio = threading.Event(), threading.Event(), threading.Event()
    _, encoded = audio
    base = p.helper_client
    drafts = []
    def helper(*, prompt, **kwargs):
        if 'LISTENING PREFIX REVIEW:' in prompt:
            entered.set()
            assert release.wait(3)
        result = base(prompt=prompt, **kwargs)
        if 'LISTENING PREFIX DRAFT:' in prompt:
            value = json.loads(result[0])
            drafts.append(value)
            if len(drafts) > 1:
                value['text'] = 'We oppose this trial on delivery and the newly raised affordability issue.'
                result = [json.dumps(value)]
        return result
    p.helper_client = Mock(side_effect=helper)
    tts = Mock(side_effect=lambda *args: encoded(2))
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config, tts)
    p._listening_prefix = prep
    data = material(p, p.status, history)
    prep.offer(data)
    try:
        assert entered.wait(3)
        tts.assert_not_called()
        release.set()
        deadline = time.monotonic() + 3
        while prep.prepared_audio(PREFIX, p.streaming_output_config) is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert prep.prepared_audio(PREFIX, p.streaming_output_config)
        changed = copy.deepcopy(data)
        changed['heard_transcript'] += ' A newly heard central issue.'
        changed['framework'].update(prefix_action='replace', core_dispute='New substantive clash.')
        prep.offer(changed)
        deadline = time.monotonic() + 3
        while not any(e.get('grounded_update') and e['status'] == 'reviewed' for e in prep.events) and time.monotonic() < deadline:
            time.sleep(.005)
        assert any(e.get('grounded_update') and e['status'] == 'reviewed' for e in prep.events)
        revised = 'We oppose this trial on delivery and the newly raised affordability issue.'
        deadline = time.monotonic() + 3
        while prep.prepared_audio(revised, p.streaming_output_config) is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert prep.prepared_audio(revised, p.streaming_output_config)
        handoff = prep.handoff(p.status, p.planner.turn, 60)
        assert handoff
        p.tts_chunk_callback = lambda *args: first_audio.set()
        def complete():
            assert first_audio.wait(3), 'Prepared opening waited for final ASR'
            return history
        p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
            listening_recognized_input=complete, listening_input_completion=complete)
        assert sum('LISTENING PREFIX REVIEW:' in c.kwargs['prompt'] for c in p.helper_client.call_args_list) == 2
        saved = trace(tmp_path)
        assert saved['initial_prefix_review']['completed']
        assert saved['initial_prefix_review']['checked_text'] == PREFIX
        assert saved['chunks'][0]['text'] == revised
        assert saved['chunks'][0]['text'] == saved['initial_prefix_review']['versions'][-1]['checked_text']
        assert all(row['verdict']['accepted'] for row in saved['initial_prefix_review']['versions'])
        assert saved['prefix_input_scope'] == 'listening_snapshot'
    finally:
        release.set()
        prep.close()


def test_concurrent_fallback_waits_for_initial_check_without_duplicate_request(tmp_path):
    p, history = once_player(tmp_path)
    candidate = dict(text=PREFIX, framework=material(p, p.status, history)['framework'])
    data = material(p, p.status, history)
    state = InitialPrefixReview()
    entered, release = threading.Event(), threading.Event()
    base = p.helper_client
    def helper(**kwargs):
        entered.set()
        assert release.wait(3)
        return base(**kwargs)
    checked = Mock(side_effect=helper)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(state.check, candidate, data, checked)
        try:
            assert entered.wait(3)
            second = pool.submit(state.check, candidate, data, checked)
            assert not second.done()
        finally:
            release.set()
        assert first.result(timeout=3)['accepted']
        assert second.result(timeout=3)['accepted']
    assert checked.call_count == 1
