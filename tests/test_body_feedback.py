"""Native Audience prompt reuse, response extraction and concurrent isolation."""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
import threading

import pytest
from agents import Audience, AudienceConfig
from streaming.body_feedback import review_whole_speech
from streaming.flat_speaking import SegmentRejected
from utils.prompts.others import audience_feedback_prompt


def review(helper, **kwargs):
    return review_whole_speech(helper, motion='Motion', side='for', stage='rebuttal',
        statement='Opening.\n\nOur mechanism.', history=[], prefix='Opening.', **kwargs)


def test_native_audience_prompt_config_and_corrections_are_used():
    audience = Audience(AudienceConfig(model='gpt-4o', temperature=.8, max_tokens=1234,
                                      system_prompt='Audience system.'))
    audience.conversation.append({'role':'user','content':'STALE unrelated review'})
    raw = '[Comprehensive Analysis]\nClear.\n[Critical Issues and Minimal Revision Suggestions]\nExplain the mechanism with an example.'
    helper = Mock(return_value=[raw])
    feedback, details = review(helper, audiences=[audience])
    call = helper.call_args.kwargs
    native = audience_feedback_prompt.format(motion='Motion', side='for', stage='Rebuttal',
        statement='Opening.\n\nOur mechanism.', history='', retrieval='')
    criteria = native.split('### Evaluation Dimensions\n', 1)[1].split('### Output Format\n', 1)[0]
    assert criteria in call['prompt']
    assert '[Comprehensive Analysis]' not in call['prompt']
    assert 'at most THREE' in call['prompt'] and '180 English words' in call['prompt']
    assert 'Do not invent issues or pad' in call['prompt']
    assert call['model'] == 'gpt-4o' and call['temperature'] == .8
    assert call['max_tokens'] == 1234 and call['sys'] == 'Audience system.'
    assert call['json_mode'] is False
    assert 'direct edits to the remaining body only' in call['prompt']
    assert 'Return JSON {"points"' not in call['prompt']
    assert 'Explain the mechanism' in feedback and 'Comprehensive Analysis' not in feedback
    assert details == [raw] and len(audience.conversation) == 2


def test_no_changes_and_multiple_audience_corrections():
    panel = [Audience(AudienceConfig()), Audience(AudienceConfig())]
    helper = Mock(side_effect=[['[Critical Issues and Minimal Revision Suggestions]\nNo changes.'],
                               ['[Critical Issues and Minimal Revision Suggestions]\nClarify the example.']])
    feedback, details = review(helper, audiences=panel)
    assert 'Audience 2 Feedback' in feedback and 'Audience 1 Feedback' not in feedback
    assert len(details) == 2
    assert review(Mock(return_value=['[Critical Issues and Minimal Revision Suggestions]\nNo changes']))[0] == 'No changes'
    with pytest.raises(SegmentRejected, match='empty'):
        review(Mock(return_value=['']))


def test_parallel_native_audience_reviews_do_not_share_conversation():
    audience = Audience(AudienceConfig())
    barrier = threading.Barrier(2)
    calls = []
    def complete(**kwargs):
        calls.append(kwargs['prompt'])
        barrier.wait(timeout=3)
        return ['No changes']
    with ThreadPoolExecutor(2) as pool:
        jobs = [pool.submit(audience.feedback, text, isolated=True, completion=complete)
                for text in ('FIRST input', 'SECOND input')]
        assert [job.result() for job in jobs] == ['No changes', 'No changes']
    assert sorted(calls) == ['FIRST input','SECOND input']
    assert audience.conversation == []


def test_provisional_review_uses_only_its_supplied_history_and_native_dimensions():
    helper = Mock(return_value=['No changes'])
    review(helper, provisional=True)
    prompt = helper.call_args.kwargs['prompt']
    assert 'Do not assume future input' in prompt
    for dimension in ('Core Message Clarity', 'Engagement Impact', 'Evidence Presentation', 'Persuasive Elements'):
        assert dimension in prompt
    assert 'at most THREE' in prompt and '180 English words' in prompt
