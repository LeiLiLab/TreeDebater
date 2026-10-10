"""The simplified listening path adds preparation, not extra review stages."""
import copy
import json
import threading
import time
from dataclasses import replace
from types import MethodType
from unittest.mock import Mock

import pytest
from ouragents import TreeDebater
from streaming.body_feedback import review_whole_speech
from streaming.body_revision import draft_prompt, revision_prompt
from streaming.listening_prefix import PrefixPreparation, material, prepare
from test_final_input_overlap import ready_handoff
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, prepared_player, trace
from utils.prompts.authoring import stage_strategy

pytestmark = pytest.mark.usefixtures('word_length_modes')


def test_prefix_requires_one_joint_draft_and_spoken_prefix_review(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    result = prepare(material(p, p.status, history), p.helper_client, p.streaming_output_config)
    assert result['text'] == PREFIX
    assert p.helper_client.call_count == 2
    assert result['audits'][-1]['accepted'] and result['audits'][-1]['semantic_review']


def test_preparation_saves_body_without_audience_calls(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    try:
        prep.offer(material(p, p.status, history))
        deadline = time.monotonic() + 3
        while not any(e['status'] == 'body_saved_unreviewed' for e in prep.events) and time.monotonic() < deadline:
            time.sleep(.005)
        saved = prep.freeze()['body_preparation']
        assert saved['draft'] == TAIL and saved['feedback'] is None
        assert saved['feedback_status'] == 'deferred_to_final_input'
        prompts = [c.kwargs['prompt'] for c in p.helper_client.call_args_list]
        assert len(prompts) == 2
        assert prompts[1].startswith('LISTENING PREFIX REVIEW:')
        assert prompts[0].startswith('LISTENING PREFIX DRAFT:')
        assert 'opening and body together' in prompts[0]
    finally:
        prep.close()


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_body_draft_uses_native_strategy_without_legacy_output_layout(tmp_path, stage):
    p, history, _ = prepared_player(tmp_path)
    data = material(p, stage, history)
    prompt = draft_prompt(data, PREFIX, 300)
    assert stage_strategy(stage) in prompt
    assert '**Statement**' not in prompt
    assert 'IMMUTABLE SPOKEN PREFIX' in prompt and PREFIX in prompt
    assert '97%' not in prompt and 'exactly TWO' not in prompt
    assert '{n_words}' not in prompt and '{tips}' not in prompt


def test_revision_keeps_assigned_context_with_one_spoken_output_format():
    values = dict(motion='Motion', side='for', stage='rebuttal', statement='Draft.', feedback='Fix.',
                  allocation_plan='Plan.', evidence=[], prefix=PREFIX, n_words=300)
    prompt = revision_prompt(**values)
    material = json.loads(prompt.split('Context and material (data):\n', 1)[1])
    assert (material['motion'], material['side'], material['stage']) == ('Motion', 'for', 'rebuttal')
    assert material['already_spoken_prefix'] == PREFIX
    assert 'Only the already spoken prefix is immutable.' in prompt
    assert 'List ALL your references' not in prompt
    assert 'If there is no overview' not in prompt
    assert 'Return only natural spoken body text' in prompt


def test_closing_skips_native_audience_feedback():
    helper = Mock(side_effect=AssertionError('Closing should skip feedback'))
    assert review_whole_speech(helper, motion='Motion', side='for', stage='closing',
        statement='Closing.', history=[], prefix=PREFIX) == ('No changes', [])
    helper.assert_not_called()


@pytest.mark.parametrize('changed_input', [False, True])
def test_first_audio_overlaps_final_input_and_revision_reuse_is_exact(audio, tmp_path, changed_input):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback=None,
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    final = copy.deepcopy(history)
    final[-1]['content'] += ' FINAL_ASR_QUALIFICATION.'
    first, revised = threading.Event(), threading.Event()
    feedbacks, revisions = [], []
    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            feedbacks.append(prompt)
            assert first.wait(3)
            assert 'FINAL_ASR_QUALIFICATION' in prompt
            return ['Keep the final qualification.']
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            assert 'FINAL_ASR_QUALIFICATION' in kwargs['history_messages'][-1]['content'] and PREFIX in prompt
            revised.set()
            return [TAIL]
        raise AssertionError('Unexpected extra call: ' + prompt[:70])
    original_helper = p.helper_client
    def with_gate(**kwargs):
        if kwargs['prompt'].startswith('LISTENING PREFIX REVIEW:'):
            return original_helper(**kwargs)
        return helper(**kwargs)
    p.helper_client = Mock(side_effect=with_gate)
    p.tts_chunk_callback = lambda *args: first.set()
    def complete():
        assert first.wait(3)
        assert revised.wait(3), 'Revision waited for tree/planning completion'
        resolved = copy.deepcopy(final)
        if changed_input:
            resolved[-1]['content'] += ' CORRECTED_TRANSCRIPT.'
        p.planner.chunks = [resolved[-1]['content']]
        return resolved
    result = p.rebuttal_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete,
        listening_recognized_input=lambda: copy.deepcopy(final))
    assert result == PREFIX + '\n\n' + TAIL
    saved = trace(tmp_path)
    assert 'body_reviews' not in saved and 'final_body_repairs' not in saved
    assert len(feedbacks) == len(revisions) == (2 if changed_input else 1)
    assert saved['parallel_body_feedback']['reused'] is (not changed_input)
    if not changed_input:
        assert saved['parallel_body_revision']['reused']
    assert saved['chunks'][0]['ready_seconds'] < saved['final_input_ready_seconds']


@pytest.mark.parametrize('repair_succeeds', [False, True])
def test_explicit_wrong_framework_uses_one_local_repair(tmp_path, repair_succeeds):
    from streaming.listening_prefix import PrefixFormatRejected
    from test_listening_prefix import FRAMEWORK
    p, history, node = prepared_player(tmp_path)
    valid = dict(text=PREFIX, draft=TAIL, target_ids=[node.node_id], framework=FRAMEWORK)
    wrong = dict(valid, framework=dict(FRAMEWORK, position='for'))
    from test_overview_review_gate import passing
    helper = Mock(side_effect=[[json.dumps(wrong)], [json.dumps(valid if repair_succeeds else wrong)],
                               [json.dumps(passing(PREFIX, 'against'))]])
    if repair_succeeds:
        assert prepare(material(p, p.status, history), helper, p.streaming_output_config)['text'] == PREFIX
    else:
        with pytest.raises(PrefixFormatRejected):
            prepare(material(p, p.status, history), helper, p.streaming_output_config)
    assert helper.call_count == (3 if repair_succeeds else 2)
    assert 'LISTENING PREFIX REPAIR:' in helper.call_args_list[1].kwargs['prompt']


def test_frozen_draft_does_not_alias_preparation(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    try:
        prep.offer(material(p, p.status, history))
        deadline = time.monotonic() + 3
        while prep._body_latest is None and time.monotonic() < deadline:
            time.sleep(.005)
        candidate = prep.freeze()
        future = prep.pending_body(candidate)
        candidate['body_preparation']['draft'] = 'Consumer change.'
        assert prep._body_latest['draft'] == TAIL
        assert json.loads(future.result())['draft'] == TAIL
    finally:
        prep.close()


def test_local_repair_explicitly_returns_candidate_not_input_envelope(tmp_path):
    from test_listening_prefix import FRAMEWORK
    p, history, node = prepared_player(tmp_path)
    wrong = dict(text=PREFIX, draft=TAIL, target_ids=[node.node_id], framework=dict(FRAMEWORK, position='for'))
    valid = dict(wrong, framework=FRAMEWORK)
    from test_overview_review_gate import passing
    helper = Mock(side_effect=[[json.dumps(wrong)], [json.dumps(valid)], [json.dumps(passing(PREFIX, 'against'))]])
    assert prepare(material(p, p.status, history), helper, p.streaming_output_config)['text'] == PREFIX
    prompt = helper.call_args_list[1].kwargs['prompt']
    assert 'Return ONLY the repaired output object with this exact schema:' in prompt
    assert '"draft":"all remaining spoken paragraphs"' in prompt
    assert 'Repair the complete unpublished speech together' in prompt
    assert 'Your assigned side is against: oppose this exact motion.' in prompt
    assert 'hard maximum 26 word equivalents' in prompt
    assert helper.call_count == 3
