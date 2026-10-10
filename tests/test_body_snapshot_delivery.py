"""Complete-ASR body publication must not wait for mutable analysis state."""
import copy
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock

import pytest

from streaming.flat_speaking import SegmentRejected
from test_full_speech import audio as audio
from test_final_input_overlap import ready_handoff
from test_listening_prefix import TAIL, prepared_player, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.mark.parametrize('streaming', [False, True])
def test_body_publishes_before_analysis_and_uses_frozen_inputs(audio, tmp_path, streaming):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True,
        listening_stream_body_revision=streaming, min_chunk_words=1)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    final = copy.deepcopy(history)
    final[-1]['content'] += ' FINAL_SCOPE_EXCEPTION.'
    recognized = Future()
    first, body, reviewed = threading.Event(), threading.Event(), threading.Event()
    prompts = []
    base = p.helper_client
    second = 'The final exception also preserves access for residents with existing commitments.'

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert 'FINAL_SCOPE_EXCEPTION' in prompt
            assert first.wait(3)
            reviewed.set()
            return [json.dumps(dict(points=[], other_corrections=['Keep the final exception.']))]
        if kwargs.get('json_mode') is False:
            assert 'FINAL_SCOPE_EXCEPTION' in kwargs['history_messages'][-1]['content']
            prompts.append(prompt)
            if streaming:
                kwargs['on_text'](TAIL + '\n\n')
                assert body.wait(3), 'First body paragraph waited for revision completion or analysis'
                kwargs['on_text'](second)
            return [TAIL + '\n\n' + second]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)

    def emit(index, *args):
        p.listen.assert_not_called()
        p._get_revision_suggestion.assert_not_called()
        p._length_adjust.assert_not_called()
        (body if index else first).set()

    def complete():
        assert body.wait(3), 'Body audio waited for complete analysis'
        # These late derived/configuration updates must not change the bound task.
        p.planner.chunks = [final[-1]['content']]
        p.planner.state['body_plan'] = [dict(issue='Late plan', covers=[], target=None,
            action='answer_objection', point='Late derived update', weight=1)]
        p.high_quality_evidence_pool = [dict(id='late', content='Later evidence pool')]
        p.config.temperature = .99
        return copy.deepcopy(final)

    p.tts_chunk_callback = emit
    with ThreadPoolExecutor(max_workers=1) as pool:
        f = pool.submit(p.rebuttal_generation, history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=recognized.result)
        assert first.wait(3)
        assert not reviewed.is_set() and not body.is_set()
        recognized.set_result(final)
        assert TAIL in f.result(timeout=5)
    saved = trace(tmp_path)
    assert saved['body_publication_mode'] == 'complete_asr_snapshot'
    assert saved['chunks'][1]['ready_seconds'] < saved['final_input_ready_seconds']
    assert saved['body_publication_snapshot']['context']['history'] == final
    assert saved['body_publication_snapshot']['evidence'] == []
    assert len(prompts) == 1
    p.listen.assert_called_once_with(final)


def test_changed_recognition_snapshot_blocks_body_publication(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    corrected = copy.deepcopy(history)
    corrected[-1]['content'] += ' CHANGED_AFTER_SNAPSHOT.'
    reads = []
    def recognized():
        reads.append(True)
        return copy.deepcopy(history if len(reads) == 1 else corrected)
    p.tts_chunk_callback = Mock()
    with pytest.raises(SegmentRejected, match='Complete-ASR history changed'):
        p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=lambda: corrected,
            listening_recognized_input=recognized)
    saved = trace(tmp_path)
    assert [c['index'] for c in saved['chunks']] == [0]
    assert saved['status'] == 'failed'


@pytest.mark.parametrize('failure', ['none', 'recognition', 'review', 'revision', 'analysis'])
def test_snapshot_delivery_success_and_failure_do_not_retry_providers(audio, tmp_path, monkeypatch, failure):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded, snapshot_delivery=True)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    first, body = threading.Event(), threading.Event()
    base = p.helper_client
    reviews, revisions = [], []
    # Exercise the branch that can publish an unchanged, reviewed body.
    if failure == 'none':
        monkeypatch.setattr('streaming.body_task.BodyTask.needs_revision', lambda *args, **kwargs: False)

    def recognized():
        assert first.wait(3)
        if failure == 'recognition':
            raise RuntimeError('recognition failed')
        return copy.deepcopy(history)

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            reviews.append(prompt)
            if failure == 'review':
                raise RuntimeError('review failed')
            return [json.dumps(dict(points=[], other_corrections=['Keep the qualification.']))]
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            if failure == 'revision':
                raise RuntimeError('revision failed')
            return [TAIL]
        return base(prompt=prompt, **kwargs)

    def complete():
        if failure in ('none', 'analysis'):
            assert body.wait(3), 'Unchanged body waited for analysis'
        if failure == 'analysis':
            raise RuntimeError('analysis failed')
        return copy.deepcopy(history)

    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda index, *args: (body if index else first).set()
    def run():
        return p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=recognized)
    if failure == 'none':
        assert TAIL in run()
        assert not revisions
        assert trace(tmp_path)['status'] == 'completed'
    else:
        with pytest.raises(RuntimeError, match=failure + ' failed'):
            run()
        assert trace(tmp_path)['status'] == 'failed'
        if failure != 'analysis':
            assert not body.is_set()
    assert len(reviews) <= 1 and len(revisions) <= 1
