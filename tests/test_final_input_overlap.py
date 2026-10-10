"""Real offline audio and event ordering for endpoint playback before final ASR."""
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streaming.listening_prefix import PrefixPreparation, material
from streaming.flat_speaking import SegmentRejected
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, prepared_player, prefix_helper, conflict, trace
from test_listening_motion_live import live, fake_turn

pytestmark = pytest.mark.usefixtures('word_length_modes')


def ready_handoff(p, history, encoded, *, snapshot_delivery=False):
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prefix_overlap_final_update=True, listening_prefix_pre_synthesize=True,
        listening_body_snapshot_delivery=snapshot_delivery)
    p.helper_client = prefix_helper(None)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config,
                             lambda *args: encoded(2))
    p._listening_prefix = prep
    prep.offer(material(p, p.status, history))
    deadline = time.monotonic() + 3
    while prep.prepared_audio(PREFIX, p.streaming_output_config) is None and time.monotonic() < deadline:
        time.sleep(.005)
    return prep, prep.handoff(p.status, p.planner.turn, 60)


def test_prefix_plays_before_final_input_and_body_sees_new_input(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    query, encoded = audio
    prep, handoff = ready_handoff(p, history, encoded)
    assert handoff and 'target_ids' not in handoff['candidate']
    first = threading.Event()
    final_history = copy.deepcopy(history)
    final_history[-1]['content'] += ' FINAL_BATCH_NEW_SAFEGUARD.'

    def complete_input():
        assert first.wait(3), 'Waiting for final input blocked the first audio'
        p.planner.chunks.append('FINAL_BATCH_NEW_SAFEGUARD.')
        return final_history

    def emit(index, path, text, duration):
        if index == 0:
            p.listen.assert_not_called()
            p._prepare_stage_prompt.assert_not_called()
            query.assert_not_called()  # Pre-synthesized bytes are reused.
            assert text == PREFIX
            first.set()

    p.tts_chunk_callback = emit
    result = p.rebuttal_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete_input, listening_recognized_input=lambda: final_history)
    assert result.startswith(PREFIX) and TAIL in result
    p.listen.assert_called_once_with(final_history)
    prompts = [x.kwargs['prompt'] for x in p.helper_client.call_args_list]
    assert not any('ENDPOINT GATE:' in x for x in prompts)
    assert not any('LISTENING FINAL BODY GATE:' in x for x in prompts)
    assert 'FINAL_BATCH_NEW_SAFEGUARD' in p._get_revision_suggestion.call_args.kwargs['history'][-1]['content']
    saved = trace(tmp_path)
    assert saved['overlaps_final_update']
    assert saved['chunks'][0]['ready_seconds'] < saved['final_input_ready_seconds']
    assert 'body_reviews' not in saved
    assert prep._thread is None and p._listening_prefix is None


def test_late_input_failure_keeps_only_published_prefix(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    query, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    first = threading.Event()
    p.tts_chunk_callback = lambda *args: first.set()

    def complete_input():
        assert first.wait(3)
        raise RuntimeError('Final ASR failed')

    with pytest.raises(RuntimeError, match='Final ASR failed'):
        p.rebuttal_generation(history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete_input, listening_recognized_input=lambda: history)
    saved = trace(tmp_path)
    assert saved['status'] == 'failed' and len(saved['chunks']) == 1
    assert saved['committed_text'] == PREFIX
    query.assert_not_called()
    p._prepare_stage_prompt.assert_not_called()




@pytest.mark.parametrize('missing', ['audio', 'precheck', 'stage', 'turn', 'stance_conflict'])
def test_handoff_requires_matching_review_and_ready_audio(audio, tmp_path, missing):
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    prep, ready = ready_handoff(p, history, encoded)
    assert ready
    if missing == 'audio':
        prep._audio_latest = None
        prep._audio_futures.clear()
    elif missing == 'precheck':
        prep._latest['handoff_prechecked'] = False
    elif missing == 'stance_conflict':
        prep._latest['framework']['position'] = p.oppo_side
    stage = 'closing' if missing == 'stage' else p.status
    turn = 'wrong:opening' if missing == 'turn' else p.planner.turn
    assert prep.handoff(stage, turn, 60) is None
    prep.close()


def test_unprepared_prefix_waits_for_complete_input(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config = replace(p.streaming_output_config, listening_prefix_overlap_final_update=True)
    p.helper_client = prefix_helper(None)
    waiting, release = threading.Event(), threading.Event()
    p.tts_chunk_callback = Mock()

    def complete_input():
        waiting.set()
        assert release.wait(3)
        return history

    with ThreadPoolExecutor(max_workers=1) as pool:
        f = pool.submit(p.rebuttal_generation, history, 60, time_control=True,
                        listening_input_completion=complete_input)
        assert waiting.wait(3)
        p.tts_chunk_callback.assert_not_called()
        p.listen.assert_not_called()
        release.set()
        f.result(timeout=3)
    p.listen.assert_called_once_with(history)


@pytest.mark.parametrize('fail_previous', [False, True])
@pytest.mark.parametrize('parallel_feedback', [False, True])
def test_motion_starts_next_prefix_before_collecting_last_asr(monkeypatch, fail_previous, parallel_feedback):
    monkeypatch.setattr(live, 'TURNS', [('opening', 'for', 240), ('opening', 'against', 240)])
    monkeypatch.setattr(live, 'CONFIG', replace(live.CONFIG, listening_prefix_overlap_final_update=True, listening_parallel_body_feedback=parallel_feedback))
    ready = {'frozen': True}
    prep = SimpleNamespace(handoff=Mock(return_value=ready))
    players = {'for': SimpleNamespace(), 'against': SimpleNamespace(_listening_prefix=prep)}
    views = {'for': [], 'against': []}
    first_next, input_failed = threading.Event(), threading.Event()
    collected = []
    meter = live.Meter()

    def run(index, history, endpoint, *, playback_complete, input_completion, handoff, **extra):
        if index == 0:
            playback_complete.set_result(100.)
            assert first_next.wait(3), 'Next prefix waited for previous final ASR'
            if fail_previous:
                raise RuntimeError('Previous ASR failed')
            extra['transcript_ready'].set_result('Final ASR text.')
            return {'listener_transcript': 'Final ASR text.'}
        assert history == [] and endpoint == 100. and handoff == ready
        first_next.set()
        try:
            final = input_completion.result(timeout=3)
        except RuntimeError:
            input_failed.set()
            raise
        assert final == [{'content': 'Final ASR text.'}]
        if parallel_feedback:
            assert extra['recognized_input']() == [dict(stage='opening', side='for',
                content='Final ASR text.', tree_via_streaming=True)]
        playback_complete.set_result(200.)
        return {'listener_transcript': 'Next ASR text.'}

    def collect(index, row):
        collected.append(index)
        views['against'].append({'content': row['listener_transcript']})

    if fail_previous:
        with pytest.raises(RuntimeError, match='Previous ASR failed'):
            live.run_motion_turns(players, views, run, collect, meter)
        assert input_failed.is_set() and meter.stopped.is_set()
    else:
        live.run_motion_turns(players, views, run, collect, meter)
        assert collected == [0, 1]


def test_playback_endpoint_is_signaled_before_final_asr_drains(tmp_path, monkeypatch):
    speaker, listener, asr, _ = fake_turn(tmp_path, monkeypatch)
    endpoint = Future()
    original = asr.audio.transcriptions.create

    def transcribe(**kwargs):
        endpoint.result(timeout=3)
        return original(**kwargs)

    asr.audio.transcriptions.create = transcribe
    row = live.play_turn(0, speaker, listener, [], 240, None,
        SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter(), playback_complete=endpoint)
    assert endpoint.result() == row['playback_endpoint_monotonic']
    assert row['heard'][-1]['analysis_end_monotonic'] > endpoint.result()


@pytest.mark.parametrize('mismatch', [False, True])
@pytest.mark.parametrize('review_enabled', [False, True])
def test_body_feedback_overlaps_analysis_and_reuses_only_exact_input(audio, tmp_path, mismatch, review_enabled):
    import json
    from types import MethodType
    from ouragents import TreeDebater
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback='No changes',
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_body_feedback=True, listening_single_body_revision=True,
        listening_prefix_review_enabled=review_enabled)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p.high_quality_evidence_pool = []
    p.used_evidence = set()
    final = copy.deepcopy(history)
    final[-1]['content'] += ' FINAL_ASR_EXCEPTION.'
    recognized = Future()
    started, first_audio = threading.Event(), threading.Event()
    reviews = []
    base = p.helper_client

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            payload = json.loads(prompt.rsplit('\n', 1)[-1])
            reviews.append(prompt)
            if len(reviews) == 1:
                p.listen.assert_not_called()
                p._prepare_stage_prompt.assert_not_called()
            started.set()
            return [json.dumps(dict(points=[], other_corrections=['Keep the final exception.']))]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda *args: first_audio.set()
    def complete():
        assert first_audio.wait(3)
        assert started.wait(3), 'Body feedback waited for tree/planning completion'
        resolved = copy.deepcopy(final)
        if mismatch:
            resolved[-1]['content'] += ' ADDITIONAL_CORRECTION.'
        return resolved
    with ThreadPoolExecutor(max_workers=1) as pool:
        f = pool.submit(p.rebuttal_generation, history, 60, time_control=True,
            listening_handoff=handoff, listening_input_completion=complete,
            listening_recognized_input=recognized.result)
        try:
            assert first_audio.wait(3), 'Final ASR blocked prepared opening publication'
            assert not recognized.done()
            assert not started.is_set()
        finally:
            recognized.set_result(final)
        f.result(timeout=5)
    assert all(h['content'].replace('\n', ' ') in reviews[0] for h in final)
    assert len(reviews) == (2 if mismatch else 1)
    assert p._length_adjust.call_count == 1
    assert p._length_adjust.call_args.kwargs['max_retry'] == 1
    saved = trace(tmp_path)
    assert saved['parallel_body_feedback']['reused'] is (not mismatch)
    assert saved['parallel_body_feedback']['start_seconds'] < saved['final_input_ready_seconds']
    assert 'body_reviews' not in saved


def test_complete_asr_is_published_before_last_tree_update(tmp_path, monkeypatch):
    speaker, listener, asr, _ = fake_turn(tmp_path, monkeypatch)
    recognized = Future()
    def observe(text, side, stage):
        if 'slice2' in text:
            assert recognized.result(timeout=3) == 'Heard slice1. Heard slice2.'
    listener.observe_opponent.side_effect = observe
    row = live.play_turn(0, speaker, listener, [], 240, None,
        SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter(), transcript_ready=recognized)
    assert recognized.result() == row['listener_transcript']
    assert row['full_asr_ready_monotonic'] < row['heard'][-1]['analysis_end_monotonic']


def test_failed_asr_releases_recognition_waiter(tmp_path, monkeypatch):
    speaker, listener, asr, _ = fake_turn(tmp_path, monkeypatch, fail_asr=True)
    recognized = Future()
    with pytest.raises(RuntimeError):
        live.play_turn(0, speaker, listener, [], 240, None,
            SimpleNamespace(artifact={'blocked_dispatches': []}), asr, live.Meter(), transcript_ready=recognized)
    with pytest.raises(RuntimeError):
        recognized.result(timeout=1)


@pytest.mark.parametrize('metadata_padding', [0, .048])
@pytest.mark.parametrize('change', ['none', 'history', 'evidence', 'plan', 'exchange', 'system', 'temperature', 'reject'])
def test_endpoint_revision_and_tree_work_overlap_with_exact_reuse(audio, tmp_path, change, metadata_padding):
    import json
    from types import MethodType
    from ouragents import TreeDebater
    p, history, node = prepared_player(tmp_path)
    from utils.prompts.authoring import DEFAULT_DEBATER_SYSTEM
    p.system_prompt = DEFAULT_DEBATER_SYSTEM + '\nCUSTOM_DEBATER_INSTRUCTION'
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback='No changes',
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    final = copy.deepcopy(history)
    handoff['audio']['tts_out']['audio_seconds'] += metadata_padding
    final[-1]['content'] += ' FINAL_ASR_QUALIFICATION.'
    endpoint_started, revision_done, first_audio = (threading.Event() for _ in range(3))
    base = p.helper_client
    revision_prompts, endpoint_prompts = [], []

    def helper(*, prompt, **kwargs):
        if 'ENDPOINT GATE:' in prompt:
            endpoint_prompts.append(prompt)
            if len(endpoint_prompts) == 1:
                p.listen.assert_not_called()
                endpoint_started.set()
                assert not first_audio.is_set(), 'First audio preceded invalidated-snapshot review'
            if change == 'reject':
                result = json.loads(base(prompt=prompt, **kwargs)[0])
                conflict(result, json.loads(prompt.rsplit('\n', 1)[-1]))
                return [json.dumps(result)]
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert 'FINAL_ASR_QUALIFICATION' in prompt
            assert first_audio.wait(3)
            return [json.dumps(dict(points=[], other_corrections=['Keep the final qualification.']))]
        if kwargs.get('json_mode') is False:
            assert kwargs.get('sys') == p.system_prompt
            assert json.loads(prompt.split('Context and material (data):\n', 1)[1])['stage'] == 'rebuttal'
            revision_prompts.append(prompt)
            if len(revision_prompts) == 1:
                p.listen.assert_not_called()
                p._prepare_stage_prompt.assert_not_called()
                assert 'FINAL_ASR_QUALIFICATION' in kwargs['history_messages'][-1]['content']
                revision_done.set()
            return [TAIL]
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda *args: first_audio.set()

    def complete():
        assert first_audio.wait(3)
        assert revision_done.wait(3), 'Revision waited for final tree/planning completion'
        resolved = copy.deepcopy(final)
        if change == 'history':
            resolved[-1]['content'] += ' CORRECTED_TRANSCRIPT.'
        if change == 'plan':
            p.planner.state['body_plan'] = [dict(issue='New final issue', covers=[], target=None,
                action='answer_objection', point='Answer the final qualification.', weight=3)]
        if change == 'exchange':
            node.source_spans.append('FINAL_ASR_QUALIFICATION')
        if change == 'evidence':
            p.high_quality_evidence_pool = [dict(id=7, content='New supplied evidence.')]
        if change == 'system':
            p.system_prompt += '\nUPDATED_DEBATER_INSTRUCTION'
        if change == 'temperature':
            p.config.temperature = .73
        p.planner.chunks = [resolved[-1]['content']]
        return resolved

    call = lambda: p.rebuttal_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete,
        listening_recognized_input=lambda: copy.deepcopy(final))
    if change == 'reject':
        handoff['candidate']['review_stamp'] = 'invalidated-preparation-review'
        with pytest.raises(SegmentRejected):
            call()
        assert not first_audio.is_set() and not revision_prompts
        assert trace(tmp_path)['chunks'] == []
        return
    assert TAIL in call()
    saved = trace(tmp_path)
    assert saved['parallel_body_revision']['start_seconds'] < saved['final_input_ready_seconds']
    assert len(revision_prompts) == (2 if change in ('history', 'evidence', 'system', 'temperature') else 1)
    assert not endpoint_prompts
    assert saved['parallel_body_revision']['reused'] is (change in ('none', 'plan', 'exchange'))
    if change == 'exchange':
        assert saved['parallel_body_feedback']['reused']
    if change in ('plan', 'exchange'):
        assert saved['body_task_binding']['final_derived_context_changed'] is (change == 'exchange')
        assert saved['body_task_binding']['context']['history'] == final
    assert 'body_reviews' not in saved


@pytest.mark.parametrize(('blocked_work', 'change'), [
    ('feedback', 'history'), ('revision', 'history'),
    ('revision', 'evidence'), ('revision', 'system'), ('evidence', 'evidence'),
])
def test_obsolete_body_work_does_not_block_first_body_audio(audio, tmp_path, change, blocked_work, monkeypatch):
    """Release obsolete work only AFTER final-input body audio has been emitted."""
    if blocked_work == 'evidence':
        # Keep exercising the retained supplement implementation while its
        # production dispatch is temporarily disabled.
        monkeypatch.setattr('streaming.listening_prefix.ENABLE_ENDPOINT_EVIDENCE_SUPPLEMENT', True)
    import json
    from types import MethodType
    from ouragents import TreeDebater
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback='No changes',
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    final = copy.deepcopy(history)
    final[-1]['content'] += ' COMPLETE_ASR_QUALIFICATION.'
    stale_started, body_emitted, first_audio = (threading.Event() for _ in range(3))
    base = p.helper_client
    feedbacks, revisions, gates = [], [], []
    if blocked_work == 'evidence':
        p.high_quality_evidence_pool = [dict(id=str(i), content=f'Old evidence {i}') for i in range(11)]
        # A thread-ordering assertion must fail immediately, not enter the native
        # model-format retry loop and its 30-second backoff.
        def select_once(helper, prompt, key, **kwargs):
            raw = helper(prompt=prompt, **kwargs)[0]
            return json.loads(raw)[key], raw
        monkeypatch.setattr('ouragents.get_response_with_retry', select_once)

    def helper(*, prompt, **kwargs):
        if blocked_work == 'evidence' and prompt.startswith('From the provided list of evidence dictionaries'):
            stale_started.set()
            assert body_emitted.wait(3), 'Obsolete evidence selection blocked final body audio'
            return [json.dumps(dict(selected_ids=['0']))]
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            feedbacks.append(prompt)
            assert first_audio.wait(3)
            if blocked_work == 'feedback' and len(feedbacks) == 1:
                stale_started.set()
                assert body_emitted.wait(3), 'Obsolete feedback blocked final body audio'
            return [json.dumps(dict(points=[], other_corrections=['Keep the final qualification.']))]
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            if blocked_work == 'revision' and len(revisions) == 1:
                stale_started.set()
                assert body_emitted.wait(3), 'Obsolete revision blocked final body audio'
            return [TAIL]
        if 'LISTENING FINAL BODY GATE:' in prompt:
            gates.append(prompt)
        return base(prompt=prompt, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda index, *args: body_emitted.set() if index > 0 else first_audio.set()

    def complete():
        assert stale_started.wait(3)
        resolved = copy.deepcopy(final)
        if change == 'history':
            resolved[-1]['content'] += ' CORRECTED_TRANSCRIPT.'
        if change == 'evidence':
            p.high_quality_evidence_pool = [dict(id=7, content='New supplied evidence.')]
        if change == 'system':
            p.system_prompt = 'Updated system instructions.'
        p.planner.chunks = [resolved[-1]['content']]
        return resolved

    assert TAIL in p.rebuttal_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete,
        listening_recognized_input=lambda: copy.deepcopy(final))
    saved = trace(tmp_path)
    assert body_emitted.is_set()
    assert len(feedbacks) == (2 if change == 'history' else 1)
    assert gates == []
    assert any('COMPLETE_ASR_QUALIFICATION' in p for p in feedbacks)
    if change == 'history':
        assert any('CORRECTED_TRANSCRIPT' in p for p in feedbacks)
        assert not saved['parallel_body_feedback']['reused']
    if blocked_work == 'revision':
        assert len(revisions) == 2
        assert not saved['parallel_body_revision']['reused']
        assert saved['chunks'][1]['ready_seconds'] < saved['parallel_body_revision']['end_seconds']


def test_matching_inflight_revision_is_joined_only_after_feedback_is_available(audio, tmp_path):
    import json
    from types import MethodType
    from ouragents import TreeDebater
    p, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    _, handoff = ready_handoff(p, history, encoded)
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback='No changes',
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p.high_quality_evidence_pool, p.used_evidence = [], set()
    first_audio, revision_started, committed_revision = (threading.Event() for _ in range(3))
    base, revisions = p.helper_client, []

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert first_audio.wait(3)
            return [json.dumps(dict(points=[], other_corrections=['Keep the qualification.']))]
        if kwargs.get('json_mode') is False:
            revisions.append(prompt)
            revision_started.set()
            assert committed_revision.wait(3), 'Feedback was blocked behind revision completion'
            return [TAIL]
        return base(prompt=prompt, **kwargs)

    def adjust(*args, **kwargs):
        committed_revision.set()
        return TreeDebater._length_adjust(p, *args, **kwargs)

    p.helper_client = Mock(side_effect=helper)
    p._length_adjust = adjust
    p.tts_chunk_callback = lambda *args: first_audio.set()

    def complete():
        assert revision_started.wait(3)
        p.planner.state['body_plan'] = [dict(issue='Reordered issue', covers=[], target=None,
            action='answer_objection', point='Keep the qualification.', weight=3)]
        p.planner.chunks = [history[-1]['content']]
        return copy.deepcopy(history)

    assert TAIL in p.rebuttal_generation(history, 60, time_control=True,
        listening_handoff=handoff, listening_input_completion=complete,
        listening_recognized_input=lambda: copy.deepcopy(history))
    assert len(revisions) == 1
    saved = trace(tmp_path)
    assert saved['parallel_body_feedback']['reused']
    assert saved['parallel_body_revision']['reused']
    assert not saved['body_task_binding']['final_derived_context_changed']  # Private tactics are excluded.


@pytest.mark.parametrize('fail_audio', [False, True])
def test_inflight_prefix_audio_survives_freeze_and_precedes_final_input(audio, tmp_path, fail_audio):
    p, history, _ = prepared_player(tmp_path)
    query, encoded = audio
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_prefix_overlap_final_update=True, listening_prefix_pre_synthesize=True)
    p.helper_client = prefix_helper(None)
    synthesis_started, release_audio, first_audio = (threading.Event() for _ in range(3))

    def synthesize(text, config):
        synthesis_started.set()
        assert release_audio.wait(3)
        if fail_audio:
            raise RuntimeError('In-flight synthesis failed')
        return encoded(2)

    synthesize = Mock(side_effect=synthesize)
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config, synthesize)
    p._listening_prefix = prep
    prep.offer(material(p, p.status, history))
    assert synthesis_started.wait(3)
    handoff = prep.handoff(p.status, p.planner.turn, 60)
    assert handoff and handoff['audio'] is None
    assert not handoff['audio_future'].done()

    def complete_input():
        assert first_audio.wait(3), 'Pending synthesis fell back to complete input'
        return history

    def emit(index, *args):
        if index == 0:
            p.listen.assert_not_called()
            query.assert_not_called()
            first_audio.set()

    p.tts_chunk_callback = emit
    release_audio.set()  # Resolves AFTER freeze: old cache discarded this result.
    try:
        if fail_audio:
            with pytest.raises(RuntimeError, match='In-flight synthesis failed'):
                p.rebuttal_generation(history, 60, time_control=True,
                    listening_handoff=handoff, listening_input_completion=complete_input,
                    listening_recognized_input=lambda: history)
            assert not first_audio.is_set()
            assert trace(tmp_path)['prefix_audio_transfer']['status'] == 'failed'
        else:
            assert TAIL in p.rebuttal_generation(history, 60, time_control=True,
                listening_handoff=handoff, listening_input_completion=complete_input,
                listening_recognized_input=lambda: history)
            assert first_audio.is_set()
            assert trace(tmp_path)['prefix_audio_transfer']['status'] == 'reused'
        synthesize.assert_called_once()
        assert all(call.args[1] != PREFIX for call in query.call_args_list)
    finally:
        release_audio.set()
        prep.close()
