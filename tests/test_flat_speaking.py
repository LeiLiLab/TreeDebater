"""Offline speech publication, grounding, routing and failure regressions."""
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("word_length_modes")
from pydub import AudioSegment

from agents import Debater
from streaming.config import OutputConfig
from streaming.flat_speaking import FlatSpeechProducer, SegmentRejected
from test_claim_constraints import condition, player, propose, trees
import tts_streaming as tts


def reviewed(prompt, *, condition_status='preserved', assertion_status='conditional', continuation=True):
    data = json.loads(prompt.rsplit('\n', 1)[-1])
    return json.dumps({
        'continuation_ok': continuation, 'issues': [],
        'checks': [{'id': c['constraint_id'], 'status': condition_status,
                    'draft_quote': data['draft'], 'reason': 'Checked scope.', 'fix': ''}
                   for c in data['checklist']],
        'assertions': [{'sentence': i, 'status': assertion_status, 'source_quote': '',
                        'reason': 'Conditional question.', 'fix': ''}
                       for i in range(len(data['sentences']))]})


def speaker():
    pair = trees()
    node = propose(pair, 'A free six-week trial, with independent costings first.',
                   [condition('with independent costings first.')])
    other = propose(pair, 'School visits require an appointment.',
                    [condition('School visits require an appointment.')])
    p = player(pair, 'flat_tree')
    p.config = SimpleNamespace(streaming_tts=True, single_pass_revision=True, type='treedebater')
    p.streaming_output_config = OutputConfig(speech_mode='incremental')
    p.simulated_audience = [SimpleNamespace(feedback=Mock(side_effect=reviewed))]
    p.helper_client = Mock(return_value=['With costings first, how will the free trial be assessed?'])
    p._get_response = Mock(return_value=json.dumps({
        'text': 'With costings first, how will the free trial be assessed?',
        'target_ids': [node.node_id], 'done': False}))
    history = [{'side': 'for', 'stage': 'opening', 'content': p.planner.chunks[0]}]
    return p, history, node, other


def test_review_before_commit_and_next_argument_receives_immutable_prefix():
    p, history, node, other = speaker()
    producer = FlatSpeechProducer(p, history)
    first = producer.prepare(12)
    assert producer.text == '' and p.conversation == []
    assert p.helper_client.call_count == 0
    review_data = json.loads(p.simulated_audience[0].feedback.call_args.args[0].rsplit('\n', 1)[-1])
    assert {c['node_id'] for c in review_data['checklist']} == {node.node_id}
    assert other.node_id in {c['node_id'] for c in review_data['context']['condition_candidates']}
    assert all(c['planned_target'] for c in review_data['checklist'])
    producer.commit(first)
    p._get_response.return_value = json.dumps({'text': 'Could appointment scheduling limit access?',
                                             'target_ids': [other.node_id], 'done': True})
    second = producer.prepare(20)
    draft_data = json.loads(p._get_response.call_args.args[0][-1]['content'].rsplit('\n', 1)[-1])
    assert draft_data['published_prefix'] == first
    assert all('ancestors' not in t and 'responses' not in t for t in draft_data['current_targets'])
    producer.commit(second)
    assert producer.text == first + '\n\n' + second
    assert producer.done and producer.prepare(20) is None


@pytest.mark.parametrize('fault', ['missing_condition', 'unsupported', 'malformed', 'continuation', 'omitted_sentence'])
def test_bad_reviews_fail_closed_after_one_repair(fault):
    p, history, _, _ = speaker()
    def feedback(prompt):
        if fault == 'malformed':
            return 'Looks fine.'
        if fault == 'omitted_sentence':
            data = json.loads(reviewed(prompt))
            data['assertions'] = []
            return json.dumps(data)
        return reviewed(prompt, condition_status='missing' if fault == 'missing_condition' else 'preserved',
                        assertion_status='unsupported' if fault == 'unsupported' else 'conditional',
                        continuation=fault != 'continuation')
    p.simulated_audience[0].feedback.side_effect = feedback
    producer = FlatSpeechProducer(p, history)
    with pytest.raises(SegmentRejected, match='failed review'):
        producer.prepare(12)
    assert producer.text == '' and producer.pending is None
    assert p.helper_client.call_count == 1
    assert p.simulated_audience[0].feedback.call_count == 2


def test_repair_is_rechecked_and_only_repaired_text_can_be_published():
    p, history, _, _ = speaker()
    p.simulated_audience[0].feedback.side_effect = lambda prompt: reviewed(
        prompt, assertion_status='unsupported' if p.simulated_audience[0].feedback.call_count == 1 else 'conditional')
    producer = FlatSpeechProducer(p, history)
    text = producer.prepare(12)
    assert p.simulated_audience[0].feedback.call_count == 2
    with pytest.raises(SegmentRejected, match='modified text'):
        producer.commit(text + ' An unreviewed assertion.')
    producer.commit(text)
    assert producer.text == p.helper_client.return_value[0]


@pytest.mark.parametrize('mutation', ['sources', 'conditions'])
def test_changed_source_or_condition_invalidates_pending_audio(mutation):
    p, history, node, _ = speaker()
    producer = FlatSpeechProducer(p, history)
    text = producer.prepare(12)
    if mutation == 'sources':
        node.argument.append('The trial now lasts twelve weeks.')
    else:
        node.constraints = []
    with pytest.raises(SegmentRejected, match='Sources changed'):
        producer.commit(text)
    assert producer.text == ''


@pytest.mark.parametrize('ids', [[], ['invented'], [1], None])
def test_invalid_targets_cannot_skip_condition_review(ids):
    p, history, _, _ = speaker()
    p._get_response.return_value = json.dumps({'text': 'A claim.', 'target_ids': ids, 'done': True})
    with pytest.raises(SegmentRejected, match='target IDs'):
        FlatSpeechProducer(p, history).prepare(12)
    p.simulated_audience[0].feedback.assert_not_called()


class AudioProducer:
    def __init__(self, events, texts=('First complete argument.', 'Second complete argument.')):
        self.events, self.texts = events, iter(texts)
        self.committed, self.done = [], False
        self.pending = None

    @property
    def text(self):
        return '\n\n'.join(self.committed)

    def prepare(self, seconds, **kwargs):
        self.events.append(('prepare', seconds, self.text))
        text = next(self.texts)
        if isinstance(text, Exception):
            raise text
        self.pending = text
        return text

    def validate(self, text):
        assert text == self.pending

    def commit(self, text, *, publish=None):
        self.validate(text)
        if publish is not None:
            publish()
        self.events.append(('commit', text))
        self.committed.append(text)
        self.pending = None
        self.done = len(self.committed) == 2


@pytest.fixture
def audio(monkeypatch):
    monkeypatch.setattr(tts, 'OpenAI', Mock())
    def encode(duration):
        stream = BytesIO()
        AudioSegment.silent(duration=duration).export(stream, format='mp3')
        return {'mp3_bytes': stream.getvalue()}
    result = encode(1000)
    query = Mock(return_value=result)
    monkeypatch.setattr(tts, '_tts_with_retry', query)
    return query, encode


def test_first_audio_published_before_next_generation_and_exact_transcript_saved(audio, tmp_path):
    events = []
    producer = AudioProducer(events)
    def published(index, path, text, duration):
        assert path.read_bytes()
        assert path.with_suffix('.txt').read_text() == text
        assert text == producer.committed[-1]
        events.append(('published', index, text))
    output = tmp_path / 'speech.mp3'
    text, _, duration = tts.convert_incremental_speech_to_audio(producer, output, 60,
        config=OutputConfig(budget_mode='audio_duration', normalize_seams=False), on_chunk=published)
    assert [e[0] for e in events] == ['prepare', 'commit', 'published', 'prepare', 'commit', 'published']
    assert events[3][2] == producer.committed[0]
    assert text == producer.text and duration == 2
    assert len(AudioSegment.from_file(output)) == 2000
    trace = json.loads((tmp_path / 'speech_chunks/speaking.json').read_text())
    assert trace['status'] == 'completed' and trace['committed_text'] == text
    assert trace['first_audio_seconds'] <= trace['chunks'][1]['prepare_start_seconds']
    assert [call.args[1] for call in audio[0].call_args_list] == producer.committed


@pytest.mark.parametrize('failure', ['generation', 'callback'])
def test_failure_preserves_prefix_without_replaying_or_publishing_unchecked_audio(audio, tmp_path, failure):
    events = []
    producer = AudioProducer(events, ('First complete argument.', SegmentRejected('Rejected next argument')))
    callback = Mock(side_effect=RuntimeError('Playback failed') if failure == 'callback' else None)
    with pytest.raises((SegmentRejected, RuntimeError)):
        tts.convert_incremental_speech_to_audio(producer, tmp_path / 'speech.mp3', 60,
            config=OutputConfig(normalize_seams=False), on_chunk=callback)
    assert callback.call_count == 1 and audio[0].call_count == 1
    assert producer.text == 'First complete argument.'
    assert (tmp_path / 'speech.mp3').exists()
    assert not (tmp_path / 'speech_chunks/chunk_001.mp3').exists()
    trace = json.loads((tmp_path / 'speech_chunks/speaking.json').read_text())
    assert trace['status'] == 'failed' and trace['committed_text'] == producer.text


def test_overlong_audio_is_replaced_with_fresh_checked_candidate_without_trimming(audio, tmp_path):
    query, encode = audio
    query.side_effect = [encode(8000), encode(3000)]
    producer = AudioProducer([], ('Too long complete argument.', 'Short complete argument.'))
    callback = Mock()
    text, _, duration = tts.convert_incremental_speech_to_audio(producer, tmp_path / 'speech.mp3', 5,
        config=OutputConfig(normalize_seams=False, budget_mode='audio_duration'), on_chunk=callback)
    assert text == 'Short complete argument.' and duration == 3
    assert callback.call_count == 1 and query.call_count == 2
    assert producer.committed == ['Short complete argument.']
    assert producer.events[1][1] < producer.events[0][1]


@pytest.mark.parametrize('mode,expected_count', [('audio_duration', 2), ('experiment_elapsed', 1)])
def test_elapsed_budget_counts_generation_gaps(audio, tmp_path, monkeypatch, mode, expected_count):
    clock = [0.0]
    monkeypatch.setattr(tts, '_now', lambda: clock[0])
    producer = AudioProducer([])
    def published(*args):
        clock[0] += 20
    tts.convert_incremental_speech_to_audio(producer, tmp_path / 'speech.mp3', 10,
        config=OutputConfig(normalize_seams=False, budget_mode=mode), on_chunk=published)
    assert len(producer.committed) == expected_count


@pytest.mark.parametrize('mode,configured,override,time_control,enabled', [
    ('flat_tree', True, None, True, True), ('flat_tree', False, True, True, True),
    ('flat_tree', True, False, True, False), ('flat_tree', True, None, False, False),
    ('flat_tree', False, None, True, False), ('linear', True, None, True, False),
    ('legacy', True, None, True, False),
])
def test_existing_streaming_switch_routes_only_flat_audio_turns(
        monkeypatch, mode, configured, override, time_control, enabled):
    p, history, _, _ = speaker()
    p.config.streaming_tts = configured
    p.planner.config.mode = mode
    p._speak_flat_streaming = Mock(return_value='Streamed speech.')
    p._get_revision_suggestion = Mock(return_value=('', [], '', 'Batch draft.'))
    p._length_adjust = Mock(return_value='Batch speech.')
    monkeypatch.setattr(Debater, 'post_process', lambda self, statement, *a, **kw: statement)
    if mode != 'flat_tree':
        with pytest.raises(ValueError, match='requires planning mode flat_tree'):
            p.speak('Speak now.', 60, time_control=time_control, history=history, streaming_tts=override)
        p._get_response.assert_not_called()
        p._speak_flat_streaming.assert_not_called()
        return
    result = p.speak('Speak now.', 60, time_control=time_control, history=history, streaming_tts=override)
    assert (result == 'Streamed speech.') == enabled
    assert p._speak_flat_streaming.call_count == int(enabled)
    assert p._get_response.call_count == int(not enabled)


def test_real_flat_speak_uses_existing_audio_callback_and_stores_only_published_text(audio, tmp_path):
    p, history, _, other = speaker()
    first = json.loads(p._get_response.return_value)
    p._get_response.side_effect = [json.dumps(first), json.dumps({
        'text': 'Could appointments limit access?', 'target_ids': [other.node_id], 'done': True})]
    p.audio_output_dir = str(tmp_path)
    p.streaming_output_config = OutputConfig(speech_mode='incremental', normalize_seams=False, budget_mode='audio_duration')
    seen = []
    def ready(index, path, text, duration):
        seen.append(text)
        assert p._get_response.call_count == index + 1
    p.tts_chunk_callback = ready
    result = p.speak('Speak now.', 60, time_control=True, history=history)
    assert result == '\n\n'.join(seen)
    assert [m['content'] for m in p.conversation if m['role'] == 'assistant'] == [result]
    assert (tmp_path / 'treedebater_rebuttal_against_chunks/chunk_000.mp3').exists()
    assert p.helper_client.call_count == 0


def test_failed_publication_does_not_commit_unheard_text():
    p, history, _, _ = speaker()
    producer = FlatSpeechProducer(p, history)
    text = producer.prepare(12)
    with pytest.raises(OSError):
        producer.commit(text, publish=Mock(side_effect=OSError('disk full')))
    assert producer.text == ''


def test_segment_protocol_excludes_whole_speech_instructions_but_retains_debate_evidence():
    p, history, _, _ = speaker()
    p.conversation = [{'role': 'user', 'content':
                       'Output **Rebuttal Plan** and **Statement** in 131 words.'}]
    p.main_claims_content = ['Our previously selected claim.']
    FlatSpeechProducer(p, history).prepare(12)
    messages = p._get_response.call_args.args[0]
    assert [m['role'] for m in messages] == ['system', 'user']
    assert '**Rebuttal Plan**' not in str(messages)
    data = json.loads(messages[-1]['content'].rsplit('\n', 1)[-1])
    assert data['debate_history'] == history
    assert data['our_main_claims'] == p.main_claims_content
    assert data['target_words'] == 26
    assert p._get_response.call_args.kwargs['response_format'] == {'type': 'json_object'}


def test_source_change_during_tts_blocks_audio_publication(audio, tmp_path):
    p, history, node, _ = speaker()
    producer = FlatSpeechProducer(p, history)
    query, encode = audio
    def changed(*args, **kwargs):
        node.constraints = []
        return encode(1000)
    query.side_effect = changed
    callback = Mock()
    with pytest.raises(SegmentRejected, match='Sources changed'):
        tts.convert_incremental_speech_to_audio(producer, tmp_path / 'speech.mp3', 60,
            config=OutputConfig(normalize_seams=False), on_chunk=callback)
    assert producer.text == ''
    callback.assert_not_called()
    assert not (tmp_path / 'speech_chunks/chunk_000.mp3').exists()


def test_real_speak_records_partial_transcript_and_never_retries_full_speech(audio, tmp_path):
    p, history, _, _ = speaker()
    first = p._get_response.return_value
    p._get_response.side_effect = [first, 'Malformed second chunk']
    p.audio_output_dir = str(tmp_path)
    p.streaming_output_config = OutputConfig(speech_mode='incremental', normalize_seams=False, budget_mode='audio_duration')
    p.tts_chunk_callback = Mock()
    with pytest.raises(SegmentRejected):
        p.speak('Speak now.', 60, time_control=True, history=history)
    assert p.tts_chunk_callback.call_count == 1
    assert p._get_response.call_count == 2
    spoken = p.tts_chunk_callback.call_args.args[2]
    assert [m['content'] for m in p.conversation if m['role'] == 'assistant'] == [spoken]


@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_no_extracted_targets_still_reviews_raw_sources_and_every_sentence(stage):
    p, history, _, _ = speaker()
    p.status = stage
    p.debate_tree, p.oppo_debate_tree = trees()
    p._get_response.return_value = json.dumps({'text': 'How would that trial be assessed?',
                                             'target_ids': [], 'done': True})
    producer = FlatSpeechProducer(p, history)
    text = producer.prepare(12)
    data = json.loads(p.simulated_audience[0].feedback.call_args.args[0].rsplit('\n', 1)[-1])
    assert data['sentences'] == [text]
    assert data['context']['opponent_sources'] == [history[0]['content']]
    producer.commit(text)
    assert producer.done
