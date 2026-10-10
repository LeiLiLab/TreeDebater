"""Whole-script delivery, immutable opening and real overlap without API calls."""
from io import BytesIO
import json
import threading
from unittest.mock import Mock

from pydub import AudioSegment
import pytest

pytestmark = pytest.mark.usefixtures("word_length_modes")

from agents import Debater
from streaming.config import OutputConfig
from streaming.full_speech import remaining_text, split_opening
from test_flat_speaking import speaker
import tts_streaming as tts


@pytest.fixture
def audio(monkeypatch):
    monkeypatch.setattr(tts, 'OpenAI', Mock())
    cache = {}
    def encoded(seconds=.1):
        if seconds not in cache:
            buffer = BytesIO()
            AudioSegment.silent(duration=int(seconds * 1000)).export(buffer, format='mp3')
            cache[seconds] = dict(mp3_bytes=buffer.getvalue(), audio_seconds=seconds,
                                  tts_api_s=.001, mp3_parse_s=.001)
        return dict(cache[seconds])
    query = Mock(side_effect=lambda *args, **kwargs: encoded())
    monkeypatch.setattr(tts, '_query_time_profiled', query)
    monkeypatch.setattr(tts, '_tts_with_retry', query)
    return query, encoded


def overlap_speaker(tmp_path, **config):
    player, history, _, _ = speaker()
    player.streaming_output_config = OutputConfig(
        speech_mode='overlap_prefix', budget_mode='audio_duration', adaptive_delivery=True,
        normalize_seams=False, max_refinements=0, early_max_refinements=0,
        first_chunk_seconds=8, **config)
    player.audio_output_dir = str(tmp_path)
    draft = 'Who funds the trial?\n\nThe complete remaining argument needs costings before launch.'
    player._get_response.return_value = '**Statement**\n'+draft
    player._get_revision_suggestion = Mock(return_value=('Keep conditions.', [], '', draft))
    player.helper_client.return_value = ['Who funds the trial?']
    player._length_adjust = Mock(return_value='The complete remaining argument needs costings before launch.')
    return player, history


def test_full_script_is_default_and_incremental_is_explicit(monkeypatch):
    player, history, _, _ = speaker()
    player.streaming_output_config = OutputConfig()
    player._speak_flat_streaming = Mock(side_effect=AssertionError('Unexpected incremental speech'))
    player._get_revision_suggestion = Mock(return_value=('', [], '', 'Whole draft.'))
    player._length_adjust = Mock(return_value='Whole revised speech.')
    monkeypatch.setattr(Debater, 'post_process', lambda self, statement, *args, **kwargs: statement)
    assert player.speak('Speak.', 60, time_control=True, history=history) == 'Whole revised speech.'
    player._get_response.assert_called_once()
    assert player._length_adjust.call_args.args[0] == 'Whole draft.'


def test_opening_tts_and_tail_revision_overlap_and_publish_before_tail_finishes(audio, tmp_path):
    player, history = overlap_speaker(tmp_path)
    revision_started, first_published = threading.Event(), threading.Event()
    query, encoded = audio
    def revise(*args, **kwargs):
        assert kwargs['frozen_prefix'] == 'Who funds the trial?'
        revision_started.set()
        assert first_published.wait(3), 'Tail blocked first audio'
        return 'The remaining argument needs costings before launch.'
    player._length_adjust.side_effect = revise
    def synthesize(*args, **kwargs):
        assert revision_started.wait(3), 'First TTS did not overlap tail revision'
        return encoded()
    query.side_effect = synthesize
    delivered = []
    def ready(index, path, text, duration):
        delivered.append(text)
        if index == 0:
            assert not first_published.is_set()
            assert text == 'Who funds the trial?'
            first_published.set()
    player.tts_chunk_callback = ready
    response = player.speak('Speak.', 60, time_control=True, history=history)
    assert response == '\n\n'.join(delivered)
    assert len(delivered) == 2
    assert [m['content'] for m in player.conversation if m['role'] == 'assistant'] == [response]
    trace = json.loads(next(tmp_path.glob('*_chunks/overlap_prefix.json')).read_text())
    assert trace['status'] == 'completed'
    assert trace['tail_revision_start_seconds'] <= trace['chunks'][0]['ready_seconds']
    assert trace['chunks'][0]['ready_seconds'] <= trace['tail_revision_end_seconds']


def test_failed_tail_preserves_audio_and_exact_published_transcript(audio, tmp_path):
    player, history = overlap_speaker(tmp_path)
    player._length_adjust.side_effect = RuntimeError('Tail failed')
    seen = []
    player.tts_chunk_callback = lambda i, p, text, d: seen.append(text)
    with pytest.raises(RuntimeError, match='Tail failed'):
        player.speak('Speak.', 60, time_control=True, history=history)
    assert seen == ['Who funds the trial?']
    assert player.conversation[-1]['content'] == seen[0]
    trace = json.loads(next(tmp_path.glob('*_chunks/overlap_prefix.json')).read_text())
    assert trace['status'] == 'failed' and trace['committed_text'] == seen[0]
    assert len(AudioSegment.from_file(next(tmp_path.glob('*.mp3')))) > 0


def test_overlong_opening_falls_back_before_any_prefix_is_published(audio, tmp_path):
    player, history = overlap_speaker(tmp_path)
    player._get_revision_suggestion.return_value = (
        'Shorten the opening.', [], '', 'Too long ' * 500 + '.\n\nRemaining argument.')
    delivered = []
    player.tts_chunk_callback = lambda i, p, text, d: delivered.append(text)
    response = player.speak('Speak.', 60, time_control=True, history=history)
    assert 'Too long' not in response
    assert response == '\n\n'.join(delivered)
    assert 'frozen_prefix' not in player._length_adjust.call_args.kwargs
    assert player._length_adjust.call_args.args[0].startswith('Too long ')
    trace = json.loads(next(tmp_path.glob('*_chunks/overlap_prefix.json')).read_text())
    assert trace['status'] == 'fallback_full_script'


def test_callback_failure_stores_prefix_and_does_not_replay(audio, tmp_path):
    player, history = overlap_speaker(tmp_path)
    callback = Mock(side_effect=OSError('Player disconnected'))
    player.tts_chunk_callback = callback
    with pytest.raises(OSError, match='Player disconnected'):
        player.speak('Speak.', 60, time_control=True, history=history)
    assert callback.call_count == 1
    assert player.conversation[-1]['content'] == 'Who funds the trial?'


def test_frozen_prefix_is_not_rewritten_but_tail_adapts_to_actual_audio(audio, tmp_path, monkeypatch):
    query, encoded = audio
    prefix, body = 'Funding remains unresolved.', 'The remaining argument is much longer than its budget.'
    rewrite = Mock(return_value='A shorter remaining argument.')
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    def synthesize(client, text, *args, **kwargs):
        return encoded(1 if text == prefix else 10 if text == body else 4.9)
    query.side_effect = synthesize
    delivered = []
    tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 6,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False,
                            first_chunk_seconds=1, max_refinements=1, early_max_refinements=1,
                            speed_adjust_min=1, speed_adjust_max=1, max_parallel_tts=1),
        tail_supplier=lambda: body,
        on_chunk=lambda i, p, text, duration: delivered.append(text))
    assert delivered == [prefix, 'A shorter remaining argument.']
    rewrite.assert_called_once()
    assert rewrite.call_args.args[1] == body
    assert rewrite.call_args.args[2] == round(len(body.split()) * 5 / 10)
    assert rewrite.call_args.args[3] == [prefix]


def test_pipeline_joins_workers_even_when_callback_fails(monkeypatch):
    finished = threading.Event()
    thread = threading.Thread(target=lambda: finished.set())
    executor = Mock()
    def run(*args, _contexts, **kwargs):
        from types import SimpleNamespace
        _contexts.append(SimpleNamespace(stop_event=threading.Event(), workers=[thread], executor=executor, config=OutputConfig()))
        thread.start()
        raise RuntimeError('Publication failed')
    monkeypatch.setattr(tts, '_run_pipeline', run)
    with pytest.raises(RuntimeError, match='Publication failed'):
        tts.run_pipeline(None, [], 60)
    assert finished.is_set() and not thread.is_alive()
    executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)


def test_validation_rejects_adaptive_prefix_echo_before_file_publication(audio, tmp_path):
    prefix = 'Funding remains unresolved.'
    seen = []
    def validate(index, text):
        if index > 0 and prefix in text:
            raise ValueError('Repeated prefix')
    with pytest.raises(ValueError, match='Repeated prefix'):
        tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 60,
            config=OutputConfig(adaptive_delivery=True, normalize_seams=False,
                                max_refinements=0, early_max_refinements=0),
            tail_supplier=lambda: prefix, validate_chunk=validate,
            on_chunk=lambda i, p, text, d: seen.append(text))
    assert seen == [prefix]
    assert (tmp_path / 'speech_chunks/chunk_000.mp3').exists()
    assert not (tmp_path / 'speech_chunks/chunk_001.mp3').exists()


def test_prefix_echo_is_removed_once_and_later_duplicates_are_rejected():
    prefix = 'Who funds the trial?'
    assert remaining_text(prefix+'\n\nDiscuss remaining costs.', prefix) == 'Discuss remaining costs.'
    with pytest.raises(ValueError, match='repeats'):
        remaining_text('Discuss costs. '+prefix, prefix)
    assert split_opening('First complete sentence. Another complete sentence.') == (
        'First complete sentence.', 'Another complete sentence.')


@pytest.mark.parametrize('mode', ['full_script', 'overlap_prefix', 'listening_prefix', 'incremental'])
def test_speech_mode_validation(mode):
    assert OutputConfig(speech_mode=mode).speech_mode == mode
    with pytest.raises(ValueError, match='speech_mode'):
        OutputConfig(speech_mode='typo')


@pytest.mark.parametrize('mode', ['fixed_prefix', 'typo'])
def test_retired_or_unknown_speech_mode_is_rejected(mode):
    with pytest.raises(ValueError, match='speech_mode'):
        OutputConfig(speech_mode=mode)


def test_disabled_refinement_publishes_out_of_range_audio_without_deadline_wait(audio, tmp_path):
    import csv
    import time
    query, encoded = audio
    query.side_effect = lambda *args, **kwargs: encoded(8)
    arrivals = []
    tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'), 10,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False,
            max_refinements=0, early_max_refinements=0, first_chunk_seconds=8,
            allow_expansion=False, speed_adjust_min=1, speed_adjust_max=1),
        tail_supplier=lambda: 'Remaining argument.',
        on_chunk=lambda *args: arrivals.append(time.monotonic()))
    assert len(arrivals) == 2
    assert arrivals[1] - arrivals[0] < 2
    rows = list(csv.DictReader((tmp_path/'speech_chunks/chunk_profile.csv').open()))
    assert rows[1]['target_reached'] == 'False'
    assert rows[1]['timed_out'] == 'False'
    assert rows[1]['audio_seconds'] == '8.0'
