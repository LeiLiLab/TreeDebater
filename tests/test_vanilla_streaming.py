import json
from io import BytesIO
import threading
from types import SimpleNamespace

import pytest
from pydub import AudioSegment

from agents import Debater, DebaterConfig
from streaming.config import OutputConfig
from streaming.vanilla import Segmenter, generate


def player(tmp_path, monkeypatch):
    import tts_streaming
    monkeypatch.setattr(tts_streaming, 'create_tts_client', lambda config: None)
    def synthesize(client, text, config, **kwargs):
        assert kwargs == {'retry': False}
        buf = BytesIO()
        AudioSegment.silent(duration=100).export(buf, format='mp3')
        return {'mp3_bytes': buf.getvalue()}
    monkeypatch.setattr(tts_streaming, 'synthesize_audio', synthesize)
    p = Debater(DebaterConfig(model='gpt-4o', streaming_output={
        'speech_mode': 'vanilla_streaming', 'vanilla_chunk_words': 2}), 'Test motion')
    p.status = 'opening'
    monkeypatch.setattr(p, '_speech_audio_file', lambda: str(tmp_path / 'speech.mp3'))
    return p


def test_audio_published_before_generation_finishes(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch)
    first_audio = threading.Event()
    published = []
    def emit(index, path, text, seconds):
        assert len(AudioSegment.from_file(path)) == 100
        published.append(text)
        first_audio.set()
    p.tts_chunk_callback = emit
    def stream(messages, on_text, **kwargs):
        assert messages[-1]['content'] == 'Unchanged prompt'
        assert 'time_control' not in kwargs and 'max_time' not in kwargs
        on_text('First sentence. ')
        assert first_audio.wait(5), 'TTS incorrectly waited for the full LLM response'
        on_text('Second sentence. Last tail')
    result = p.speak('Unchanged prompt', max_time=30, time_control=True, _stream_completion=stream)
    assert published == ['First sentence.', 'Second sentence.', 'Last tail']
    assert result == '\n\n'.join(published)
    trace = json.loads((tmp_path/'speech_chunks/vanilla_streaming.json').read_text())
    assert trace['status'] == 'completed'
    assert trace['chunks'][0]['ready_seconds'] < trace['generation_finished_seconds']
    assert trace['signed_duration_error_seconds'] == pytest.approx(-29.7)
    assert p.conversation[-1] == {'role': 'assistant', 'content': result}


def test_generation_failure_preserves_only_published_prefix(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch)
    emitted = threading.Event()
    p.tts_chunk_callback = lambda *args: emitted.set()
    def stream(messages, on_text, **kwargs):
        on_text('Published sentence. ')
        assert emitted.wait(5)
        on_text('Unfinished')
        raise RuntimeError('provider disconnected')
    with pytest.raises(RuntimeError, match='disconnected'):
        p.speak('Prompt', _stream_completion=stream)
    trace = json.loads((tmp_path/'speech_chunks/vanilla_streaming.json').read_text())
    assert trace['status'] == 'failed'
    assert trace['committed_text'] == 'Published sentence.'
    assert p.conversation[-1]['content'] == 'Published sentence.'
    with pytest.raises(FileExistsError):
        p.speak('Prompt', _stream_completion=stream)


def test_tts_failure_does_not_deadlock_full_queue(tmp_path, monkeypatch):
    p = player(tmp_path, monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('TTS failed')
    monkeypatch.setattr('tts_streaming.synthesize_audio', fail)
    def stream(messages, on_text, **kwargs):
        on_text('Many sentences. ' * 30)
    with pytest.raises(RuntimeError, match='TTS failed'):
        p.speak('Prompt', _stream_completion=stream)
    trace = json.loads((tmp_path/'speech_chunks/vanilla_streaming.json').read_text())
    assert trace['status'] == 'failed' and not trace['chunks']


def test_segmenter_handles_token_boundaries_and_punctuation_free_text():
    s = Segmenter(2, 100)
    assert list(s.feed('It costs 3.')) == []
    assert list(s.feed('14 dollars. Next')) == ['It costs 3.14 dollars.']
    assert list(s.feed('', final=True)) == ['Next']
    s = Segmenter(30, 50)
    parts = list(s.feed('x'*125)) + list(s.feed('', final=True))
    assert ''.join(parts) == 'x'*125 and max(map(len, parts)) <= 50


def test_provider_stream_uses_configured_client_and_no_retry(monkeypatch):
    closed = []
    def event(text=None, finish=None):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)],
                               _hidden_params={'response_cost': .01} if finish else {})
    class Stream:
        def __iter__(self):
            yield event('Hello')
            yield event(finish='stop')
        def close(self):
            closed.append(True)
    def client(**kwargs):
        assert kwargs['stream'] and kwargs['num_retries'] == 0
        return Stream()
    p = SimpleNamespace(client=client, config=SimpleNamespace(model='gpt-4o'),
                        client_cost=0, _cost_lock=threading.Lock())
    text = []
    generate(p, [], text.append, {})
    assert text == ['Hello'] and closed == [True] and p.client_cost == .01


def test_default_mode_still_uses_existing_agent_path(monkeypatch):
    from agents import Agent
    p = Debater(DebaterConfig(model='gpt-4o'), 'Test')
    monkeypatch.setattr(Agent, 'speak', lambda self, prompt, **kwargs: 'original')
    assert p.speak('Prompt') == 'original'


def test_matrix_adapter_streams_output_without_listening(tmp_path, monkeypatch):
    from optional_experiments import require_local_experiment
    require_local_experiment("experiments/tools/matrix_4_1/adapter.py")
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('vanilla_native_adapter',
        Path(__file__).resolve().parents[1]/'experiments/tools/matrix_4_1/adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    native = adapter.NativePlayer('Debater', dict(model='gpt-4o', type='default',
        streaming_output=dict(speech_mode='vanilla_streaming')), 'Motion',
        {}, dict(opening=30, rebuttal=30, closing=30))
    assert native.streaming_output and not native.streaming_listen
    audio = tmp_path/'chunk.mp3'
    AudioSegment.silent(duration=100).export(audio, format='mp3')
    def stage(**kwargs):
        native.player.tts_chunk_callback(0, str(audio), 'Spoken.', .1)
        return 'Spoken.'
    monkeypatch.setattr(native.player, 'opening_generation', stage)
    emitted = []
    result = native.generate('opening', [], 30, tmp_path, lambda *args: emitted.append(args), None)
    assert result == 'Spoken.' and len(emitted) == 1
    assert native.player.tts_chunk_callback is None
