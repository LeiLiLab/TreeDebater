"""Exact whole-script synthesis and measured duration fitting without an API."""
from pathlib import Path
import json
import pytest
from pydub import AudioSegment
from utils import batch_audio


def test_provider_limit_never_discards_tail_or_whitespace():
    for text in ['A full sentence. '*700,'x'*9000,'word '*1900]:
        parts=batch_audio.split_text(text)
        assert ''.join(parts)==text
        assert all(0<len(p)<=4096 for p in parts)


def test_full_synthesis_and_trim_keep_exact_rendered_text(tmp_path,monkeypatch):
    calls=[]
    def render(client,text,voice='echo'):
        calls.append(text)
        return AudioSegment.silent(duration=len(text)*10)
    monkeypatch.setattr(batch_audio,'render',render)
    source=tmp_path/'speech.mp3'
    text='Sentence one. Sentence two. Sentence three.'
    duration=batch_audio.synthesize(None,text,source)
    assert ''.join(calls)==text
    assert duration==pytest.approx(len(text)*.01)
    dest=tmp_path/'trimmed.mp3'
    seconds,parts=batch_audio.trim_measured(None,source,dest,280)
    assert ''.join(parts)=='Sentence one. Sentence two.'
    assert seconds<=.28
    assert len(AudioSegment.from_file(dest))<=280
    assert json.loads(batch_audio.metadata_path(dest).read_text())['text']==''.join(parts)


def test_no_sentence_fits_is_failure_not_original_transcript(tmp_path,monkeypatch):
    monkeypatch.setattr(batch_audio,'render',lambda *a,**k:AudioSegment.silent(duration=1000))
    source=tmp_path/'speech.mp3';batch_audio.synthesize(None,'Too long.',source)
    with pytest.raises(ValueError,match='No complete sentence'):
        batch_audio.trim_measured(None,source,tmp_path/'trimmed.mp3',10)


def test_local_batch_and_trim_never_construct_provider(tmp_path, monkeypatch):
    import numpy as np
    import openai
    import sys
    from types import SimpleNamespace
    calls=[]
    class Local:
        def synthesize_waveform(self, text):
            calls.append(text)
            return np.zeros(len(text)*160, dtype=np.int16), 16000
    def forbidden(*args, **kwargs):
        raise AssertionError('External TTS must not be used')
    monkeypatch.setattr(openai, 'OpenAI', forbidden)
    monkeypatch.setattr(batch_audio, 'render', forbidden)
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper',
                        SimpleNamespace(get_shared_wrapper=lambda: Local()))
    from tts import convert_text_to_speech, trim_audio_by_sentences
    source=tmp_path/'local.mp3'
    text='Sentence one. Sentence two. Sentence three.'
    spoken, _, duration=convert_text_to_speech(text,source,tts_backend='fastspeech')
    assert spoken==text and duration>0
    seconds, parts=trim_audio_by_sentences(source,tmp_path/'trim.mp3',280,tts_backend='fastspeech')
    assert ''.join(parts)=='Sentence one. Sentence two.'
    assert seconds<=.28 and len(calls)>1
