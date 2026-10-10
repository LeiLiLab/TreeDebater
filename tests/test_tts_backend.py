"""Backend routing and encoded audio regressions, without API or model inference."""
from dataclasses import replace
from io import BytesIO
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pydub import AudioSegment

from streaming.body_audio import FirstBodyAudio
from streaming.config import OutputConfig, resolve_config
from streaming.listening_prefix import PrefixPreparation, synthesize_prefix
from utils import constants
from utils.time_estimator import LengthEstimator
import tts_streaming as tts


@pytest.fixture
def local_audio(monkeypatch):
    wrapper = SimpleNamespace(
        synthesize_waveform=Mock(return_value=(np.zeros(22050, dtype=np.int16), 22050)),
        query_time=Mock(side_effect=lambda items: [120.] * len(items)))
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper', SimpleNamespace(get_shared_wrapper=lambda **kw: wrapper))
    monkeypatch.setattr(tts, 'OpenAI', Mock(side_effect=AssertionError('Unexpected OpenAI client')))
    return wrapper


def test_backend_config_roundtrip():
    full = {'streaming': {'output': {'tts_backend': 'fastspeech'}}}
    assert resolve_config(full).output.tts_backend == 'fastspeech'
    assert full['streaming']['output']['tts_backend'] == 'fastspeech'
    assert OutputConfig().tts_backend == 'openai'
    with pytest.raises(ValueError, match='tts_backend'):
        OutputConfig(tts_backend='typo')


def test_native_duration_disables_openai_fit(local_audio, monkeypatch):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'fastspeech')
    assert tts.estimate_statement_seconds('Long speech.', OutputConfig()) == pytest.approx(126.2)
    cfg = OutputConfig(tts_backend='fastspeech')
    assert tts.estimate_statement_seconds('Long speech.', cfg) == 120.
    assert tts._estimate_duration('Long speech.', config=cfg) == 120.
    local_audio.query_time.side_effect = lambda items: [100., 120.]
    assert LengthEstimator('fastspeech').query_time(['short', 'long']) == pytest.approx([100., 126.2])
    assert LengthEstimator('fastspeech', tts_backend='fastspeech').query_time(['short', 'long']) == [100., 120.]


def test_local_prefix_and_body_emit_mp3_without_openai(local_audio):
    cfg = OutputConfig(tts_backend='fastspeech')
    prefix = synthesize_prefix('Prefix.', cfg)
    assert len(AudioSegment.from_file(BytesIO(prefix['mp3_bytes']), format='mp3')) == 1000
    assert prefix['audio_seconds'] > 0
    body = FirstBodyAudio(cfg, time.perf_counter())
    try:
        body.prepare_chunk('Body.')
        future = body.match('Body.', cfg.voice, cfg.model, 'fastspeech')
        assert future.result(timeout=5)['mp3_bytes']
        assert body.match('Body.', cfg.voice, cfg.model, 'openai') is None
    finally:
        body.close()
    assert [c.args[0] for c in local_audio.synthesize_waveform.call_args_list] == ['Prefix.', 'Body.']


def test_prefix_cache_rejects_another_backend():
    import threading
    prep = PrefixPreparation.__new__(PrefixPreparation)
    prep._lock = threading.Lock()
    cfg = OutputConfig(tts_backend='fastspeech')
    prep._audio_latest = dict(text='Prefix.', voice=cfg.voice, model=cfg.model,
                              tts_backend='fastspeech', tts_out={})
    assert prep.prepared_audio('Prefix.', cfg) is not None
    assert prep.prepared_audio('Prefix.', replace(cfg, tts_backend='openai')) is None


def test_regular_chunks_use_local_audio(local_audio, monkeypatch, tmp_path):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'time')
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', 'words')
    cfg = OutputConfig(tts_backend='fastspeech', max_refinements=0, early_max_refinements=0,
                       adaptive_delivery=True, normalize_seams=False, budget_mode='audio_duration',
                       min_chunk_words=1, enable_early_cut=False)
    text, _, duration = tts.convert_text_to_speech_streaming(
        'This is the full speech.', tmp_path / 'speech.mp3', total_budget_s=3, config=cfg)
    assert text == 'This is the full speech.'
    assert duration > 0
    assert local_audio.synthesize_waveform.called
    assert (tmp_path / 'speech.mp3').is_file()


def test_fastspeech_waveform_uses_vocoder_and_shared_lock(monkeypatch):
    # Exercise tensor shapes and lazy vocoder wiring without loading model weights.
    import threading
    import torch
    import utils.fs_wrapper as fs
    import fastspeech2.utils.model as models
    wrapper = fs.FastSpeechWrapper.__new__(fs.FastSpeechWrapper)
    wrapper.configs = ({'preprocessing': {'stft': {'hop_length': 256}, 'audio': {'sampling_rate': 22050}}}, {}, {})
    wrapper.device = torch.device('cpu')
    wrapper._lock = threading.Lock()
    wrapper.vocoder = None
    wrapper.process_text = lambda texts: [('Part-0', texts[0], 0, np.array([1, 2, 3]), 3)]
    predictions = [None] * 10
    predictions[1] = torch.zeros(1, 5, 80)
    predictions[9] = torch.tensor([5])
    def model(*args, **kwargs):
        assert wrapper._lock.locked()
        assert kwargs == dict(p_control=1., e_control=1., d_control=1.)
        return predictions
    wrapper.model = model
    loader = Mock(return_value=object())
    monkeypatch.setattr(fs, 'get_vocoder', loader)
    def vocoder(mels, *args, lengths):
        assert wrapper._lock.locked()
        assert tuple(mels.shape) == (1, 80, 5)
        assert lengths.tolist() == [1280]
        return [np.zeros(1280, dtype=np.int16)]
    monkeypatch.setattr(models, 'vocoder_infer', vocoder)
    for _ in range(2):
        waveform, sr = wrapper.synthesize_waveform('Words.')
        assert waveform.shape == (1280,)
        assert sr == 22050
    loader.assert_called_once()
