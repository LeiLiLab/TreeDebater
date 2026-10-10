"""All delivery modes reuse the original TreeDebater paragraph splitter."""
from unittest.mock import Mock

import pytest

from streaming.config import OutputConfig
import tts_streaming as tts
from test_full_speech import audio as audio

pytestmark = pytest.mark.usefixtures('word_length_modes')


def paragraph(label, sentences=4):
    return ' '.join(f'{label}{i} ' + ' '.join(['evidence'] * 14) + '.'
                    for i in range(sentences))


def config(**kwargs):
    return OutputConfig(budget_mode='audio_duration', normalize_seams=False, min_chunk_words=50, max_stream_chunks=5,
        max_refinements=0, early_max_refinements=0, speed_adjust_min=1,
        speed_adjust_max=1, **kwargs)


@pytest.mark.parametrize('adaptive', [False, True])
def test_complete_speech_reuses_original_splitter_and_merge(audio, tmp_path, monkeypatch, adaptive):
    text = '\n\n'.join([paragraph('first'), 'Brief transition.',
                         paragraph('long', 16), paragraph('last')])
    cfg = config(adaptive_delivery=adaptive)
    expected = tts._merge_short_chunks(tts.split_into_chunks(text, 240, cfg), cfg.min_chunk_words)
    split = Mock(wraps=tts.split_into_chunks)
    merge = Mock(wraps=tts._merge_short_chunks)
    monkeypatch.setattr(tts, 'split_into_chunks', split)
    monkeypatch.setattr(tts, '_merge_short_chunks', merge)
    delivered = []
    tts.convert_text_to_speech_streaming(text, str(tmp_path / 'speech.mp3'), 240,
        config=cfg, on_chunk=lambda i, p, text, duration: delivered.append(text))
    split.assert_called_once_with(text, 240, cfg)
    merge.assert_called_once()
    assert delivered == expected
    assert ' '.join(delivered).split() == text.split()


def test_deferred_body_reuses_splitter_with_remaining_budget_and_keeps_prefix(audio, tmp_path, monkeypatch):
    query, encoded = audio
    prefix = 'The opening remains fixed.'
    text = '\n\n'.join(paragraph(str(i)) for i in range(4))
    cfg = config(adaptive_delivery=True)
    query.side_effect = lambda client, content, *args, **kw: encoded(10 if content == prefix else .1)
    expected = tts._merge_short_chunks(tts.split_into_chunks(text, 110, cfg), cfg.min_chunk_words)
    split, merge = Mock(wraps=tts.split_into_chunks), Mock(wraps=tts._merge_short_chunks)
    monkeypatch.setattr(tts, 'split_into_chunks', split)
    monkeypatch.setattr(tts, '_merge_short_chunks', merge)
    delivered = []
    tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 120,
        config=cfg, tail_supplier=lambda: text,
        on_chunk=lambda i, p, text, duration: delivered.append(text))
    split.assert_called_once_with(text, 110, cfg)
    merge.assert_called_once()
    assert delivered == [prefix] + expected
    assert delivered[1:] == text.split('\n\n')


@pytest.mark.parametrize('text', [
    paragraph('unbroken', 32),
    '\n\n'.join(paragraph(str(i), 1) for i in range(20)),
    'A brief conclusion.',
    ' '.join(['indivisible'] * 500) + '.',
])
def test_original_splitter_preserves_content_and_whole_sentences(text):
    cfg = config()
    chunks = tts._merge_short_chunks(tts.split_into_chunks(text, 120, cfg), cfg.min_chunk_words)
    assert ' '.join(chunks).split() == text.split()
    assert all(chunk.endswith('.') for chunk in chunks)
    assert tts._split_sentences(' '.join(chunks)) == tts._split_sentences(text)
