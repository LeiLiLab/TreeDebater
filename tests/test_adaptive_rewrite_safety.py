"""Offline regression coverage for semantic-safe adaptive length edits."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.usefixtures("word_length_modes")

from streaming.config import OutputConfig
from test_full_speech import audio as audio, overlap_speaker
import tts_streaming as tts


def test_split_keeps_seven_day_limit_and_question_pause_in_same_sentence():
    sentence = 'Recordings require consent, a seven-day limit, and pausing during student questions.'
    chunks = tts.split_into_chunks(sentence + ' Funding needs review.', 120)
    assert any(sentence in chunk for chunk in chunks)
    assert ' '.join(chunks) == sentence + ' Funding needs review.'


def test_gemma_rewrite_and_json_review_use_configured_proxy_and_close_client(monkeypatch):
    factory, client = Mock(), Mock()
    factory.return_value = client
    monkeypatch.setattr(tts, 'OpenAI', factory)
    monkeypatch.setenv('DEBATE_LLM_API_BASE', 'http://localhost:4321/v1')
    monkeypatch.setenv('DEBATE_LLM_API_KEY', 'local-test')
    tts._text_request(Mock(), 'google.gemma-test', [], 400, json_mode=True)
    factory.assert_called_once_with(base_url='http://localhost:4321/v1', api_key='local-test')
    assert client.chat.completions.create.call_args.kwargs['response_format'] == {'type': 'json_object'}
    client.close.assert_called_once()


def test_length_edit_publishes_without_extra_review(audio, tmp_path, monkeypatch):
    query, encoded = audio
    prefix = 'Funding remains unresolved.'
    original = 'Litter could increase unless the promised cleanup team is properly supervised.'
    candidate = 'Litter could increase without supervised cleanup.'
    rewrite = Mock(return_value=candidate)
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    query.side_effect = lambda client, text, *a, **kw: encoded(1 if text == prefix else 10 if text == original else 4.9)
    seen = []
    tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 6,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False, first_chunk_seconds=1,
            max_refinements=1, early_max_refinements=1, speed_adjust_min=1, speed_adjust_max=1,
            allow_expansion=False), tail_supplier=lambda: original,
        on_chunk=lambda i, p, text, duration: seen.append(text))
    assert seen == [prefix, candidate]
    audits = json.loads((tmp_path / 'speech_chunks/rewrite_audit.json').read_text())
    assert any(c['accepted'] and not c['verified'] for a in audits for c in a['checks'])


def test_short_audio_is_not_expanded_or_slowed_to_fill_budget(audio, tmp_path, monkeypatch):
    query, encoded = audio
    query.side_effect = lambda *args, **kwargs: encoded(1)
    rewrite = Mock(side_effect=AssertionError('No length expansion allowed'))
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    seen = []
    tts.convert_text_to_speech_streaming('Funding remains unresolved.', str(tmp_path / 'speech.mp3'), 60,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False,
                           allow_expansion=False),
        tail_supplier=lambda: 'Costings need independent review.',
        on_chunk=lambda i, p, text, duration: seen.append((text, duration)))
    assert sum(duration for _, duration in seen) == 2
    rewrite.assert_not_called()


def test_enabled_expansion_publishes_without_extra_review(audio, tmp_path, monkeypatch):
    query, encoded = audio
    prefix = 'Funding remains unresolved.'
    original = 'Costings need independent review.'
    candidate = 'The costings remain unresolved and need a review conducted independently.'
    rewrite = Mock(return_value=candidate)
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    query.side_effect = lambda client, text, *a, **kw: encoded(1 if text != candidate else 4.9)
    seen = []
    tts.convert_text_to_speech_streaming(prefix, str(tmp_path / 'speech.mp3'), 6,
        config=OutputConfig(adaptive_delivery=True, normalize_seams=False, first_chunk_seconds=1,
            max_refinements=1, early_max_refinements=1, speed_adjust_min=1, speed_adjust_max=1,
            allow_expansion=True), tail_supplier=lambda: original,
        on_chunk=lambda i, p, text, duration: seen.append((text, duration)))
    assert rewrite.call_args.args[2] > len(original.split())
    assert [text for text, _ in seen] == [prefix, candidate]
    assert sum(duration for _, duration in seen) == pytest.approx(5.9, abs=.05)


def test_frozen_prefix_honors_two_pass_whole_review_setting(audio, tmp_path):
    player, history = overlap_speaker(tmp_path)
    player.config.single_pass_revision = False
    player.speak('Speak.', 60, time_control=True, history=history)
    assert player._get_revision_suggestion.call_count == 2
    assert player._length_adjust.call_count == 2
    assert 'frozen_prefix' not in player._length_adjust.call_args_list[0].kwargs
    assert player._length_adjust.call_args_list[1].kwargs['frozen_prefix'] == 'Who funds the trial?'
