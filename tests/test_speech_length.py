"""Configured estimators govern draft, listening and adaptive-TTS decisions.

FastSpeech is replaced at its model boundary; these tests make no API calls or
model downloads, and deliberately disagree with word counts to detect bypasses.
"""
import sys
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import constants, speech_length
from utils.time_estimator import LengthEstimator
from streaming.config import OutputConfig
from streaming.listening_prefix import _valid_candidate, material
from ouragents import TreeDebater
import tts_streaming as tts
from test_listening_prefix import FRAMEWORK, PREFIX, prepared_player
from test_full_speech import audio as audio

pytestmark = pytest.mark.usefixtures('word_length_modes')


def fastspeech(monkeypatch, predict):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'fastspeech')
    wrapper = Mock()
    wrapper.query_time.side_effect = lambda texts: [predict(text) for text in texts]
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper',
        SimpleNamespace(get_shared_wrapper=Mock(return_value=wrapper)))
    return wrapper


@pytest.mark.parametrize('mode,units,equivalent', [
    ('words', None, 2), ('syllables', 7, 4), ('phonemes', 90, 20)])
def test_draft_measurement_uses_selected_unit(monkeypatch, mode, units, equivalent):
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', mode)
    if units is not None:
        monkeypatch.setattr(LengthEstimator, 'count_' + mode, lambda text: units)
    assert speech_length.draft_word_count('Two words.') == equivalent


@pytest.mark.parametrize('mode,unit_target', [('words', 100), ('syllables', 175), ('phonemes', 450)])
def test_initial_draft_prompt_uses_configured_length_unit(monkeypatch, mode, unit_target):
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', mode)
    player = SimpleNamespace(status='rebuttal', motion='A motion', act='oppose', counter_act='support',
        use_debate_flow_tree=False, _add_additional_info=lambda prompt, *a, **kw: prompt)
    prompt, _ = TreeDebater._prepare_stage_prompt(player, [], 46)
    assert f'approximately {unit_target} {mode}' in prompt


def test_prefix_length_guard_uses_draft_mode_not_whitespace(monkeypatch, tmp_path):
    player, history, node = prepared_player(tmp_path)
    data = material(player, player.status, history)
    candidate = dict(text=PREFIX, target_ids=[node.node_id], framework=FRAMEWORK)
    assert _valid_candidate(candidate, player.streaming_output_config)
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', 'phonemes')
    monkeypatch.setattr(LengthEstimator, 'count_phonemes', lambda text: 450)
    assert not _valid_candidate(candidate, player.streaming_output_config)


def test_fastspeech_is_not_replaced_by_observed_word_rate(monkeypatch):
    wrapper = fastspeech(monkeypatch, lambda text: 41.)
    assert speech_length.estimate_seconds('Two words.', measured_seconds_per_word=.1) == 41
    wrapper.query_time.assert_called_once_with(['Two words.'])
    assert tts._estimate_duration('Two words.', measured_seconds_per_word=.1) == 41


def test_only_time_backend_uses_observed_rate(monkeypatch):
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper', None)
    assert speech_length.estimate_seconds('One two.', measured_seconds_per_word=.3) == pytest.approx(.6)


def test_estimator_failure_does_not_fall_back_to_words(monkeypatch):
    wrapper = fastspeech(monkeypatch, lambda text: 1)
    wrapper.query_time.side_effect = RuntimeError('FastSpeech unavailable')
    with pytest.raises(RuntimeError, match='FastSpeech unavailable'):
        speech_length.estimate_seconds('Two words.', measured_seconds_per_word=.1)


def test_statement_mode_rejects_count_units(monkeypatch):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'words')
    with pytest.raises(ValueError, match='seconds estimator'):
        speech_length.estimate_seconds('Two words.')


@pytest.mark.parametrize('actual_seconds,accepted', [(2., True), (6., False)])
def test_adaptive_worker_estimates_with_fastspeech_but_accepts_actual_audio(monkeypatch, actual_seconds, accepted):
    fastspeech(monkeypatch, lambda text: 2.)
    monkeypatch.setattr(tts, '_tts_with_retry', lambda *a, **kw: {'audio_seconds': actual_seconds})
    ctx = tts._ChunkRefineContext(client=None, original_text='One point.', target_s=2,
        tol_s=.1, tol_upper_s=.1, prev_texts=[], next_chunk_text='', voice='echo',
        max_ref=0, kickoff_iter=0, kickoff_kind='',
        config=OutputConfig(adaptive_delivery=True, max_parallel_tts=1))
    ctx.seconds_per_word = .01
    try:
        tts._refine_worker(ctx, 'normal')
        assert ctx.candidates[0].fs_estimated_s == 2
        assert (ctx.chosen_cand is not None) is accepted
    finally:
        ctx.executor.shutdown(wait=True)


@pytest.mark.parametrize('words,estimate,expect_revision', [(10, 60., False), (130, 90., True)])
def test_listening_no_change_gate_uses_seconds_not_word_band(audio, monkeypatch, tmp_path,
                                                           words, estimate, expect_revision):
    from test_listening_prefix import prefix_helper
    player, history, node = prepared_player(tmp_path)
    tail = ' '.join(['reason'] * words) + '.'
    fastspeech(monkeypatch, lambda text: estimate if text == tail else 1.)
    player.config.single_pass_revision = True
    player.helper_client = prefix_helper(node, draft=tail)
    player._length_adjust.return_value = tail
    published = threading.Event()
    player.tts_chunk_callback = lambda *a: published.set()
    def feedback(**kwargs):
        assert published.wait(3)
        return 'Revision Guidance:\nNo changes', [], '', tail
    player._get_revision_suggestion.side_effect = feedback
    player.rebuttal_generation(history, 60, time_control=True)
    assert bool(player._length_adjust.call_count) is expect_revision


def test_listening_revision_uses_fastspeech_even_when_word_count_disagrees(monkeypatch, tmp_path):
    player, _, _ = prepared_player(tmp_path)
    first, second = (' '.join(['reason'] * n) + '.' for n in (130, 10))
    fastspeech(monkeypatch, lambda text: 80. if text == first else 59.)
    player.helper_client = Mock(side_effect=[[first], [second]])
    result = MethodType(TreeDebater._length_adjust, player)(
        'Draft.', 'Corrections.', [], '', 60, max_retry=2, frozen_prefix=PREFIX)
    assert result == second
    assert player.debate_thoughts[-1]['final_cost'] == 59
    assert not player.debate_thoughts[-1]['soft_duration_target_missed']


def test_bounded_fallback_selects_closest_estimated_seconds(monkeypatch, tmp_path):
    player, _, _ = prepared_player(tmp_path)
    first, second = (' '.join(['reason'] * n) + '.' for n in (160, 135))
    fastspeech(monkeypatch, lambda text: 70. if text == first else 80.)
    player.helper_client = Mock(side_effect=[[first], [second]])
    result = MethodType(TreeDebater._length_adjust, player)(
        'Draft.', 'Corrections.', [], '', 60, max_retry=2, frozen_prefix=PREFIX)
    assert result == first
    assert player.debate_thoughts[-1]['soft_duration_target_missed']


def test_single_body_revision_defers_out_of_range_duration_to_tts(monkeypatch, tmp_path):
    player, _, _ = prepared_player(tmp_path)
    body = 'We oppose the trial because funding remains uncertain.'
    fastspeech(monkeypatch, lambda text: 30.)
    player.helper_client = Mock(return_value=[body])
    result = MethodType(TreeDebater._length_adjust, player)(
        'Draft.', 'Corrections.', [], '', 60, max_retry=1,
        frozen_prefix=PREFIX, defer_duration_fit=True)
    assert result == body
    player.helper_client.assert_called_once()
    assert player.debate_thoughts[-1]['final_cost'] == 30
    # Native single-pass mode accepts the revision; audio fitting follows in TTS.
    assert not player.debate_thoughts[-1]['soft_duration_target_missed']
