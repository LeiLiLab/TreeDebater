"""Real offline FFmpeg checks for pitch, duration, retained ending and publication."""
from io import BytesIO
import subprocess
from unittest.mock import Mock

import numpy as np
from pydub import AudioSegment
from pydub.generators import Sine
import pytest

from streaming.audio_tempo import change_audio_tempo, fit_audio_tempo
from streaming.config import OutputConfig, resolve_config
import tts_streaming as tts


def tone(milliseconds, frequency=440):
    return Sine(frequency, sample_rate=24000).to_audio_segment(duration=milliseconds).apply_gain(-12)


def peak_hz(audio):
    values = np.asarray(audio.get_array_of_samples(), dtype=float)
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values))))
    return float(np.fft.rfftfreq(len(values), 1 / audio.frame_rate)[np.argmax(spectrum)])


@pytest.mark.parametrize('duration,target', [(8800, 8), (7200, 8), (13200, 12)])
def test_tempo_preserves_pitch_and_stereo_while_fitting_duration(duration, target):
    stereo = AudioSegment.from_mono_audiosegments(tone(duration), tone(duration, 660))
    adjusted, result = fit_audio_tempo(stereo, target)
    assert result.status == 'applied'
    assert abs(len(adjusted) / 1000 - target) < .08
    assert adjusted.frame_rate == stereo.frame_rate and adjusted.channels == 2
    left, right = adjusted.split_to_mono()
    assert abs(peak_hz(left[500:1500]) - 440) < 2
    assert abs(peak_hz(right[500:1500]) - 660) < 2


def test_tempo_keeps_end_marker_and_clamps_instead_of_cutting_or_padding():
    source = tone(9400) + tone(600, 880)
    adjusted, result = fit_audio_tempo(source, 8)
    assert result.clamped and result.speed == 1.15
    assert 8.5 < result.output_seconds < 8.8
    assert abs(peak_hz(adjusted[-300:]) - 880) < 4
    short, result = fit_audio_tempo(tone(6000), 12)
    assert result.clamped and result.speed == .85 and 6.9 < len(short) / 1000 < 7.2


def test_near_target_and_unity_bounds_do_not_run_ffmpeg(monkeypatch):
    run = Mock(side_effect=AssertionError('Unexpected subprocess'))
    monkeypatch.setattr(subprocess, 'run', run)
    source = tone(8050)
    assert fit_audio_tempo(source, 8)[0] is source
    assert fit_audio_tempo(source, 6, min_speed=1, max_speed=1)[1].status == 'speed_limit'
    assert change_audio_tempo(source, 1) is source
    run.assert_not_called()


def test_ffmpeg_is_bounded_and_uses_pcm_pipes(monkeypatch):
    run = Mock(side_effect=subprocess.TimeoutExpired('ffmpeg', 3))
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(subprocess.TimeoutExpired):
        change_audio_tempo(tone(1000), 1.1)
    args, kwargs = run.call_args
    assert '-nostdin' in args[0] and 'atempo=1.10000000' in args[0]
    assert kwargs['timeout'] == 3 and isinstance(kwargs['input'], bytes)
    assert kwargs['check'] and kwargs['capture_output']


@pytest.mark.parametrize('settings', [
    {'first_chunk_local_tempo': 'true'}, {'local_tempo_min': .4},
    {'local_tempo_min': 1.1}, {'local_tempo_max': .9}, {'local_tempo_max': 2.1},
    {'local_tempo_deadband_seconds': -1}, {'local_tempo_max': float('nan')},
])
def test_invalid_tempo_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        OutputConfig(**settings)


def test_tempo_settings_round_trip_and_default_is_opt_in():
    assert not OutputConfig().first_chunk_local_tempo
    raw = {'streaming': {'output': {'first_chunk_local_tempo': True, 'first_chunk_seconds': 12}}}
    resolved = resolve_config(raw)
    assert resolved.output.first_chunk_local_tempo
    assert raw['streaming']['output']['local_tempo_min'] == .85
    assert raw['streaming']['output']['first_chunk_seconds'] == 12


def encoded(audio):
    buffer = BytesIO()
    audio.export(buffer, format='mp3')
    return dict(mp3_bytes=buffer.getvalue(), audio_seconds=len(audio) / 1000,
                tts_api_s=.001, mp3_parse_s=.001)


def test_pipeline_publishes_adjusted_audio_once_and_accounts_for_actual_duration(tmp_path, monkeypatch):
    prefix, tail = 'Funding remains unresolved.', 'We need independent costings before the trial.'
    head_audio, tail_audio = encoded(tone(8800)), encoded(tone(1000))
    query = Mock(side_effect=lambda client, text, **kw: head_audio if text == prefix else tail_audio)
    monkeypatch.setattr(tts, '_query_time_profiled', query)
    monkeypatch.setattr(tts, '_revise_to_n_words', Mock(side_effect=AssertionError('Text changed')))
    events = []
    def ready(index, path, text, seconds):
        assert abs(len(AudioSegment.from_file(path)) / 1000 - seconds) < .002
        events.append((text, seconds))
    profiles, summary, combined, texts = tts.run_pipeline(Mock(), [prefix], 20,
        config=OutputConfig(first_chunk_local_tempo=True, first_chunk_seconds=8, normalize_seams=False,
            budget_mode='audio_duration', adaptive_delivery=True, allow_expansion=False,
            max_refinements=0, early_max_refinements=0),
        out_dir=tmp_path, tail_supplier=lambda: tail, on_chunk=ready)
    assert query.call_count == 2  # One provider request per chunk, including adjusted prefix.
    assert texts == [prefix, tail] == [e[0] for e in events]
    assert abs(events[0][1] - 8) < .08
    assert profiles[0].local_tempo_status == 'applied' and profiles[0].target_s == 8
    assert profiles[0].local_tempo_speed == pytest.approx(1.1)
    assert profiles[1].local_tempo_status == 'disabled'
    assert profiles[1].target_s == pytest.approx(20 - events[0][1])
    assert summary.audio_seconds_total == pytest.approx(sum(e[1] for e in events))
    assert summary.budget_remaining_s == pytest.approx(20 - summary.audio_seconds_total)
    assert abs(len(AudioSegment.from_file(BytesIO(combined), format='mp3')) / 1000
               - summary.audio_seconds_total) < .003


@pytest.mark.parametrize('error', [FileNotFoundError('ffmpeg'), subprocess.TimeoutExpired('ffmpeg', 3)])
def test_local_failure_publishes_original_without_another_provider_request(tmp_path, monkeypatch, error):
    original = encoded(tone(8800))
    query = Mock(return_value=original)
    monkeypatch.setattr(tts, '_query_time_profiled', query)
    monkeypatch.setattr(tts, 'fit_audio_tempo', Mock(side_effect=error))
    profiles, _, _, texts = tts.run_pipeline(Mock(), ['A complete sentence.'], 60,
        config=OutputConfig(first_chunk_local_tempo=True, first_chunk_seconds=8, normalize_seams=False,
                            adaptive_delivery=True), out_dir=tmp_path)
    query.assert_called_once()
    assert (tmp_path / 'chunk_000.mp3').read_bytes() == original['mp3_bytes']
    assert profiles[0].target_s == 8  # Explicit first target, not proportional share of 60s.
    assert profiles[0].local_tempo_status == 'failed_original_used'
    assert profiles[0].local_tempo_error == type(error).__name__
    assert texts == ['A complete sentence.']
