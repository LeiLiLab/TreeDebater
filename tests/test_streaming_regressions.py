"""Offline streaming regressions; replace API/model imports, exercise real modules."""
import importlib.util
import logging
import sys
import threading
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydub import AudioSegment

SRC = Path(__file__).resolve().parents[1] / 'src'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC))
    tool = types.ModuleType('utils.tool')
    tool.logger = logging.getLogger('streaming-tests')
    tool.remove_citation = lambda text, **kw: (text, 'references')
    tool.remove_subtitles = lambda text: text
    constants = types.ModuleType('utils.constants')
    constants.CLOSING_TIME = constants.OPENING_TIME = constants.REBUTTAL_TIME = 60
    estimator = types.ModuleType('utils.time_estimator')
    estimator.LengthEstimator = Mock()
    fs = types.ModuleType('utils.fs_wrapper')
    fs.FastSpeechWrapper = Mock()
    mp3 = types.ModuleType("mutagen.mp3")
    mp3.MP3 = Mock()
    for module in (tool, constants, estimator, fs, mp3):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, SRC / file)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    env = load('streaming.env', 'streaming/env.py')
    bridges = load('streaming.bridges', 'streaming/bridges.py')
    overlap = load('streaming.overlap', 'streaming/overlap.py')
    tts = load('tts_streaming', 'tts_streaming.py')
    return types.SimpleNamespace(env=env, bridges=bridges, overlap=overlap, tts=tts)


def listener(modules, tmp_path, cursor=None):
    deb = types.SimpleNamespace(use_debate_flow_tree=True, _analyze_statement=Mock())
    cfg = modules.env.StreamingInputConfig(
        tmp_path, 'motion', 'opening', 'for', min_audio_seconds=15,
        min_text_words=1, playback_cursor=cursor, poll_interval=0.001,
    )
    return modules.env.StreamingInputEnv(deb, cfg)


@pytest.mark.parametrize('side', ['for', 'against'])
def test_timing_rewrite_receives_stance(modules, side):
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        types.SimpleNamespace(message=types.SimpleNamespace(content='Rewritten paragraph.'))
    ]
    result = modules.tts._revise_to_n_words(
        client, 'Original paragraph.', 40, [], motion='Writing still matters', side=side,
    )
    assert result == 'Rewritten paragraph.'
    prompt = client.chat.completions.create.call_args.kwargs['messages'][0]['content']
    assert 'Debate motion: Writing still matters.' in prompt
    assert f'Assigned side: {side}.' in prompt


@pytest.mark.parametrize('duration', [5000, 20000])
def test_cursor_shutdown_drains_tail(modules, monkeypatch, tmp_path, duration):
    (tmp_path / 'continuous_audio.mp3').touch()
    env = listener(modules, tmp_path, [duration / 1000])
    monkeypatch.setattr(modules.env.AudioSegment, 'from_file', lambda *a: AudioSegment.silent(duration))
    transcribed = []
    monkeypatch.setattr(modules.env, 'transcribe_audio_segment', lambda audio, **kw: transcribed.append(len(audio)) or 'words')
    # Stop after a normal polling pass; the subsequent pass must drain the tail.
    monkeypatch.setattr(env._stop, 'wait', lambda _: env.stop())
    env.run()
    assert sum(transcribed) == duration
    assert env.succeeded


def test_cursor_never_falls_back_to_unplayed_chunks(modules, monkeypatch, tmp_path):
    (tmp_path / 'for_chunk001.mp3').touch()
    env = listener(modules, tmp_path, [0.0])
    read = Mock(return_value=AudioSegment.silent(30000))
    monkeypatch.setattr(modules.env.AudioSegment, 'from_file', read)
    transcribe = Mock(return_value='words')
    monkeypatch.setattr(modules.env, 'transcribe_audio_segment', transcribe)
    monkeypatch.setattr(env._stop, 'wait', lambda _: env.stop())
    env.run()
    transcribe.assert_not_called()
    assert not env.succeeded


def test_audio_decode_does_not_inherit_terminal_stdin(modules, monkeypatch, tmp_path):
    import shutil
    import subprocess
    import pydub.utils

    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and ffprobe required for real decoding')
    path = tmp_path / 'continuous_audio.mp3'
    AudioSegment.silent(duration=1000).export(str(path), format='mp3').close()
    popen = subprocess.Popen
    stdin_values = []

    def guarded_popen(*args, **kwargs):
        stdin_values.append(kwargs.get('stdin'))
        assert kwargs.get('stdin') is not None, 'Decoder inherited terminal stdin'
        return popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', guarded_popen)
    monkeypatch.setattr(pydub.utils, 'Popen', guarded_popen)
    audio = listener(modules, tmp_path)._read_audio_up_to_cursor(path, 0.5)
    assert audio is not None and len(audio) == 500
    assert stdin_values


def test_chunk_shutdown_scans_and_drains_all_files(modules, monkeypatch, tmp_path):
    for i in range(3):
        (tmp_path / f'for_chunk{i:03}.mp3').touch()
    env = listener(modules, tmp_path)
    monkeypatch.setattr(modules.env.AudioSegment, 'from_file', lambda *a: AudioSegment.silent(2000))
    transcribed = []
    monkeypatch.setattr(modules.env, 'transcribe_audio_segment', lambda audio, **kw: transcribed.append(len(audio)) or 'words')
    env.stop()  # Includes chunks not yet seen when the producer stops.
    env.run()
    assert transcribed == [6000]
    assert env.succeeded


@pytest.mark.parametrize('failure', ['asr', 'tree', 'empty', 'missing', 'cap'])
def test_incomplete_ingestion_does_not_report_success(modules, monkeypatch, tmp_path, failure):
    if failure != 'missing':
        (tmp_path / 'continuous_audio.mp3').touch()
    env = listener(modules, tmp_path, [20.0])
    monkeypatch.setattr(modules.env.AudioSegment, 'from_file', lambda *a: AudioSegment.silent(20000))
    transcribe = Mock(return_value='' if failure == 'empty' else 'words')
    if failure == 'asr':
        transcribe.side_effect = RuntimeError('ASR unavailable')
    if failure == 'tree':
        env.debater._analyze_statement.side_effect = RuntimeError('tree unavailable')
    if failure == 'cap':
        env.config.max_total_audio_seconds = 10
    monkeypatch.setattr(modules.env, 'transcribe_audio_segment', transcribe)
    env.stop()
    env.run()
    assert not env.succeeded


def test_bridge_drains_last_chunk_after_producer_finishes(modules, tmp_path, monkeypatch):
    source, watch = tmp_path / 'source', tmp_path / 'watch'
    source.mkdir()
    (source / 'chunk_000.mp3').write_bytes(b'a' * 1024)
    stop, drained = threading.Event(), threading.Event()
    count = [0]

    def next_poll(_):
        if count[0] == 1:
            (source / 'chunk_001.mp3').write_bytes(b'b' * 1024)
            stop.set()
    monkeypatch.setattr(stop, 'wait', next_poll)
    modules.bridges.run_streaming_tts_chunk_copy_bridge(
        source, watch, 'for', stop, count, drained_event=drained,
    )
    assert count == [2]
    assert drained.is_set()
    assert (watch / 'for_chunk002.mp3').read_bytes() == b'b' * 1024
    assert not list(watch.glob('*.tmp'))


def test_bridge_does_not_acknowledge_missing_chunk(modules, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'chunk_001.mp3').write_bytes(b'a' * 1024)
    stop, drained = threading.Event(), threading.Event()
    stop.set()
    modules.bridges.run_streaming_tts_chunk_copy_bridge(
        source, tmp_path / 'watch', 'for', stop, [0], drained_event=drained,
    )
    assert not drained.is_set()


def test_tts_returns_revised_spoken_text(modules, monkeypatch, tmp_path):
    profile = types.SimpleNamespace(audio_seconds_total=1, overrun_total_s=0,
                                   budget_remaining_s=0, round_total_s=1)
    monkeypatch.setattr(modules.tts, 'OpenAI', Mock())
    monkeypatch.setattr(modules.tts, 'run_pipeline', lambda *a, **kw: ([], profile, b'audio', ['Revised first.', 'Revised second.']))
    monkeypatch.setattr(modules.tts, 'MP3', lambda _: types.SimpleNamespace(info=types.SimpleNamespace(length=1)))
    text, reference, _ = modules.tts.convert_text_to_speech_streaming('Original.', tmp_path / 'speech.mp3', 10)
    assert text == 'Revised first.\n\nRevised second.'
    assert reference == 'references'


@pytest.mark.parametrize('success', [True, False])
def test_turn_only_marks_acknowledged_ingestion(modules, monkeypatch, tmp_path, success):
    env = object.__new__(modules.overlap.OverlappingStreamingDebateEnv)
    speaker = types.SimpleNamespace(config=types.SimpleNamespace(streaming_tts=False, type='test'), side='for')
    deb = types.SimpleNamespace(type='treedebater', config=types.SimpleNamespace(streaming_listen=True),
                                start_streaming_listen=Mock(), stop_streaming_listen=Mock(return_value=success))
    env._env = types.SimpleNamespace(debaters={'for': speaker, 'against': deb}, time_control=False, debate_process=[])
    env._watch_root = tmp_path
    for name, value in dict(listener_join_timeout=2, min_audio_seconds=15, min_text_words=1,
                            poll_interval=0.001, audio_format='mp3', max_audio_wait=None,
                            max_text_wait=None, max_total_audio=None).items():
        setattr(env, '_' + name, value)
    monkeypatch.setattr(modules.overlap, 'tts_outputs_dir_from_log', lambda: tmp_path)
    monkeypatch.setattr(modules.overlap.time, 'sleep', lambda _: None)
    env._playback_main_loop = lambda *a, **kw: a[3].wait(2)
    env._play_speech_turn('opening', 'for', 60, lambda: 'Speech')
    assert env._env.debate_process[0].get('tree_via_streaming', False) is success


def test_bridge_copy_failure_cannot_acknowledge_partial_delivery(modules, monkeypatch, tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for i in range(2):
        (source / f'chunk_{i:03}.mp3').write_bytes(b'a' * 1024)
    original_copy = modules.bridges.shutil.copy2

    def copy(path, destination):
        if path.name == 'chunk_001.mp3':
            raise OSError('copy failed')
        return original_copy(path, destination)

    monkeypatch.setattr(modules.bridges.shutil, 'copy2', copy)
    stop, drained = threading.Event(), threading.Event()
    stop.set()
    count = [0]
    modules.bridges.run_streaming_tts_chunk_copy_bridge(
        source, tmp_path / 'watch', 'for', stop, count, drained_event=drained,
    )
    assert count == [1]
    assert not drained.is_set()


def test_pipeline_uses_configured_tts_and_parallelism(modules, monkeypatch):
    from streaming.config import OutputConfig
    cfg = OutputConfig(model='custom-tts', voice='alloy', min_chunk_words=1,
                       max_parallel_tts=2, max_refinements=0)
    monkeypatch.setattr(modules.tts.LengthEstimator, 'count_words', lambda s: len(s.split()))
    monkeypatch.setattr(modules.tts, '_estimate_duration', lambda _: 1.0)
    requests = []

    def query(client, text, **kwargs):
        requests.append(kwargs)
        return {'audio_seconds': 1., 'tts_api_s': 0., 'mp3_parse_s': 0., 'mp3_bytes': b'audio'}

    monkeypatch.setattr(modules.tts, '_query_time_profiled', query)
    monkeypatch.setattr(modules.tts.AudioSegment, 'from_file', lambda *a, **kw: AudioSegment.silent(1000))
    monkeypatch.setattr(modules.tts.AudioSegment, 'export', lambda self, output, **kw: output.write(b'audio'))
    executor = modules.tts.concurrent.futures.ThreadPoolExecutor
    pools = []

    def make_pool(**kwargs):
        pools.append(kwargs['max_workers'])
        return executor(**kwargs)

    monkeypatch.setattr(modules.tts.concurrent.futures, 'ThreadPoolExecutor', make_pool)
    profiles, _, _, texts = modules.tts.run_pipeline(Mock(), ['First paragraph.', 'Second paragraph.'], 2, config=cfg)
    assert len(profiles) == 2
    assert texts == ['First paragraph.', 'Second paragraph.']
    assert pools == [2]
    assert all(r['model'] == 'custom-tts' and r['voice'] == 'alloy' for r in requests)


def test_refinement_model_is_forwarded(modules):
    client = Mock()
    client.chat.completions.create.return_value.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content='revised'))]
    assert modules.tts._revise_to_n_words(client, 'original', 20, [], model='custom-refiner') == 'revised'
    assert client.chat.completions.create.call_args.kwargs['model'] == 'custom-refiner'


@pytest.mark.parametrize('entrypoint', ['env', 'overlap'])
def test_entrypoint_cli_keeps_yaml_unless_overridden(modules, monkeypatch, entrypoint):
    from streaming.config import resolve_config
    monkeypatch.setattr(sys, 'argv', ['test', '--config', 'example.yml', '--min-text-words', '17'])
    args = getattr(modules, entrypoint).parse_args()
    full = {'streaming': {'input': {'min_audio_seconds': 8, 'min_text_words': 99}}}
    resolve_config(full, args, overlap=entrypoint == 'overlap')
    assert args.min_audio_seconds == 8
    assert args.min_text_words == 17


def test_streaming_scheduling_uses_configured_budgets(modules):
    from streaming.config import SpeechBudgets
    players = {side: types.SimpleNamespace(type='default', config=types.SimpleNamespace(streaming_tts=True),
               opening_generation=Mock(return_value='opening'), rebuttal_generation=Mock(return_value='rebuttal'),
               closing_generation=Mock(return_value='closing')) for side in ['for', 'against']}
    wrapper = object.__new__(modules.env.StreamingDebateEnv)
    wrapper._env = types.SimpleNamespace(config=types.SimpleNamespace(speech_budgets=SpeechBudgets(11, 22, 33)),
                                         debaters=players, reverse=False, time_control=True,
                                         debug=False, debate_process=[{}])
    turn_budgets = []
    wrapper._play_speech_turn = lambda stage, side, budget, generate: (turn_budgets.append(budget), generate())
    wrapper.play()
    assert turn_budgets == [11, 11, 22, 22, 33, 33]
    for player in players.values():
        assert player.opening_generation.call_args.kwargs['max_time'] == 11
        assert player.rebuttal_generation.call_args.kwargs['max_time'] == 22
        assert player.closing_generation.call_args.kwargs['max_time'] == 33


def test_base_env_passes_output_config_and_budgets(modules, monkeypatch):
    agents = types.ModuleType('agents')

    class Player:
        def __init__(self, config, **kwargs):
            self.config = config
            self.type = 'default'
            self.opening_generation = Mock(return_value='opening')
            self.rebuttal_generation = Mock(return_value='rebuttal')
            self.closing_generation = Mock(return_value='closing')

    for name in ['Audience', 'AudienceConfig', 'BaselineDebater', 'Debater', 'DebaterConfig',
                 'HumanDebater', 'Judge', 'JudgeConfig']:
        setattr(agents, name, Player)
    ouragents = types.ModuleType('ouragents')
    ouragents.TreeDebater = Player
    timing = types.ModuleType('utils.timing_log')
    timing.log_timing = Mock()
    for module in (agents, ouragents, timing):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location('test_base_env', SRC / 'env.py')
    base = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'test_base_env', base)
    spec.loader.exec_module(base)
    cfg = base.EnvConfig(
        motion='motion', debater_config=[types.SimpleNamespace(side=s, type='default', streaming_tts=True)
                                        for s in ['for', 'against']],
        judge_config=None, audience_config=None, judge_num=1, audience_num=0,
        streaming={'output': {'model': 'custom-tts', 'max_parallel_tts': 2}},
        speech_budgets={'opening': 12, 'rebuttal': 23, 'closing': 34},
    )
    env = base.Env(cfg, False)
    env.play()
    for player in env.debaters.values():
        assert player.streaming_output_config.model == 'custom-tts'
        assert player.streaming_output_config.max_parallel_tts == 2
        assert player.opening_generation.call_args.kwargs['max_time'] == 12
        assert player.rebuttal_generation.call_args.kwargs['max_time'] == 23
        assert player.closing_generation.call_args.kwargs['max_time'] == 34


@pytest.mark.parametrize('mode,remaining', [('audio_duration', 9), ('experiment_elapsed', 6)])
def test_app_budget_mode_and_chunk_notification(modules, monkeypatch, tmp_path, mode, remaining):
    from streaming.config import OutputConfig
    monkeypatch.setattr(modules.tts.LengthEstimator, 'count_words', lambda text: len(text.split()))
    monkeypatch.setattr(modules.tts, '_query_time_profiled', lambda *a, **kw: {
        'audio_seconds': 1., 'tts_api_s': 3., 'mp3_parse_s': 0., 'mp3_bytes': b'audio'})
    monkeypatch.setattr(modules.tts.AudioSegment, 'from_file', lambda *a, **kw: AudioSegment.silent(1000))
    def export(audio, target, **kwargs):
        if hasattr(target, 'write'):
            target.write(b'audio')
        else:
            Path(target).write_bytes(b'audio')
    monkeypatch.setattr(modules.tts.AudioSegment, 'export', export)
    notifications = []
    def ready(index, path, text, duration):
        assert path.read_bytes() == b'audio'
        notifications.append((index, text, duration))
    _, profile, _, _ = modules.tts.run_pipeline(Mock(), ['A spoken argument.'], 10,
        config=OutputConfig(budget_mode=mode, min_chunk_words=1), out_dir=tmp_path, on_chunk=ready)
    assert profile.budget_remaining_s == remaining
    assert notifications == [(0, 'A spoken argument.', 1.)]


def test_deepseek_refinement_uses_separate_client_from_audio(modules, monkeypatch):
    audio_client = Mock()
    rewrite_client = Mock()
    rewrite_client.chat.completions.create.return_value.choices = [
        types.SimpleNamespace(message=types.SimpleNamespace(content='Length edit.'))
    ]
    factory = Mock(return_value=rewrite_client)
    monkeypatch.setattr(modules.tts, 'OpenAI', factory)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-placeholder')
    result = modules.tts._revise_to_n_words(
        audio_client, 'Original.', 30, [], model='deepseek/deepseek-v4-flash',
        motion='Writing matters', side='against',
    )
    assert result == 'Length edit.'
    factory.assert_called_once_with(api_key='test-placeholder', base_url='https://api.deepseek.com')
    audio_client.chat.completions.create.assert_not_called()
    args = rewrite_client.chat.completions.create.call_args.kwargs
    assert args['model'] == 'deepseek-v4-flash'
    assert args['extra_body']['thinking']['type'] == 'disabled'


def test_adaptive_split_preserves_text_and_short_first_chunk(modules):
    text = 'One two three four. Five six seven eight. ' + ' '.join(['long'] * 80) + '.'
    chunks = modules.tts.split_for_adaptive_delivery([text], 2.3, 4.6)
    assert ' '.join(chunks).split() == text.split()
    assert len(chunks[0].split()) <= 5
    assert all(len(c.split()) <= 10 for c in chunks[1:])
    assert len(chunks) > 2


def adaptive_context(modules):
    from streaming.config import OutputConfig
    return modules.tts._ChunkRefineContext(
        Mock(), 'some words', 10, 1, 1, [], '', 'echo', 3, 0, 'last',
        config=OutputConfig(adaptive_delivery=True),
    )


def test_adaptive_adoption_checks_actual_audio_and_changed_target(modules):
    ctx = adaptive_context(modules)
    candidate = types.SimpleNamespace(fs_estimated_s=10)
    assert not ctx.try_adopt(candidate, {'audio_seconds': 4})
    assert ctx.try_adopt(candidate, {'audio_seconds': 10})
    ctx.update_target(25, 2, 2)
    assert ctx.chosen_cand is None
    assert not ctx.done_event.is_set()
    assert not ctx.stop_event.is_set()
    # A late result computed for the old target must not be adopted either.
    assert not ctx.try_adopt(candidate, {'audio_seconds': 10})
    assert ctx.try_adopt(candidate, {'audio_seconds': 25})
    ctx.executor.shutdown(wait=True)


@pytest.mark.parametrize("recorded_closing", [False, True])
def test_adaptive_pipeline_delivers_first_audio_before_rewriting_and_fits_remaining_budget(modules, monkeypatch, tmp_path, recorded_closing):
    from streaming.config import OutputConfig
    tts = modules.tts
    monkeypatch.setattr(tts.LengthEstimator, 'count_words', lambda text: len(text.split()))
    monkeypatch.setattr(tts, '_estimate_duration', lambda text: len(text.split()) * .46)
    events = []
    speech_rate = 80.352 / 215 if recorded_closing else .05
    budget = 120. if recorded_closing else 2.
    source = " ".join(["word"] * 215) if recorded_closing else "One two. Three four five six seven. Eight nine ten eleven twelve."

    def query(client, text, **kwargs):
        seconds = len(text.split()) * speech_rate
        return {'audio_seconds': seconds, 'tts_api_s': 0., 'mp3_parse_s': 0.,
                'mp3_bytes': str(seconds).encode()}

    def rewrite(client, text, words, *args, **kwargs):
        events.append('rewrite')
        assert events[0] == 'first_audio'
        return ' '.join(['word'] * words)

    monkeypatch.setattr(tts, '_query_time_profiled', query)
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    monkeypatch.setattr(tts.AudioSegment, 'from_file',
                        lambda stream, **kw: AudioSegment.silent(round(float(stream.getvalue()) * 1000)))
    monkeypatch.setattr(tts.AudioSegment, 'export', lambda self, output, **kw: output.write_bytes(b'audio') if isinstance(output, Path) else output.write(b'audio'))
    cfg = OutputConfig(budget_mode="audio_duration", adaptive_delivery=True, first_chunk_seconds=12 if recorded_closing else .92,
                       later_chunk_seconds=30 if recorded_closing else 2.3,
                       min_tolerance_seconds=.03, tolerance_ratio=.05,
                       last_chunk_upper_tolerance_ratio=.05, max_refinements=6)
    profiles, _, _, _ = tts.run_pipeline(
        Mock(), [source], budget,
        config=cfg, out_dir=tmp_path,
        on_chunk=lambda i, *a: events.append('first_audio' if i == 0 else 'later_audio'),
    )
    assert events[0] == 'first_audio'
    assert 'rewrite' in events
    assert profiles[0].n_ref_used == 0
    assert len(profiles) >= 3
    assert abs(sum(p.audio_seconds for p in profiles) - budget) <= (6. if recorded_closing else .12)
    assert all(p.target_reached == tts._in_range(p.audio_seconds, p.target_s, p.tolerance_s, p.tol_upper_s)
               for p in profiles)
