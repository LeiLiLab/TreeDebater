"""Duration estimation works without FastSpeech, CUDA, or model downloads."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'


def load(monkeypatch, name, path):
    monkeypatch.syspath_prepend(str(SRC))
    spec = importlib.util.spec_from_file_location(name, SRC / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('text,words', [('', 0), ('... — !!!', 0),
    ("Don't forget long-term plans.\n  2026 matters!", 6), ('word ' * 130, 130)])
def test_cpu_estimate_seconds(monkeypatch, text, words):
    duration = load(monkeypatch, 'utils.speech_duration', 'utils/speech_duration.py')
    assert duration.estimate_speech_seconds(text) == pytest.approx(words * 0.46)


def test_time_mode_does_not_import_fastspeech(monkeypatch):
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper', None)
    monkeypatch.setitem(sys.modules, 'g2p_en', None)
    monkeypatch.setitem(sys.modules, 'syllables', None)
    constants = types.ModuleType('utils.constants')
    constants.openai_api_key = 'unused'
    tool = types.ModuleType('utils.tool')
    tool.remove_citation = lambda text: (text.replace('[1]', ''), '')
    tool.remove_subtitles = lambda text: text
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, tool.__name__, tool)
    module = load(monkeypatch, 'utils.time_estimator', 'utils/time_estimator.py')
    estimator = module.LengthEstimator('time')
    assert estimator.query_time('one two [1]') == pytest.approx(0.92)
    assert estimator.query_time(['one', 'one two']) == pytest.approx([0.46, 0.92])
    assert estimator.query_time('one two', mode='words') == 2
