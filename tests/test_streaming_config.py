import argparse
from pathlib import Path

import pytest
import yaml

from streaming.config import OutputConfig, add_streaming_arguments, resolve_config


def parse(*args):
    parser = argparse.ArgumentParser()
    add_streaming_arguments(parser)
    return parser.parse_args(args)


def test_cli_yaml_defaults_and_saved_values():
    full = {'streaming': {'input': {'min_audio_seconds': 9, 'min_text_words': 12},
                          'output': {'voice': 'alloy', 'max_parallel_tts': 2}},
            'env': {'speech_budgets': {'opening': 100}}}
    args = parse('--min-audio-seconds', '7', '--max-text-wait-seconds', '0')
    settings = resolve_config(full, args)
    assert settings.input.min_audio_seconds == args.min_audio_seconds == 7
    assert args.min_text_words == 12
    assert args.poll_interval == 1
    assert settings.output.voice == 'alloy'
    assert full['streaming']['input']['min_audio_seconds'] == 7
    assert full['streaming']['output']['max_parallel_tts'] == 2
    assert full['env']['speech_budgets'] == {'opening': 100, 'rebuttal': 240, 'closing': 120}


@pytest.mark.parametrize('overlap,audio,words', [(True, 15, 40), (False, 30, 50)])
def test_legacy_cli_defaults(overlap, audio, words):
    args = parse()
    resolve_config({}, args, overlap=overlap)
    assert args.min_audio_seconds == audio
    assert args.min_text_words == words


@pytest.mark.parametrize('section,values', [
    ('input', {'poll_interval': 0}), ('input', {'min_audio_seconds': -1}),
    ('input', {'min_text_words': 1.5}), ('input', {'min_audio_seconds': float('nan')}),
    ('output', {'max_parallel_tts': 0}), ('output', {'enable_early_cut': 'false'}),
    ('output', {'speed_adjust_min': 2, 'speed_adjust_max': 1}),
    ('output', {'adaptive_delivery': 'true'}),
    ('output', {'first_chunk_seconds': 0}), ('output', {'later_chunk_seconds': -1}),
    ('output', {'refinement_model': ''}), ('output', {'typo': 1}),
    ('posthoc', {'split_mode': 'unknown'}), ('playback', {'increment_seconds': 0}),
])
def test_invalid_yaml_rejected(section, values):
    with pytest.raises(ValueError):
        resolve_config({'streaming': {section: values}})


def test_invalid_cli_override_rejected():
    with pytest.raises(ValueError):
        resolve_config({}, parse('--chunk-seconds', '0'))


def test_invalid_speech_budget_rejected():
    with pytest.raises(ValueError):
        resolve_config({'env': {'speech_budgets': {'opening': -1}}})


def test_runs_do_not_share_settings():
    first = resolve_config({'streaming': {'output': {'voice': 'alloy'}}})
    second = resolve_config({})
    first.output.max_parallel_tts = 1
    assert second.output == OutputConfig()


def test_existing_and_updated_configs_resolve():
    root = Path(__file__).resolve().parents[1] / 'src' / 'configs'
    for path in [root / 'overlap_debate_2.yml', root / 'overlap_debate_base.yml',
                 root / 'non_overlap_debate.yml', *root.glob('emnlp/case*/*.yml')]:
        full = yaml.safe_load(path.read_text())
        settings = resolve_config(full)
        assert settings.output.max_parallel_tts == 8


def test_adaptive_delivery_config_round_trip():
    full = {"streaming": {"output": {"adaptive_delivery": True, "first_chunk_seconds": 10, "later_chunk_seconds": 25}}}
    settings = resolve_config(full, parse())
    assert settings.output.adaptive_delivery is True
    assert settings.output.first_chunk_seconds == 10
    assert full["streaming"]["output"]["later_chunk_seconds"] == 25
    assert OutputConfig().adaptive_delivery is False
