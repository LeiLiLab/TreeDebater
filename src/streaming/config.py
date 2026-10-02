"""Validated streaming settings. Precedence: explicit CLI values > YAML > defaults."""
from dataclasses import asdict, dataclass, field, fields, replace
import math
from typing import Mapping


class Validated:
    def __post_init__(self):
        nonnegative = {'max_audio_wait_seconds', 'max_text_wait_seconds', 'max_total_audio_seconds',
                       'max_refinements', 'early_max_refinements', 'tolerance_ratio',
                       'last_chunk_upper_tolerance_ratio'}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is bool:
                if type(value) is not bool:
                    raise ValueError(f'{f.name} must be a boolean')
            elif f.type in (float, int):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f'{f.name} must be numeric')
                if f.type is int and type(value) is not int:
                    raise ValueError(f'{f.name} must be an integer')
                if not math.isfinite(value) or (value < 0 if f.name in nonnegative else value <= 0):
                    raise ValueError(f'{f.name} must be finite and {"nonnegative" if f.name in nonnegative else "positive"}')
            elif f.type is str and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{f.name} must be a nonempty string')


def from_mapping(cls, value):
    if isinstance(value, cls):
        return value
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f'{cls.__name__} must be a mapping')
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f'Unknown {cls.__name__} settings: {sorted(unknown)}')
    return cls(**value)


@dataclass
class InputConfig(Validated):
    min_audio_seconds: float = 15.0
    min_text_words: int = 40
    poll_interval: float = 1.0
    audio_format: str = 'mp3'
    max_audio_wait_seconds: float = 0.0
    max_text_wait_seconds: float = 0.0
    max_total_audio_seconds: float = 0.0


@dataclass
class PlaybackConfig(Validated):
    increment_seconds: float = 3.0
    listener_join_timeout: float = 300.0


@dataclass
class OutputConfig(Validated):
    budget_mode: str = 'experiment_elapsed'
    model: str = 'tts-1'
    voice: str = 'echo'
    refinement_model: str = 'gpt-5-mini'
    max_refinements: int = 10
    early_max_refinements: int = 3
    max_parallel_tts: int = 8
    min_chunk_words: int = 30
    tolerance_ratio: float = 0.10
    last_chunk_upper_tolerance_ratio: float = 0.05
    min_tolerance_seconds: float = 1.0
    enable_early_cut: bool = False
    adaptive_delivery: bool = False
    first_chunk_seconds: float = 12.0
    later_chunk_seconds: float = 30.0
    early_cut_ratio: float = 1.25
    ratio_prestart_threshold: float = 2.0
    abs_prestart_chars: int = 1000
    speed_adjust_min: float = 0.85
    speed_adjust_max: float = 1.15

    def __post_init__(self):
        super().__post_init__()
        if self.budget_mode not in ("experiment_elapsed", "audio_duration"):
            raise ValueError("budget_mode must be experiment_elapsed or audio_duration")
        if self.speed_adjust_min > self.speed_adjust_max:
            raise ValueError('speed_adjust_min must not exceed speed_adjust_max')


@dataclass
class PosthocConfig(Validated):
    split_mode: str = 'fixed'
    chunk_seconds: float = 10.0
    silence_window_seconds: float = 0.7

    def __post_init__(self):
        super().__post_init__()
        if self.split_mode not in ('fixed', 'silence'):
            raise ValueError('split_mode must be fixed or silence')


@dataclass
class SpeechBudgets(Validated):
    opening: float = 240.0
    rebuttal: float = 240.0
    closing: float = 120.0


@dataclass
class StreamingConfig:
    input: InputConfig = field(default_factory=InputConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    posthoc: PosthocConfig = field(default_factory=PosthocConfig)

    def __post_init__(self):
        for name, cls in [('input', InputConfig), ('playback', PlaybackConfig),
                          ('output', OutputConfig), ('posthoc', PosthocConfig)]:
            setattr(self, name, from_mapping(cls, getattr(self, name)))


# Existing CLI names remain valid. None means no explicit override.
CLI_FIELDS = {
    'min_audio_seconds': ('input', 'min_audio_seconds'),
    'min_text_words': ('input', 'min_text_words'),
    'poll_interval': ('input', 'poll_interval'),
    'audio_format': ('input', 'audio_format'),
    'max_audio_wait_seconds': ('input', 'max_audio_wait_seconds'),
    'max_text_wait_seconds': ('input', 'max_text_wait_seconds'),
    'max_total_audio_seconds': ('input', 'max_total_audio_seconds'),
    'min_playback_increment': ('playback', 'increment_seconds'),
    'listener_join_timeout': ('playback', 'listener_join_timeout'),
    'split_mode': ('posthoc', 'split_mode'),
    'chunk_seconds': ('posthoc', 'chunk_seconds'),
    'silence_window_seconds': ('posthoc', 'silence_window_seconds'),
}


def add_streaming_arguments(parser):
    defaults = StreamingConfig()
    for arg, (section, name) in CLI_FIELDS.items():
        value = getattr(getattr(defaults, section), name)
        parser.add_argument('--' + arg.replace('_', '-'), type=type(value), default=None,
                            help=f'Override streaming.{section}.{name} from YAML.')


def resolve_config(full_config, args=None, *, overlap=True):
    """Resolve and persist effective settings in the run's config; populate legacy CLI attrs."""
    raw = full_config.get('streaming', {})
    if not isinstance(raw, Mapping):
        raise ValueError('streaming must be a mapping')
    raw = dict(raw)
    if not overlap:
        input_raw = raw.get('input', {})
        if not isinstance(input_raw, Mapping):
            raise ValueError('streaming.input must be a mapping')
        raw['input'] = {'min_audio_seconds': 30.0, 'min_text_words': 50, **input_raw}
    settings = from_mapping(StreamingConfig, raw)
    if args is not None:
        for arg, (section, name) in CLI_FIELDS.items():
            value = getattr(args, arg, None)
            if value is not None:
                setattr(settings, section, replace(getattr(settings, section), **{name: value}))
        for arg, (section, name) in CLI_FIELDS.items():
            setattr(args, arg, getattr(getattr(settings, section), name))
    budgets = from_mapping(SpeechBudgets, full_config.setdefault('env', {}).get('speech_budgets'))
    full_config['env']['speech_budgets'] = asdict(budgets)
    full_config['streaming'] = asdict(settings)
    return settings
