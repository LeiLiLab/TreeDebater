"""One boundary for draft-length budgets and statement-duration estimates.

LENGTH_MODE_FOR_DRAFT selects the unit used to check unpublished drafts.
TIME_MODE_FOR_STATEMENT selects the estimator whose result is in seconds.
Prompt word budgets are rough conversions; decoded audio remains authoritative.
"""
import math

from .time_estimator import LengthEstimator


def _settings():
    # Read at use time so every caller sees the same two settings.
    from . import constants
    return constants


def _draft_mode():
    mode = _settings().LENGTH_MODE_FOR_DRAFT
    if mode not in ('words', 'syllables', 'phonemes'):
        raise ValueError(f'Unsupported LENGTH_MODE_FOR_DRAFT: {mode}')
    return mode


def seconds_per_word():
    """Coarse conversion for requesting words before any text exists."""
    return _settings().WORDRATIO['time']


def draft_word_budget(seconds, *, rounding='ceil'):
    """Express a time budget as prompt words; this is not a duration check."""
    _draft_mode()
    rounder = {'ceil': math.ceil, 'floor': math.floor, 'round': round}[rounding]
    return max(1, rounder(seconds / seconds_per_word()))


def draft_word_count(text):
    """Measure a draft in the configured unit, normalized to word equivalents.

Existing max_words settings and prompts remain in words. For example, a
phoneme-mode limit of 100 words means 100 * WORDRATIO['phonemes'] phonemes.
"""
    mode = _draft_mode()
    return LengthEstimator(mode).query_time(text) / _settings().WORDRATIO[mode]


def draft_length_instruction(words):
    """Make the configured draft measurement explicit in the drafting prompt."""
    mode = _draft_mode()
    units = math.ceil(words * _settings().WORDRATIO[mode])
    if mode == 'words':
        return f'Draft length target: approximately {units} words. '
    return (f'Draft length target: approximately {units} {mode} '
            f'(roughly {words} words). ')


def statement_estimator(*, audio_duration=None, tts_backend="openai"):
    mode = _settings().TIME_MODE_FOR_STATEMENT
    if mode not in ('time', 'fastspeech', 'openai'):
        raise ValueError(f'Unsupported TIME_MODE_FOR_STATEMENT: {mode}; expected a seconds estimator')
    return LengthEstimator(mode, audio_duration=audio_duration, tts_backend=tts_backend)


def estimate_seconds(text, *, measured_seconds_per_word=None, audio_duration=None, tts_backend="openai"):
    """Return estimated seconds without overriding an explicitly chosen backend.

Only the word-rate ('time') backend uses an observed speaking rate. FastSpeech
and OpenAI modes always run their selected estimator, even after audio arrives.
"""
    estimator = statement_estimator(audio_duration=audio_duration, tts_backend=tts_backend)
    if estimator.mode == 'time' and measured_seconds_per_word is not None:
        rate = float(measured_seconds_per_word)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('Measured seconds per word must be finite and positive')
        value = LengthEstimator('words').query_time(text) * rate
    else:
        value = estimator.query_time(text)
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Estimated speech duration must be finite and nonnegative')
    return value


def duration_fits(estimated_seconds, target_seconds, tolerance=.10):
    return target_seconds * (1 - tolerance) <= estimated_seconds <= target_seconds * (1 + tolerance)


def corrected_word_budget(requested_words, estimated_seconds, target_seconds):
    """Bound the next prompt correction using the selected duration estimate."""
    target_words = draft_word_budget(target_seconds)
    return max(math.ceil(target_words * .5), min(target_words * 2,
        round(requested_words * target_seconds / max(estimated_seconds, .001))))
