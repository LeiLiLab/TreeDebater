from pathlib import Path
import os
import sys

APP_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = APP_ROOT.parent
sys.path.insert(0, str(ENGINE_ROOT / "src"))
from streaming.config import resolve_config  # pure dataclasses; no model initialization

DATA_ROOT = Path(os.getenv("DEBATE_APP_DATA", str(APP_ROOT / "var"))).resolve()
LIVE_INPUT_FIELDS = {"min_audio_seconds", "min_text_words", "max_text_wait_seconds"}


def positive_env_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def engine_config(settings):
    raw = {"env": {"speech_budgets": settings.budgets}, "streaming": settings.streaming}
    unsupported = set(raw["streaming"]) - {"input", "output"}
    if unsupported:
        raise ValueError(
            "Unsupported live web app streaming sections: " + ", ".join(sorted(unsupported))
            + ". Remove these sections; only input and output are supported."
        )
    if not isinstance(raw["streaming"].get("input", {}), dict):
        raise ValueError("streaming.input must be a mapping")
    unsupported = set(raw["streaming"].get("input", {})) - LIVE_INPUT_FIELDS
    if unsupported:
        raise ValueError(
            "Unsupported live web app streaming.input settings: " + ", ".join(sorted(unsupported))
            + ". Remove these settings; supported settings are "
            + ", ".join(sorted(LIVE_INPUT_FIELDS)) + "."
        )
    output = raw["streaming"].get("output", {})
    if not isinstance(output, dict):
        raise ValueError("streaming.output must be a mapping")
    if output.get("budget_mode", "audio_duration") != "audio_duration":
        raise ValueError("The live web app requires streaming.output.budget_mode: audio_duration")
    raw["streaming"] = {
        "input": {
            "min_audio_seconds": 3.0,
            "min_text_words": 60,
            "max_text_wait_seconds": 15.0,
            **raw["streaming"].get("input", {}),
        },
        **{k: v for k, v in raw["streaming"].items() if k != "input"},
    }
    resolve_config(raw)
    # The shared resolver also supplies CLI-only defaults. Do not expose those
    # as editable app settings or put them back into exported app configs.
    raw["streaming"] = {
        "input": {k: v for k, v in raw["streaming"]["input"].items() if k in LIVE_INPUT_FIELDS},
        "output": raw["streaming"]["output"],
    }
    raw["streaming"]["output"]["budget_mode"] = "audio_duration"
    return raw
