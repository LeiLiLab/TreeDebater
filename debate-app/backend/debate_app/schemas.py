from typing import Literal, Optional
import math
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, protected_namespaces=()
    )


class RehearsalSettings(Model):
    enabled: bool = False
    pool_name: Literal['gemma-4-26b-a4b', 'deepseek-chat'] = 'gemma-4-26b-a4b'
    mode: Literal['hybrid', 'local'] = 'hybrid'


class SessionSettings(Model):
    motion: str = Field(min_length=3, max_length=2000)
    mode: Literal["human_ai", "ai_ai"] = "human_ai"
    human_side: Literal["for", "against"] = "for"
    first_side: Literal["for", "against"] = "for"
    engine: Literal["treedebater", "demo"] = "treedebater"
    ai_model: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    helper_model: Optional[str] = None
    planning: dict = Field(default_factory=dict)
    rehearsal: RehearsalSettings = Field(default_factory=RehearsalSettings)
    claim_pool_size: int = Field(default=4, ge=1, le=50)
    claim_selection_strategy: Literal['native', 'saved_scores'] = 'native'
    budgets: dict[str, float] = Field(
        default_factory=lambda: {"opening": 60.0, "rebuttal": 60.0, "closing": 30.0}
    )
    streaming: dict = Field(default_factory=dict)
    transport_grace_seconds: float = Field(default=3.0, gt=0, le=10)
    model_timeout_seconds: float = Field(default=300.0, gt=0, le=900)
    max_rerecords: int = Field(default=1, ge=0, le=3)
    evaluation: bool = True

    @model_validator(mode="after")
    def validate_engine(self):
        from .config import engine_config
        from streaming.planning import PlanningConfig

        PlanningConfig(**self.planning)

        self.motion = self.motion.strip()
        if len(self.motion) < 3:
            raise ValueError("Enter a debate motion")
        resolved = engine_config(self)
        self.budgets = resolved["env"]["speech_budgets"]
        if any(not math.isfinite(v) or v > 600 for v in self.budgets.values()):
            raise ValueError("Speech budgets must be at most 600 seconds")
        self.streaming = resolved["streaming"]
        return self


class Command(Model):
    key: str = Field(min_length=1, max_length=128)
    attempt_id: Optional[str] = None
    action: Optional[
        Literal[
            "resume", "retry_processing", "rerecord", "finish_received", "skip_empty"
        ]
    ] = None


class ConfigImport(Model):
    text: str = Field(max_length=100000)


class Epoch(Model):
    attempt_id: str
    epoch_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9-]+$")
    sample_rate: int = Field(ge=8000, le=96000)
    capture_server_ms: float
    uncertainty_ms: float = Field(ge=0, le=250)


class Frame(Model):
    type: Literal["frame"] = "frame"
    attempt_id: str
    epoch_id: str
    sequence: int = Field(ge=0)
    sample_start: int = Field(ge=0)
    samples: int = Field(ge=1, le=9600)
    pcm: str = Field(max_length=26000)


class Finish(Model):
    type: Literal["finish"] = "finish"
    attempt_id: str
    epoch_id: str
    last_sequence: int = Field(ge=-1)
    total_samples: int = Field(ge=0)
    key: str = Field(min_length=1, max_length=128)


class Playback(Model):
    turn_id: str
    played_ms: float = Field(ge=0)
