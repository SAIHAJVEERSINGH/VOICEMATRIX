from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    app: str
    detector_ready: bool


class DetectionResponse(BaseModel):
    verdict: Literal[
        "likely_genuine",
        "potentially_synthetic",
        "inconclusive",
    ]

    # Existing normalized score, retained for frontend compatibility.
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    reliability: Literal[
        "experimental",
        "limited",
        "insufficient_data",
    ]

    explanation: str
    duration_seconds: float
    model_name: str

    inference_ms: float = Field(default=0.0, ge=0.0)
    window_count: int = Field(default=1, ge=1)
    window_score_range: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    # Additional diagnostic fields; older frontend fields remain unchanged.
    bona_fide_logit: float | None = None
    window_bona_fide_logits: list[float] = Field(default_factory=list)
    quality_reasons: list[str] = Field(default_factory=list)

    audio_rms: float | None = Field(default=None, ge=0.0)
    peak_amplitude: float | None = Field(default=None, ge=0.0)
    clipping_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    near_silent_frame_fraction: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    warning: str
