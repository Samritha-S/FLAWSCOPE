"""Data contract schemas for FlawScope using Pydantic v2."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class FlawType(str, Enum):
    MONOTONE = "MONOTONE"
    TOO_FAST = "TOO_FAST"
    TOO_SLOW = "TOO_SLOW"
    LONG_PAUSE = "LONG_PAUSE"
    LOW_VOLUME = "LOW_VOLUME"
    UNCLEAR_ARTICULATION = "UNCLEAR_ARTICULATION"


class Word(BaseModel):
    idx: int
    text: str
    norm: str
    has_punct_after: bool
    b_start: float | None = None
    b_end: float | None = None
    p_start: float | None = None
    p_end: float | None = None
    matched_b: bool
    matched_p: bool


class SideFeat(BaseModel):
    dur: float | None = None
    f0_mean_st: float | None = None
    f0_std_st: float | None = None
    rms_db_rel: float | None = None
    pause_before: float | None = None
    rate_sps: float | None = None
    mfcc_dtw_cost: float | None = None
    syllables: int | None = None


class WordFeat(BaseModel):
    idx: int
    p: SideFeat
    b: SideFeat


class CurveSet(BaseModel):
    f0_st: list[float | None]
    rms_db_rel: list[float | None]
    rate_sps: list[float | None]


class Curves(BaseModel):
    t: list[float]  # participant grid, 100 Hz
    p: CurveSet
    b: CurveSet  # baseline warped onto t


class Flaw(BaseModel):
    id: str  # "F1".."Fn"
    type: FlawType
    start: float
    end: float
    word_start_idx: int
    word_end_idx: int
    severity: float = Field(ge=0.0, le=1.0)
    metric: str
    unit: str
    participant_value: float | None = None
    baseline_value: float | None = None
    delta_pct: float | None = None
    z_score: float | None = None
    explanation: str
    recommendation: str


class Scores(BaseModel):
    overall: float = Field(ge=0.0, le=100.0)
    pace: float = Field(ge=0.0, le=100.0)
    intonation: float = Field(ge=0.0, le=100.0)
    volume: float = Field(ge=0.0, le=100.0)
    pausing: float = Field(ge=0.0, le=100.0)
    clarity: float = Field(ge=0.0, le=100.0)


class Meta(BaseModel):
    baseline_file: str
    participant_file: str
    duration_b: float
    duration_p: float
    align_backend: str
    matched_ratio_b: float
    matched_ratio_p: float
    config: dict[str, Any]
    timings_s: dict[str, float]
    warnings: list[str]


class AnalysisResult(BaseModel):
    meta: Meta
    words: list[Word]
    word_feats: list[WordFeat]
    curves: Curves
    flaws: list[Flaw]
    scores: Scores
