"""Rubric dimension and overall scoring for FlawScope."""

import math
from typing import Any
import numpy as np

import config
from contracts.schema import Flaw, Scores, WordFeat


def compute_scores(
    flaws: list[Flaw],
    word_feats: list[WordFeat],
    config_overrides: dict[str, Any] | None = None,
) -> Scores:
    """Compute 5 rubric dimension scores and overall score.

    For each dimension d:
      penalty_d = sum(severity_i * duration_i for flaws in d) / T_speech
      score_d = 100 * exp(-PENALTY_K * penalty_d)
      overall = sum(WEIGHTS[d] * score_d)

    Args:
        flaws: List of validated Flaw objects.
        word_feats: List of WordFeat objects to compute T_speech.
        config_overrides: Optional runtime configuration overrides.

    Returns:
        Validated contracts.schema.Scores object rounded to 1 decimal place.
    """
    conf = config.snapshot()
    if config_overrides:
        conf.update(config_overrides)

    penalty_k = conf["PENALTY_K"]
    weights = conf["WEIGHTS"]
    flaw_dim = conf["FLAW_DIMENSION"]

    # Total participant speech duration (sum of participant word durations)
    t_speech = sum(
        float(wf.p.dur or 0.0)
        for wf in word_feats
        if wf.p and wf.p.dur is not None
    )
    if t_speech <= 0.0:
        t_speech = 1.0

    # Dimension penalties
    dim_penalties: dict[str, float] = {dim: 0.0 for dim in weights}

    for f in flaws:
        dim = flaw_dim.get(str(f.type.value if hasattr(f.type, "value") else f.type))
        if dim in dim_penalties:
            dur = max(0.0, f.end - f.start)
            dim_penalties[dim] += float(f.severity * dur)

    # Compute exponential dimension scores
    dim_scores: dict[str, float] = {}
    for dim in weights:
        penalty_ratio = dim_penalties[dim] / t_speech
        score = 100.0 * math.exp(-penalty_k * penalty_ratio)
        dim_scores[dim] = round(float(np.clip(score, 0.0, 100.0)), 1)

    # Overall weighted score
    overall = sum(weights[dim] * dim_scores[dim] for dim in weights)
    overall = round(float(np.clip(overall, 0.0, 100.0)), 1)

    return Scores(
        overall=overall,
        pace=dim_scores["pace"],
        intonation=dim_scores["intonation"],
        volume=dim_scores["volume"],
        pausing=dim_scores["pausing"],
        clarity=dim_scores["clarity"],
    )
