"""Deterministic causal explanation and drill recommendations for detected flaws."""

from typing import Any
from contracts.schema import Flaw, FlawType, Word

METRIC_CONFIG = {
    FlawType.MONOTONE: {
        "metric": "std_f0_st",
        "unit": "semitones",
    },
    FlawType.TOO_FAST: {
        "metric": "rate_sps",
        "unit": "syllables/s",
    },
    FlawType.TOO_SLOW: {
        "metric": "rate_sps",
        "unit": "syllables/s",
    },
    FlawType.LONG_PAUSE: {
        "metric": "pause_gap_s",
        "unit": "seconds",
    },
    FlawType.LOW_VOLUME: {
        "metric": "rms_db_rel",
        "unit": "dB",
    },
    FlawType.UNCLEAR_ARTICULATION: {
        "metric": "mfcc_dtw_cost",
        "unit": "MFCC DTW cost",
    },
}

RECOMMENDATIONS = {
    FlawType.MONOTONE: (
        "Practice exaggerating vocal inflection upward on key verbs and downward on statement endings."
    ),
    FlawType.TOO_FAST: (
        "Slow down cadence on transition words and take deliberate breaths between clauses."
    ),
    FlawType.TOO_SLOW: (
        "Pick up pacing and maintain forward momentum through multi-syllable phrases."
    ),
    FlawType.LONG_PAUSE: (
        "Bridge conceptual thoughts smoothly without prolonged hesitation between phrases."
    ),
    FlawType.LOW_VOLUME: (
        "Maintain steady breath support through clause endings to ensure voice projection does not fade."
    ),
    FlawType.UNCLEAR_ARTICULATION: (
        "Crisply articulate initial and terminal consonants, particularly on multisyllabic words."
    ),
}


def _quote_words(words: list[Word], start_idx: int, end_idx: int) -> str:
    """Extract quote of up to 8 words from region with ellipsis if truncated."""
    n_words = len(words)
    s = max(0, min(n_words - 1, start_idx))
    e = max(s, min(n_words - 1, end_idx))
    sub = words[s : min(e + 1, s + 8)]
    text = " ".join(w.text for w in sub)
    if (e + 1 - s) > 8:
        text += " ..."
    return text


def explain_flaw(raw_flaw: dict[str, Any], words: list[Word]) -> Flaw:
    """Generate deterministic explanation, acoustic deltas, and recommendation for a flaw.

    Args:
        raw_flaw: Raw dictionary from detectors.detect_flaws.
        words: List of Word contract objects.

    Returns:
        Validated contracts.schema.Flaw instance.
    """
    flaw_type = FlawType(raw_flaw["type"])
    start = float(raw_flaw["start"])
    end = float(raw_flaw["end"])
    w_start = int(raw_flaw["word_start_idx"])
    w_end = int(raw_flaw["word_end_idx"])
    sev = float(raw_flaw["severity"])
    p_val = raw_flaw.get("participant_value")
    b_val = raw_flaw.get("baseline_value")
    z = raw_flaw.get("z_score")

    cfg = METRIC_CONFIG[flaw_type]
    metric_name = cfg["metric"]
    unit = cfg["unit"]
    recommendation = RECOMMENDATIONS[flaw_type]

    # Delta percentage calculation
    if p_val is not None and b_val is not None and abs(b_val) > 1e-4:
        delta_pct = round(((p_val - b_val) / abs(b_val)) * 100.0, 1)
        delta_str = f"{delta_pct:+.1f}%"
    else:
        delta_pct = None
        delta_str = "N/A"

    quoted = _quote_words(words, w_start, w_end)

    z_str = f"{z:+.1f}" if z is not None else "0.0"

    # Template per flaw type
    if flaw_type == FlawType.MONOTONE:
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') pitch varied by only "
            f"{p_val:.1f} {unit} vs {b_val:.1f} in the reference ({delta_str}, z = {z_str}). "
            "Flat pitch removes the emphasis cues listeners use to follow the argument."
        )
    elif flaw_type == FlawType.TOO_FAST:
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') speaking rate accelerated to "
            f"{p_val:.1f} {unit} vs {b_val:.1f} in the reference ({delta_str}, z = {z_str}). "
            "Rushed delivery impairs clarity and prevents listeners from processing the content."
        )
    elif flaw_type == FlawType.TOO_SLOW:
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') speaking rate dragged at "
            f"{p_val:.1f} {unit} vs {b_val:.1f} in the reference ({delta_str}, z = {z_str}). "
            "Dragging tempo causes energy to drop and risks losing audience engagement."
        )
    elif flaw_type == FlawType.LONG_PAUSE:
        gap_diff = (p_val - b_val) if (p_val is not None and b_val is not None) else 0.0
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') silent pause lasted "
            f"{p_val:.1f} {unit} vs {b_val:.1f} in the reference ({delta_str}, gap delta: +{gap_diff:.1f} s). "
            "Unintended silence breaks conversational rhythm."
        )
    elif flaw_type == FlawType.LOW_VOLUME:
        drop = (b_val - p_val) if (p_val is not None and b_val is not None) else 0.0
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') vocal volume dropped to "
            f"{p_val:.1f} {unit} vs {b_val:.1f} in the reference ({delta_str}, drop: {drop:.1f} dB, z = {z_str}). "
            "Trailing off reduces intelligibility and impact."
        )
    elif flaw_type == FlawType.UNCLEAR_ARTICULATION:
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') spectral divergence rose to "
            f"{p_val:.2f} {unit} vs reference baseline ({delta_str}, z = {z_str}). "
            "Muffled phonetic transitions obscure consonant clarity."
        )
    else:
        explanation = (
            f"Between {start:.1f} s and {end:.1f} s ('{quoted}') deviation detected "
            f"on {metric_name} ({delta_str}, z = {z_str})."
        )

    return Flaw(
        id=raw_flaw.get("id", "F1"),
        type=flaw_type,
        start=start,
        end=end,
        word_start_idx=w_start,
        word_end_idx=w_end,
        severity=sev,
        metric=metric_name,
        unit=unit,
        participant_value=p_val,
        baseline_value=b_val,
        delta_pct=delta_pct,
        z_score=z,
        explanation=explanation,
        recommendation=recommendation,
    )


def explain_flaws(raw_flaws: list[dict[str, Any]], words: list[Word]) -> list[Flaw]:
    """Generate Flaw contract instances with causal explanations for all raw flaws.

    Args:
        raw_flaws: List of raw flaw dictionaries.
        words: List of Word contract objects.

    Returns:
        List of validated Flaw objects.
    """
    return [explain_flaw(rf, words) for rf in raw_flaws]
