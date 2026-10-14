"""Generate mock AnalysisResult payload and write to contracts/mock_result.json."""

import json
import math
import sys
from pathlib import Path

# Add project root to sys.path so config and contracts can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from contracts.schema import (
    AnalysisResult,
    CurveSet,
    Curves,
    Flaw,
    FlawType,
    Meta,
    Scores,
    SideFeat,
    Word,
    WordFeat,
)

SAMPLE_TRANSCRIPT = (
    "Good morning ladies and gentlemen today I want to share a critical perspective on "
    "modern communication where every spoken word carries weight and subtle inflections "
    "shape our understanding. When delivery lacks precision or energy the true message is "
    "often lost leaving listeners confused disengaged and uninspired. We must practice "
    "deliberate cadence appropriate pausing clear articulation and expressive tone to succeed."
)


def generate_mock_result() -> AnalysisResult:
    raw_words = SAMPLE_TRANSCRIPT.split()
    total_words = len(raw_words)
    duration_p = 24.0
    duration_b = 23.8

    # Word-level timing generation
    words: list[Word] = []
    word_feats: list[WordFeat] = []

    # Allocate rough time per word (0.35s speech + spacing)
    cur_p = 0.5
    cur_b = 0.4
    for idx, w in enumerate(raw_words):
        clean_text = w.rstrip(".,;:")
        has_punct = w[-1] in ".,;:"
        dur_p = 0.32
        dur_b = 0.31

        # Pause after punctuation
        pause_p = 0.45 if has_punct else 0.06
        pause_b = 0.35 if has_punct else 0.05

        # Intentionally inject longer pause around words 27-28
        if idx == 27:
            pause_p = 1.40
            pause_b = 0.30

        p_start = round(cur_p, 3)
        p_end = round(cur_p + dur_p, 3)
        b_start = round(cur_b, 3)
        b_end = round(cur_b + dur_b, 3)

        cur_p = p_end + pause_p
        cur_b = b_end + pause_b

        word_obj = Word(
            idx=idx,
            text=w,
            norm=clean_text.lower(),
            has_punct_after=has_punct,
            b_start=b_start,
            b_end=b_end,
            p_start=p_start,
            p_end=p_end,
            matched_b=True,
            matched_p=True,
        )
        words.append(word_obj)

        # Baseline & participant features
        p_side = SideFeat(
            dur=dur_p,
            f0_mean_st=0.0 if not (6 <= idx <= 10) else -0.8,
            f0_std_st=1.8 if not (6 <= idx <= 10) else 0.5,
            rms_db_rel=-18.0 if not (33 <= idx <= 37) else -26.5,
            pause_before=pause_p,
            rate_sps=3.6 if not (16 <= idx <= 21) else 5.8,
            mfcc_dtw_cost=0.15 if not (50 <= idx <= 54) else 0.65,
            syllables=max(1, len(clean_text) // 3),
        )
        b_side = SideFeat(
            dur=dur_b,
            f0_mean_st=0.2,
            f0_std_st=2.2,
            rms_db_rel=-17.8,
            pause_before=pause_b,
            rate_sps=3.5,
            mfcc_dtw_cost=0.12,
            syllables=max(1, len(clean_text) // 3),
        )
        word_feats.append(WordFeat(idx=idx, p=p_side, b=b_side))

    # 100 Hz curves generation (2400 points)
    num_points = int(duration_p * 100)
    t_grid = [round(i * 0.01, 2) for i in range(num_points)]

    p_f0: list[float | None] = []
    b_f0: list[float | None] = []
    p_rms: list[float | None] = []
    b_rms: list[float | None] = []
    p_rate: list[float | None] = []
    b_rate: list[float | None] = []

    for t in t_grid:
        # Base natural oscillations
        b_pitch = 2.0 * math.sin(2 * math.pi * 0.4 * t) + 0.5 * math.cos(2 * math.pi * 0.9 * t)
        b_loud = -18.0 + 3.0 * math.sin(2 * math.pi * 0.25 * t)
        b_r = 3.6 + 0.4 * math.sin(2 * math.pi * 0.15 * t)

        # Participant defaults tracking baseline with small variations
        p_pitch = b_pitch + 0.3 * math.sin(2 * math.pi * 0.5 * t)
        p_loud = b_loud + 0.2 * math.cos(2 * math.pi * 0.3 * t)
        p_r = b_r

        # MONOTONE region: 2.5s - 4.5s
        if 2.5 <= t <= 4.5:
            p_pitch = 0.2  # almost zero variation

        # TOO_FAST region: 6.5s - 8.2s
        if 6.5 <= t <= 8.2:
            p_r = 5.8

        # LONG_PAUSE region: 10.0s - 11.4s
        if 10.0 <= t <= 11.4:
            p_loud = -45.0  # silence
            p_pitch = None

        # LOW_VOLUME region: 13.5s - 15.2s
        if 13.5 <= t <= 15.2:
            p_loud = -26.5

        # TOO_SLOW region: 16.5s - 18.5s
        if 16.5 <= t <= 18.5:
            p_r = 1.9

        # UNCLEAR_ARTICULATION region: 19.5s - 21.2s
        if 19.5 <= t <= 21.2:
            p_loud = b_loud - 1.5

        p_f0.append(round(p_pitch, 2) if p_pitch is not None else None)
        b_f0.append(round(b_pitch, 2) if b_pitch is not None else None)
        p_rms.append(round(p_loud, 2) if p_loud is not None else None)
        b_rms.append(round(b_loud, 2) if b_loud is not None else None)
        p_rate.append(round(p_r, 2) if p_r is not None else None)
        b_rate.append(round(b_r, 2) if b_r is not None else None)

    curves = Curves(
        t=t_grid,
        p=CurveSet(f0_st=p_f0, rms_db_rel=p_rms, rate_sps=p_rate),
        b=CurveSet(f0_st=b_f0, rms_db_rel=b_rms, rate_sps=b_rate),
    )

    # 6 Flaws covering each FlawType
    flaws = [
        Flaw(
            id="F1",
            type=FlawType.MONOTONE,
            start=2.5,
            end=4.5,
            word_start_idx=6,
            word_end_idx=10,
            severity=0.72,
            metric="std_f0_st",
            unit="semitones",
            participant_value=0.6,
            baseline_value=2.4,
            delta_pct=-75.0,
            z_score=-2.2,
            explanation="Pitch variation dropped to 0.6 semitones (baseline: 2.4 semitones, -75%, z = -2.2), resulting in flat, monotone delivery.",
            recommendation="Vary pitch inflection across key phrases to emphasize important concepts and maintain engagement.",
        ),
        Flaw(
            id="F2",
            type=FlawType.TOO_FAST,
            start=6.5,
            end=8.2,
            word_start_idx=16,
            word_end_idx=21,
            severity=0.68,
            metric="rate_sps",
            unit="syllables/s",
            participant_value=5.8,
            baseline_value=3.6,
            delta_pct=61.1,
            z_score=2.4,
            explanation="Speech tempo accelerated to 5.8 syllables per second (baseline: 3.6 sps, +61%, z = +2.4), creating a rushed cadence.",
            recommendation="Slow down delivery during conceptual points to allow listeners time to process the message.",
        ),
        Flaw(
            id="F3",
            type=FlawType.LONG_PAUSE,
            start=10.0,
            end=11.4,
            word_start_idx=27,
            word_end_idx=28,
            severity=0.44,
            metric="pause_gap_s",
            unit="seconds",
            participant_value=1.4,
            baseline_value=0.3,
            delta_pct=366.7,
            z_score=2.1,
            explanation="Unintended pause duration of 1.4 seconds between words (baseline: 0.3 seconds, gap delta: +1.1 s).",
            recommendation="Shorten hesitation pauses between adjacent clauses to maintain narrative momentum.",
        ),
        Flaw(
            id="F4",
            type=FlawType.LOW_VOLUME,
            start=13.5,
            end=15.2,
            word_start_idx=33,
            word_end_idx=37,
            severity=0.62,
            metric="rms_db_rel",
            unit="dB",
            participant_value=-26.5,
            baseline_value=-18.2,
            delta_pct=-45.6,
            z_score=-2.3,
            explanation="Vocal volume dropped by 8.3 dB relative to baseline (participant: -26.5 dB, baseline: -18.2 dB, z = -2.3).",
            recommendation="Project your voice steadily through sentence endings so phrase closures remain distinct.",
        ),
        Flaw(
            id="F5",
            type=FlawType.TOO_SLOW,
            start=16.5,
            end=18.5,
            word_start_idx=42,
            word_end_idx=47,
            severity=0.55,
            metric="rate_sps",
            unit="syllables/s",
            participant_value=1.9,
            baseline_value=3.8,
            delta_pct=-50.0,
            z_score=-2.1,
            explanation="Speaking tempo slowed to 1.9 syllables per second (baseline: 3.8 sps, -50%, z = -2.1).",
            recommendation="Pick up pacing to preserve conversational rhythm and energy.",
        ),
        Flaw(
            id="F6",
            type=FlawType.UNCLEAR_ARTICULATION,
            start=19.5,
            end=21.2,
            word_start_idx=50,
            word_end_idx=54,
            severity=0.58,
            metric="mfcc_dtw_cost",
            unit="cost",
            participant_value=0.65,
            baseline_value=0.12,
            delta_pct=441.7,
            z_score=2.4,
            explanation="Phonetic deviation detected with MFCC DTW divergence (robust z = 2.4, threshold: 2.0), signaling muffled articulation.",
            recommendation="Enunciate consonant clusters distinctly and ensure crisp closure on syllable boundaries.",
        ),
    ]

    # Weighted scores computation
    # Weights: pace: 0.25, intonation: 0.25, volume: 0.15, pausing: 0.15, clarity: 0.20
    score_pace = 78.5
    score_intonation = 82.0
    score_volume = 85.0
    score_pausing = 80.0
    score_clarity = 84.0
    overall = (
        0.25 * score_pace
        + 0.25 * score_intonation
        + 0.15 * score_volume
        + 0.15 * score_pausing
        + 0.20 * score_clarity
    )

    scores = Scores(
        overall=round(overall, 1),
        pace=score_pace,
        intonation=score_intonation,
        volume=score_volume,
        pausing=score_pausing,
        clarity=score_clarity,
    )

    meta = Meta(
        baseline_file="clip01_baseline.wav",
        participant_file="clip01_participant.wav",
        duration_b=duration_b,
        duration_p=duration_p,
        align_backend="whisper",
        matched_ratio_b=1.0,
        matched_ratio_p=1.0,
        config=config.snapshot(),
        timings_s={
            "load": 0.08,
            "align": 1.25,
            "features": 0.45,
            "warp": 0.12,
            "detect": 0.09,
            "explain_score": 0.04,
            "total": 2.03,
        },
        warnings=[],
    )

    return AnalysisResult(
        meta=meta,
        words=words,
        word_feats=word_feats,
        curves=curves,
        flaws=flaws,
        scores=scores,
    )


def main():
    result = generate_mock_result()
    # Validate result
    validated = AnalysisResult.model_validate(result)
    output_path = Path(__file__).resolve().parent / "mock_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(validated.model_dump_json(indent=2))
    print(f"Successfully generated and validated {output_path} ({len(validated.words)} words, {len(validated.flaws)} flaws)")


if __name__ == "__main__":
    main()
