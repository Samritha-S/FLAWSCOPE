"""Main execution pipeline for FlawScope audio alignment, features, and analysis."""

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import numpy as np

import config
from contracts.schema import (
    AnalysisResult,
    CurveSet,
    Curves,
    Meta,
    Scores,
    SideFeat,
    Word,
    WordFeat,
)
from core.alignment import align
from core.audio_io import file_sha1, load_audio
from core.features import frame_features, word_features
from core.warp import build_time_map, rate_curve, warp_baseline


def _clean_curve(arr: Any) -> list[float | None]:
    """Clean and round a curve array: NaN/Inf -> None, round floats to 2 dp."""
    cleaned: list[float | None] = []
    for v in arr:
        if v is None:
            cleaned.append(None)
        else:
            try:
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    cleaned.append(None)
                else:
                    cleaned.append(round(fv, 2))
            except (TypeError, ValueError):
                cleaned.append(None)
    return cleaned


def analyze(
    baseline_audio: str,
    participant_audio: str,
    transcript: str,
    participant_transcript: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Analyze spoken performance against a baseline reference.

    Extracts features, time-warps baseline to participant timeline, and produces
    validated AnalysisResult payload.

    Args:
        baseline_audio: Path to reference baseline audio file.
        participant_audio: Path to participant audio file.
        transcript: Reference transcript text or path to transcript file.
        participant_transcript: Optional participant transcript text.
        config_overrides: Optional dictionary of configuration overrides.

    Returns:
        JSON-serializable dictionary matching contracts.schema.AnalysisResult.
    """
    if os.environ.get("FLAWSCOPE_MOCK") == "1":
        mock_path = Path(__file__).resolve().parent.parent / "contracts" / "mock_result.json"
        with open(mock_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        validated = AnalysisResult.model_validate(raw_data)
        return validated.model_dump(mode="json")

    t_start_total = time.time()

    # Read transcript text if a file path was passed
    p_trans_path = Path(transcript)
    if p_trans_path.exists() and p_trans_path.is_file():
        with open(p_trans_path, "r", encoding="utf-8") as f:
            transcript_text = f.read()
    else:
        transcript_text = transcript

    if participant_transcript:
        p_pt_path = Path(participant_transcript)
        if p_pt_path.exists() and p_pt_path.is_file():
            with open(p_pt_path, "r", encoding="utf-8") as f:
                p_transcript_text = f.read()
        else:
            p_transcript_text = participant_transcript
    else:
        p_transcript_text = transcript_text

    # --- Stage (a): Load audio files ---
    t0 = time.time()
    y_b, dur_b = load_audio(baseline_audio)
    sha1_b = file_sha1(baseline_audio)
    y_p, dur_p = load_audio(participant_audio)
    sha1_p = file_sha1(participant_audio)
    t_load = time.time() - t0

    # --- Stage (b): Align transcripts ---
    t0 = time.time()
    words_b_list, ratio_b, warns_b = align(y_b, transcript_text, sha1=sha1_b)
    words_p_list, ratio_p, warns_p = align(y_p, p_transcript_text, sha1=sha1_p)
    t_align = time.time() - t0

    # --- Stage (c): Extract features ---
    t0 = time.time()
    frames_b = frame_features(y_b)
    frames_p = frame_features(y_p)
    wfeats_b = word_features(words_b_list, frames_b)
    wfeats_p = word_features(words_p_list, frames_p)
    t_features = time.time() - t0

    # --- Stage (d): Warp baseline onto participant grid ---
    t0 = time.time()
    num_words = max(len(words_b_list), len(words_p_list))
    words: list[Word] = []
    word_feats: list[WordFeat] = []

    for i in range(num_words):
        wb = words_b_list[i] if i < len(words_b_list) else None
        wp = words_p_list[i] if i < len(words_p_list) else None

        text = wp["text"] if wp else (wb["text"] if wb else "")
        norm = wp["norm"] if wp else (wb["norm"] if wb else "")
        has_punct = wp["has_punct_after"] if wp else (wb["has_punct_after"] if wb else False)

        word_obj = Word(
            idx=i,
            text=text,
            norm=norm,
            has_punct_after=has_punct,
            b_start=wb["start"] if wb else None,
            b_end=wb["end"] if wb else None,
            p_start=wp["start"] if wp else None,
            p_end=wp["end"] if wp else None,
            matched_b=wb["matched"] if wb else False,
            matched_p=wp["matched"] if wp else False,
        )
        words.append(word_obj)

        p_feat_dict = wfeats_p[i] if i < len(wfeats_p) else {}
        b_feat_dict = wfeats_b[i] if i < len(wfeats_b) else {}
        p_side = SideFeat(
            dur=p_feat_dict.get("dur"),
            f0_mean_st=p_feat_dict.get("f0_mean_st"),
            f0_std_st=p_feat_dict.get("f0_std_st"),
            rms_db_rel=p_feat_dict.get("rms_db_rel"),
            pause_before=p_feat_dict.get("pause_before"),
            rate_sps=p_feat_dict.get("rate_sps"),
            mfcc_dtw_cost=p_feat_dict.get("mfcc_dtw_cost"),
            syllables=p_feat_dict.get("syllables"),
        )
        b_side = SideFeat(
            dur=b_feat_dict.get("dur"),
            f0_mean_st=b_feat_dict.get("f0_mean_st"),
            f0_std_st=b_feat_dict.get("f0_std_st"),
            rms_db_rel=b_feat_dict.get("rms_db_rel"),
            pause_before=b_feat_dict.get("pause_before"),
            rate_sps=b_feat_dict.get("rate_sps"),
            mfcc_dtw_cost=b_feat_dict.get("mfcc_dtw_cost"),
            syllables=b_feat_dict.get("syllables"),
        )
        word_feats.append(WordFeat(idx=i, p=p_side, b=b_side))

    # Build time map and participant grid
    N_p = int(math.floor(dur_p / config.HOP_S)) + 1
    t_p_grid = [round(i * config.HOP_S, 2) for i in range(N_p)]

    time_map = build_time_map(words)
    warped_b = warp_baseline(frames_b, t_p_grid, time_map)

    # Step function rate curves
    rate_p = rate_curve(words, "p", t_p_grid, word_feats)
    rate_b = rate_curve(words, "b", t_p_grid, word_feats)
    t_warp = time.time() - t0

    # --- Stage (e): Assemble Curves and Meta ---
    curves = Curves(
        t=t_p_grid,
        p=CurveSet(
            f0_st=_clean_curve(frames_p["f0_st"][:N_p]),
            rms_db_rel=_clean_curve(frames_p["rms_db_rel"][:N_p]),
            rate_sps=_clean_curve(rate_p[:N_p]),
        ),
        b=CurveSet(
            f0_st=_clean_curve(warped_b["f0_st"][:N_p]),
            rms_db_rel=_clean_curve(warped_b["rms_db_rel"][:N_p]),
            rate_sps=_clean_curve(rate_b[:N_p]),
        ),
    )

    t_total = time.time() - t_start_total
    timings_s = {
        "load": round(t_load, 3),
        "align": round(t_align, 3),
        "features": round(t_features, 3),
        "warp": round(t_warp, 3),
        "detect": 0.0,
        "explain_score": 0.0,
        "total": round(t_total, 3),
    }

    meta = Meta(
        baseline_file=Path(baseline_audio).name,
        participant_file=Path(participant_audio).name,
        duration_b=round(dur_b, 2),
        duration_p=round(dur_p, 2),
        align_backend=config.ALIGN_BACKEND,
        matched_ratio_b=ratio_b,
        matched_ratio_p=ratio_p,
        config=config.snapshot(),
        timings_s=timings_s,
        warnings=warns_b + warns_p,
    )

    # --- Stage (f): Placeholder flaws & scores for Phase A2 ---
    scores = Scores(
        overall=100.0,
        pace=100.0,
        intonation=100.0,
        volume=100.0,
        pausing=100.0,
        clarity=100.0,
    )

    result = AnalysisResult(
        meta=meta,
        words=words,
        word_feats=word_feats,
        curves=curves,
        flaws=[],
        scores=scores,
    )

    return result.model_dump(mode="json")


def main():
    """CLI: python -m core.pipeline <baseline> <participant> <transcript.txt> [--out .cache/result.json]."""
    parser = argparse.ArgumentParser(description="FlawScope Audio Pipeline Runner")
    parser.add_argument("baseline", help="Path to baseline reference audio")
    parser.add_argument("participant", help="Path to participant performance audio")
    parser.add_argument("transcript", help="Path to reference transcript file")
    parser.add_argument("--out", default=".cache/result.json", help="Path for JSON output")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = analyze(args.baseline, args.participant, args.transcript)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    meta = result_dict["meta"]
    timings = meta["timings_s"]

    print("\nFlawScope Analysis Pipeline Complete")
    print("=" * 45)
    print(f"Baseline:     {meta['baseline_file']} ({meta['duration_b']}s)")
    print(f"Participant:  {meta['participant_file']} ({meta['duration_p']}s)")
    print(f"Match B / P:  {meta['matched_ratio_b']:.1%} / {meta['matched_ratio_p']:.1%}")
    print("\nStage Timings:")
    for stage, t in timings.items():
        print(f"  - {stage:<13} {t:>6.3f}s")
    print(f"\nOverall Score: {result_dict['scores']['overall']}")
    print(f"Flaws Detected: {len(result_dict['flaws'])}")
    print(f"Output saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
