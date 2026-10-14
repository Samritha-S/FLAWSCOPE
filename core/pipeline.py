"""Main execution pipeline for FlawScope audio alignment, features, detection, and scoring."""

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import config
from contracts.schema import (
    AnalysisResult,
    CurveSet,
    Curves,
    Meta,
    SideFeat,
    Word,
    WordFeat,
)
from core.alignment import align
from core.audio_io import file_sha1, load_audio
from core.cache import (
    compute_result_hash,
    get_cached_features,
    get_cached_result,
    set_cached_features,
    set_cached_result,
)
from core.detectors import detect_flaws
from core.explain import explain_flaws
from core.features import frame_features, word_features
from core.scoring import compute_scores
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
    no_cache: bool = False,
) -> dict:
    """Analyze spoken performance against a baseline reference.

    Extracts features, time-warps baseline to participant timeline, detects flaws,
    generates causal explanations, and computes rubric scores.

    Args:
        baseline_audio: Path to reference baseline audio file.
        participant_audio: Path to participant audio file.
        transcript: Reference transcript text or path to transcript file.
        participant_transcript: Optional participant transcript text.
        config_overrides: Optional dictionary of configuration overrides.
        no_cache: If True, bypass reading and writing result and feature cache.

    Returns:
        JSON-serializable dictionary matching contracts.schema.AnalysisResult.

    Raises:
        ValueError: On file loading, parsing, or pipeline execution failures.
    """
    if os.environ.get("FLAWSCOPE_MOCK") == "1":
        mock_path = Path(__file__).resolve().parent.parent / "contracts" / "mock_result.json"
        with open(mock_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        validated = AnalysisResult.model_validate(raw_data)
        return validated.model_dump(mode="json")

    t_start_total = time.time()

    p_base = Path(baseline_audio)
    p_part = Path(participant_audio)
    if not p_base.exists():
        raise ValueError(f"Baseline audio file not found: {baseline_audio}")
    if not p_part.exists():
        raise ValueError(f"Participant audio file not found: {participant_audio}")

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

    effective_config = config.snapshot()
    if config_overrides:
        effective_config.update(config_overrides)

    sha1_b = file_sha1(baseline_audio)
    sha1_p = file_sha1(participant_audio)

    # Check whole-result cache
    res_hash = compute_result_hash(sha1_b, sha1_p, transcript_text, effective_config)
    if not no_cache:
        cached_res = get_cached_result(res_hash)
        if cached_res is not None:
            return cached_res

    try:
        # --- Stage (a): Load audio files ---
        t0 = time.time()
        y_b, dur_b = load_audio(baseline_audio)
        y_p, dur_p = load_audio(participant_audio)
        t_load = time.time() - t0

        short_warnings: list[str] = []
        if dur_b < 3.0:
            short_warnings.append(
                f"Baseline audio duration {dur_b:.2f}s is very short (< 3.0s). Results may have degraded accuracy."
            )
        if dur_p < 3.0:
            short_warnings.append(
                f"Participant audio duration {dur_p:.2f}s is very short (< 3.0s). Results may have degraded accuracy."
            )

        # --- Stage (b): Align transcripts ---
        t0 = time.time()
        words_b_list, ratio_b, warns_b = align(y_b, transcript_text, sha1=sha1_b)
        words_p_list, ratio_p, warns_p = align(y_p, p_transcript_text, sha1=sha1_p)
        t_align = time.time() - t0

        # --- Stage (c): Extract features ---
        t0 = time.time()
        frames_b = None if no_cache else get_cached_features(sha1_b)
        if frames_b is None:
            frames_b = frame_features(y_b)
            if not no_cache:
                set_cached_features(sha1_b, frames_b)

        frames_p = None if no_cache else get_cached_features(sha1_p)
        if frames_p is None:
            frames_p = frame_features(y_p)
            if not no_cache:
                set_cached_features(sha1_p, frames_p)

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

        # Assemble Curves
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

        # --- Stage (e): Detect Delivery Flaws ---
        t0 = time.time()
        raw_flaws = detect_flaws(words, word_feats, frames_p, frames_b, config_overrides)
        t_detect = time.time() - t0

        # --- Stage (f): Explain and Score ---
        t0 = time.time()
        flaws = explain_flaws(raw_flaws, words)
        scores = compute_scores(flaws, word_feats, config_overrides)
        t_explain_score = time.time() - t0

        t_total = time.time() - t_start_total
        timings_s = {
            "load": round(t_load, 3),
            "align": round(t_align, 3),
            "features": round(t_features, 3),
            "warp": round(t_warp, 3),
            "detect": round(t_detect, 3),
            "explain_score": round(t_explain_score, 3),
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
            config=effective_config,
            timings_s=timings_s,
            warnings=warns_b + warns_p + short_warnings,
        )

        result = AnalysisResult(
            meta=meta,
            words=words,
            word_feats=word_feats,
            curves=curves,
            flaws=flaws,
            scores=scores,
        )

        res_dict = result.model_dump(mode="json")
        if not no_cache:
            set_cached_result(res_hash, res_dict)

        return res_dict

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"FlawScope analysis failed: {exc}") from exc


def main():
    """CLI: python -m core.pipeline <baseline> <participant> <transcript.txt> [--out .cache/result.json] [--no-cache]."""
    parser = argparse.ArgumentParser(description="FlawScope Audio Pipeline Runner")
    parser.add_argument("baseline", help="Path to baseline reference audio")
    parser.add_argument("participant", help="Path to participant performance audio")
    parser.add_argument("transcript", help="Path to reference transcript file")
    parser.add_argument("--out", default=".cache/result.json", help="Path for JSON output")
    parser.add_argument("--no-cache", action="store_true", help="Bypass reading and writing cache")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = analyze(
        baseline_audio=args.baseline,
        participant_audio=args.participant,
        transcript=args.transcript,
        no_cache=args.no_cache,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    meta = result_dict["meta"]
    timings = meta["timings_s"]

    print("\nFlawScope Analysis Pipeline Complete")
    print("=" * 45)
    print(f"Baseline:     {meta['baseline_file']} ({meta['duration_b']}s)")
    print(f"Participant:  {meta['participant_file']} ({meta['duration_p']}s)")
    print(f"Match B / P:  {meta['matched_ratio_b']:.1%} / {meta['matched_ratio_p']:.1%}")
    if meta.get("warnings"):
        print(f"Warnings:     {len(meta['warnings'])}")
        for w in meta["warnings"]:
            print(f"  * {w}")
    print("\nStage Timings:")
    for stage, t in timings.items():
        print(f"  - {stage:<13} {t:>6.3f}s")
    print(f"\nOverall Score: {result_dict['scores']['overall']}")
    print(f"Flaws Detected: {len(result_dict['flaws'])}")
    for f in result_dict["flaws"]:
        print(f"  [{f['id']}] {f['type']:<20} {f['start']:>5.2f}s - {f['end']:>5.2f}s  sev={f['severity']:.2f}  {f['explanation'][:65]}...")
    print(f"\nOutput saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
