"""Benchmark evaluation runner assessing flaw detection accuracy and scoring monotonicity."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from core.pipeline import analyze

FLAW_TYPES = list(config.FLAW_TYPES)


def compute_temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Compute temporal Intersection-over-Union between two intervals."""
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0.0 else 0.0


def spearman_rank_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation coefficient between two numeric sequences."""
    n = len(x)
    if n < 2:
        return 0.0

    def get_ranks(arr: List[float]) -> np.ndarray:
        temp = np.argsort(arr)
        ranks = np.empty_like(temp, dtype=float)
        ranks[temp] = np.arange(len(arr), dtype=float)
        return ranks

    rx = get_ranks(x)
    ry = get_ranks(y)

    # Standard Pearson correlation of ranks
    std_x = np.std(rx)
    std_y = np.std(ry)
    if std_x == 0.0 or std_y == 0.0:
        return 0.0
    cov = np.mean((rx - np.mean(rx)) * (ry - np.mean(ry)))
    return float(cov / (std_x * std_y))


def load_ground_truth(labels_csv_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load ground-truth flaw intervals grouped by pair_id."""
    truth_by_pair: Dict[str, List[Dict[str, Any]]] = {}
    if not labels_csv_path.exists():
        return truth_by_pair

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("role") != "bad" or row.get("flaw_type") == "NONE":
                continue
            pair_id = row["pair_id"]
            truth_by_pair.setdefault(pair_id, []).append({
                "type": row["flaw_type"],
                "start": float(row["start_s"]),
                "end": float(row["end_s"]),
                "level": int(row["severity_level"]),
                "param": row.get("param", ""),
            })
    return truth_by_pair


def evaluate_pair(
    detected_flaws: List[Dict[str, Any]],
    truth_flaws: List[Dict[str, Any]],
    iou_thresh: float = 0.3,
) -> Tuple[Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    """Greedily match detected flaws to ground truth by IoU >= 0.3 per flaw type."""
    type_stats: Dict[str, Dict[str, int]] = {
        ft: {"TP": 0, "FP": 0, "FN": 0} for ft in FLAW_TYPES
    }
    matches: List[Dict[str, Any]] = []

    for ft in FLAW_TYPES:
        dets = [d for d in detected_flaws if d.get("type") == ft]
        gts = [g for g in truth_flaws if g.get("type") == ft]

        # Calculate all candidate pairs
        candidates = []
        for d_idx, d in enumerate(dets):
            for g_idx, g in enumerate(gts):
                iou = compute_temporal_iou(d["start"], d["end"], g["start"], g["end"])
                if iou >= iou_thresh:
                    candidates.append((iou, d_idx, g_idx))

        # Sort greedy by highest IoU
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_d = set()
        matched_g = set()

        for iou, d_idx, g_idx in candidates:
            if d_idx not in matched_d and g_idx not in matched_g:
                matched_d.add(d_idx)
                matched_g.add(g_idx)
                matches.append({
                    "type": ft,
                    "iou": round(iou, 3),
                    "detected": dets[d_idx],
                    "truth": gts[g_idx],
                })

        tp = len(matched_d)
        fp = len(dets) - tp
        fn = len(gts) - tp

        type_stats[ft]["TP"] = tp
        type_stats[ft]["FP"] = fp
        type_stats[ft]["FN"] = fn

    return type_stats, matches


def run_evaluation(quick: bool = False) -> Dict[str, Any]:
    """Run full evaluation suite over pairs.json and produce benchmark results."""
    pairs_file = PROJECT_ROOT / "dataset" / "pairs.json"
    labels_file = PROJECT_ROOT / "dataset" / "labels.csv"

    if not pairs_file.exists():
        raise FileNotFoundError(f"Missing {pairs_file}")

    with open(pairs_file, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    if quick:
        pairs = [p for p in all_pairs if p["pair_id"].startswith("clip01")]
    else:
        pairs = all_pairs

    truth_by_pair = load_ground_truth(labels_file)

    print(f"Running FlawScope evaluation on {len(pairs)} pairs (quick={quick})...")

    # Aggregate confusion matrix
    agg_type_stats = {ft: {"TP": 0, "FP": 0, "FN": 0} for ft in FLAW_TYPES}
    pair_results = []
    level_scores: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: [], 4: []}
    all_levels_for_rho: List[float] = []
    all_scores_for_rho: List[float] = []

    # 1. Clean reference baseline evaluation (Level 0)
    baseline_clips = sorted(list(set(p["baseline"] for p in pairs)))
    for b_clip in baseline_clips:
        clip_stem = Path(b_clip).stem
        tr_file = PROJECT_ROOT / "dataset" / "transcripts" / f"{clip_stem}.txt"
        if tr_file.exists():
            tr_text = tr_file.read_text(encoding="utf-8").strip()
            res_clean = analyze(
                baseline_audio=str(PROJECT_ROOT / b_clip),
                participant_audio=str(PROJECT_ROOT / b_clip),
                transcript=tr_text,
            )
            clean_score = float(res_clean["scores"]["overall"])
            level_scores[0].append(clean_score)
            all_levels_for_rho.append(0.0)
            all_scores_for_rho.append(clean_score)

    # 2. Evaluate participant pairs
    for p_info in pairs:
        pair_id = p_info["pair_id"]
        base_path = str(PROJECT_ROOT / p_info["baseline"])
        part_path = str(PROJECT_ROOT / p_info["participant"])
        tr_path = PROJECT_ROOT / p_info["transcript"]
        level = int(p_info.get("level", 3))

        tr_text = tr_path.read_text(encoding="utf-8").strip() if tr_path.exists() else ""

        print(f"  Evaluating {pair_id} (Level {level})...")
        res = analyze(
            baseline_audio=base_path,
            participant_audio=part_path,
            transcript=tr_text,
        )

        overall_score = float(res["scores"]["overall"])
        level_scores[level].append(overall_score)
        all_levels_for_rho.append(float(level))
        all_scores_for_rho.append(overall_score)

        dets = res.get("flaws", [])
        gts = truth_by_pair.get(pair_id, [])

        t_stats, matches = evaluate_pair(dets, gts, iou_thresh=0.3)

        for ft in FLAW_TYPES:
            agg_type_stats[ft]["TP"] += t_stats[ft]["TP"]
            agg_type_stats[ft]["FP"] += t_stats[ft]["FP"]
            agg_type_stats[ft]["FN"] += t_stats[ft]["FN"]

        pair_results.append({
            "pair_id": pair_id,
            "level": level,
            "overall_score": overall_score,
            "scores": res["scores"],
            "detected_count": len(dets),
            "truth_count": len(gts),
            "detected": dets,
            "truth": gts,
            "matches": matches,
        })

    # Calculate per-type Precision, Recall, F1
    per_type_metrics = {}
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0

    for ft in FLAW_TYPES:
        tp = agg_type_stats[ft]["TP"]
        fp = agg_type_stats[ft]["FP"]
        fn = agg_type_stats[ft]["FN"]
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_type_metrics[ft] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }

    overall_prec = float(tot_tp / (tot_tp + tot_fp)) if (tot_tp + tot_fp) > 0 else 0.0
    overall_rec = float(tot_tp / (tot_tp + tot_fn)) if (tot_tp + tot_fn) > 0 else 0.0
    overall_f1 = (
        float(2 * overall_prec * overall_rec / (overall_prec + overall_rec))
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    # Severity ladder summary
    ladder_summary = {}
    for lvl in sorted(level_scores.keys()):
        scores = level_scores[lvl]
        mean_s = float(np.mean(scores)) if scores else 0.0
        std_s = float(np.std(scores)) if scores else 0.0
        ladder_summary[f"L{lvl}"] = {
            "level": lvl,
            "count": len(scores),
            "mean_score": round(mean_s, 2),
            "std_score": round(std_s, 2),
        }

    spearman_rho = spearman_rank_correlation(all_levels_for_rho, all_scores_for_rho)

    results_data = {
        "summary": {
            "total_pairs": len(pairs),
            "total_detected": sum(len(p["detected"]) for p in pair_results),
            "total_truth": sum(len(p["truth"]) for p in pair_results),
            "overall_precision": round(overall_prec, 3),
            "overall_recall": round(overall_rec, 3),
            "overall_f1": round(overall_f1, 3),
            "spearman_rho": round(spearman_rho, 3),
        },
        "per_type": per_type_metrics,
        "severity_ladder": ladder_summary,
        "pairs": pair_results,
    }

    # Save results.json
    eval_dir = PROJECT_ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    json_path = eval_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    # Save results.csv
    csv_path = eval_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["flaw_type", "TP", "FP", "FN", "precision", "recall", "f1"])
        for ft in FLAW_TYPES:
            m = per_type_metrics[ft]
            writer.writerow([ft, m["TP"], m["FP"], m["FN"], m["precision"], m["recall"], m["f1"]])
        writer.writerow(["OVERALL", tot_tp, tot_fp, tot_fn, round(overall_prec, 3), round(overall_rec, 3), round(overall_f1, 3)])

    # Print summary table
    print("\n" + "=" * 76)
    print(f"{'Flaw Category':<24} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1':<8}")
    print("-" * 76)
    for ft in FLAW_TYPES:
        m = per_type_metrics[ft]
        print(f"{ft:<24} {m['TP']:<6} {m['FP']:<6} {m['FN']:<6} {m['precision']:<10.3f} {m['recall']:<10.3f} {m['f1']:<8.3f}")
    print("-" * 76)
    print(f"{'OVERALL':<24} {tot_tp:<6} {tot_fp:<6} {tot_fn:<6} {overall_prec:<10.3f} {overall_rec:<10.3f} {overall_f1:<8.3f}")
    print("=" * 76)

    print("\nSeverity Ladder Progression (Mean Overall Score):")
    for lvl_name, l_info in ladder_summary.items():
        print(f"  {lvl_name} (Level {l_info['level']}): {l_info['mean_score']:.1f} pts (n={l_info['count']})")
    print(f"Spearman rank correlation (Level vs Score): rho = {spearman_rho:.3f} (strongly negative expected)")
    print(f"Wrote {json_path} and {csv_path}\n")

    return results_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FlawScope flaw detection and scoring performance.")
    parser.add_argument("--quick", action="store_true", help="Evaluate clip01 pairs only for fast testing")
    args = parser.parse_args()

    run_evaluation(quick=args.quick)


if __name__ == "__main__":
    main()
