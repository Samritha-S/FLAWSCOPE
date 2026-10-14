"""Time-warping and alignment mapping of baseline features onto participant timeline."""

from typing import Any, Callable
import numpy as np

import config


def build_time_map(words: list[Any]) -> Callable[[np.ndarray | float], np.ndarray | float]:
    """Build piecewise linear time map from participant time to baseline time.

    Uses word boundary anchors (p_start -> b_start) and (p_end -> b_end)
    for words matched in BOTH recordings.

    Args:
        words: List of Word contract objects or dicts with timing fields.

    Returns:
        Callable time_map(t_p) returning t_b.
    """
    p_anchors = [0.0]
    b_anchors = [0.0]

    for w in words:
        if isinstance(w, dict):
            matched_b = w.get("matched_b", False)
            matched_p = w.get("matched_p", False)
            p_s = w.get("p_start")
            p_e = w.get("p_end")
            b_s = w.get("b_start")
            b_e = w.get("b_end")
        else:
            matched_b = getattr(w, "matched_b", False)
            matched_p = getattr(w, "matched_p", False)
            p_s = getattr(w, "p_start", None)
            p_e = getattr(w, "p_end", None)
            b_s = getattr(w, "b_start", None)
            b_e = getattr(w, "b_end", None)

        if matched_b and matched_p and p_s is not None and b_s is not None:
            p_anchors.append(float(p_s))
            b_anchors.append(float(b_s))
        if matched_b and matched_p and p_e is not None and b_e is not None:
            p_anchors.append(float(p_e))
            b_anchors.append(float(b_e))

    # Ensure anchors are sorted and remove duplicate p_anchors
    p_arr = np.array(p_anchors, dtype=np.float64)
    b_arr = np.array(b_anchors, dtype=np.float64)

    # Sort by p_arr
    sort_idx = np.argsort(p_arr)
    p_sorted = p_arr[sort_idx]
    b_sorted = b_arr[sort_idx]

    # Deduplicate consecutive identical p values
    unique_mask = np.concatenate(([True], np.diff(p_sorted) > 1e-4))
    p_final = p_sorted[unique_mask]
    b_final = b_sorted[unique_mask]

    def time_map(t_p: np.ndarray | float) -> np.ndarray | float:
        return np.interp(t_p, p_final, b_final, left=b_final[0], right=b_final[-1])

    return time_map


def warp_baseline(
    frames_b: dict[str, np.ndarray],
    t_p_grid: list[float] | np.ndarray,
    time_map: Callable[[np.ndarray | float], np.ndarray | float],
) -> dict[str, np.ndarray]:
    """Warp baseline curves onto the participant 100 Hz grid using time_map.

    - rms_db_rel: linear interpolation
    - f0_st: nearest neighbour (so NaN unvoiced gaps are not smeared)

    Args:
        frames_b: Baseline frame features dict.
        t_p_grid: Participant grid times (100 Hz).
        time_map: Time mapping function t_p -> t_b.

    Returns:
        Dict of warped arrays: {'f0_st', 'rms_db_rel'}.
    """
    t_p = np.array(t_p_grid, dtype=np.float64)
    t_b = np.array(time_map(t_p), dtype=np.float64)

    N_b = len(frames_b["f0_st"])
    t_b_grid = np.arange(N_b, dtype=np.float64) * config.HOP_S

    # Linear interpolation for rms_db_rel
    warped_rms = np.interp(
        t_b, t_b_grid, frames_b["rms_db_rel"], left=frames_b["rms_db_rel"][0], right=frames_b["rms_db_rel"][-1]
    ).astype(np.float32)

    # Nearest neighbour for f0_st to avoid smearing NaNs
    idx_b = np.clip(np.round(t_b / config.HOP_S).astype(int), 0, N_b - 1)
    warped_f0 = frames_b["f0_st"][idx_b].astype(np.float32)

    return {
        "f0_st": warped_f0,
        "rms_db_rel": warped_rms,
    }


def rate_curve(
    words: list[Any],
    side: str,
    grid: list[float] | np.ndarray,
    word_feats: list[Any] | None = None,
) -> np.ndarray:
    """Generate step function curve of speech rate (syllables/s) per word on timeline.

    Returns 0.0 where there is no active word.

    Args:
        words: List of Word schema objects or word dicts.
        side: 'p' (participant) or 'b' (baseline).
        grid: 100 Hz timeline grid points.
        word_feats: Optional list of WordFeat schema objects or dicts.

    Returns:
        Numpy array of rate_sps step values aligned with grid.
    """
    t_arr = np.array(grid, dtype=np.float64)
    curve = np.zeros(len(t_arr), dtype=np.float32)

    for idx, w in enumerate(words):
        if isinstance(w, dict):
            p_s = w.get("p_start", w.get("start"))
            p_e = w.get("p_end", w.get("end"))
        else:
            p_s = getattr(w, "p_start", None) or getattr(w, "start", None)
            p_e = getattr(w, "p_end", None) or getattr(w, "end", None)

        if p_s is None or p_e is None:
            continue

        # Extract rate_sps from word_feats if provided, otherwise from w
        rate = 0.0
        if word_feats and idx < len(word_feats):
            wf = word_feats[idx]
            side_feat = getattr(wf, side, None) if not isinstance(wf, dict) else wf.get(side)
            if side_feat:
                rate = (
                    getattr(side_feat, "rate_sps", 0.0)
                    if not isinstance(side_feat, dict)
                    else side_feat.get("rate_sps", 0.0)
                ) or 0.0
        elif isinstance(w, dict):
            rate = w.get(f"{side}_rate", w.get("rate_sps", 0.0))
        else:
            rate = getattr(w, f"{side}_rate", getattr(w, "rate_sps", 0.0))

        # Fill word interval [p_s, p_e)
        mask = (t_arr >= float(p_s)) & (t_arr < float(p_e))
        curve[mask] = float(rate)

    return curve
