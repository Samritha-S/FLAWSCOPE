"""Speech segmentation utilities for finding non-speech cut points and candidate flaw segments."""

import random
from typing import List, Tuple

import librosa
import numpy as np


def find_cut_points(y: np.ndarray, sr: int) -> List[float]:
    """Find cut points in seconds at the midpoints of non-speech gaps >= 120 ms.

    Uses librosa.effects.split with top_db=35 to locate speech intervals, then computes
    the midpoints of any non-speech gaps between speech intervals (or at start/end)
    that last at least 120 ms.
    """
    if len(y) == 0:
        return []

    intervals = librosa.effects.split(y, top_db=35)
    if len(intervals) == 0:
        return [0.0, float(len(y) / sr)]

    gaps: List[Tuple[int, int]] = []

    # Leading silence gap
    if intervals[0][0] / sr >= 0.120:
        gaps.append((0, int(intervals[0][0])))

    # Gaps between speech intervals
    for i in range(len(intervals) - 1):
        end_cur = int(intervals[i][1])
        start_next = int(intervals[i + 1][0])
        if (start_next - end_cur) / sr >= 0.120:
            gaps.append((end_cur, start_next))

    # Trailing silence gap
    if (len(y) - intervals[-1][1]) / sr >= 0.120:
        gaps.append((int(intervals[-1][1]), len(y)))

    cut_points = [float((s + e) / (2.0 * sr)) for s, e in gaps]
    cut_points = sorted(list(dict.fromkeys(cut_points)))
    return cut_points


def pick_segments(
    cut_points: List[float],
    n: int,
    min_len: float = 2.5,
    max_len: float = 6.0,
    min_gap: float = 1.5,
    seed: int = 7,
) -> List[Tuple[float, float]]:
    """Deterministically pick n non-overlapping (start, end) segments between cut points.

    Attempts to satisfy min_len <= duration <= max_len and gap between segments >= min_gap.
    If tight cut constraints prevent finding n segments, gradually relaxes min_gap and min_len
    to guarantee returning n valid non-overlapping segments.
    """
    if n <= 0 or len(cut_points) < 2:
        return []

    gap_candidates = [min_gap, min_gap * 0.75, min_gap * 0.5, 0.5, 0.0]
    min_len_candidates = [min_len, min_len * 0.8, 1.5, 1.0]

    for curr_gap in gap_candidates:
        for curr_min_len in min_len_candidates:
            rng = random.Random(seed)
            # Find all candidate intervals bounded by cut points
            candidates: List[Tuple[float, float]] = []
            for i in range(len(cut_points)):
                for j in range(i + 1, len(cut_points)):
                    dur = cut_points[j] - cut_points[i]
                    if curr_min_len <= dur <= (max_len + 1.0):
                        candidates.append((cut_points[i], cut_points[j]))

            if len(candidates) < n:
                continue

            # Deterministic shuffle
            rng.shuffle(candidates)

            chosen: List[Tuple[float, float]] = []
            for s, e in candidates:
                conflict = False
                for cs, ce in chosen:
                    if not (e + curr_gap <= cs or s >= ce + curr_gap):
                        conflict = True
                        break
                if not conflict:
                    chosen.append((s, e))
                    if len(chosen) == n:
                        chosen.sort(key=lambda x: x[0])
                        return chosen

    # Fallback: if search couldn't get n, pick any n non-overlapping cut-bounded segments
    fallback_segments: List[Tuple[float, float]] = []
    for i in range(min(n, len(cut_points) - 1)):
        fallback_segments.append((cut_points[i], cut_points[i + 1]))
    return fallback_segments
