"""Flaw detection comparing participant vs baseline over word-matched windows."""

from typing import Any
import numpy as np
import librosa

import config
from contracts.schema import FlawType, Word, WordFeat


def compute_word_mfcc_dtw_costs(
    words: list[Word],
    word_feats: list[WordFeat],
    frames_p: dict[str, np.ndarray],
    frames_b: dict[str, np.ndarray],
    conf: dict[str, Any],
) -> None:
    """Compute and populate per-word MFCC DTW costs in word_feats[i].p.mfcc_dtw_cost.

    Args:
        words: List of Word schema objects.
        word_feats: List of WordFeat schema objects to populate.
        frames_p: Participant frame features dictionary.
        frames_b: Baseline frame features dictionary.
        conf: Active configuration dictionary.
    """
    hop_s = conf["HOP_S"]
    min_frames = conf["DTW_MIN_FRAMES"]

    for idx, w in enumerate(words):
        if idx >= len(word_feats):
            continue
        if word_feats[idx].p.mfcc_dtw_cost is not None:
            continue
        if not (w.matched_p and w.matched_b):
            continue
        if w.p_start is None or w.p_end is None or w.b_start is None or w.b_end is None:
            continue

        p_i0 = int(round(w.p_start / hop_s))
        p_i1 = int(round(w.p_end / hop_s))
        b_i0 = int(round(w.b_start / hop_s))
        b_i1 = int(round(w.b_end / hop_s))

        mfcc_p = frames_p["mfcc"][p_i0:p_i1]
        mfcc_b = frames_b["mfcc"][b_i0:b_i1]

        if len(mfcc_p) >= min_frames and len(mfcc_b) >= min_frames:
            try:
                D, wp = librosa.sequence.dtw(X=mfcc_p.T, Y=mfcc_b.T, metric="euclidean")
                cost = float(D[wp[0][0], wp[0][1]] / len(wp))
                word_feats[idx].p.mfcc_dtw_cost = round(cost, 3)
            except Exception:
                word_feats[idx].p.mfcc_dtw_cost = None
        else:
            word_feats[idx].p.mfcc_dtw_cost = None


def detect_flaws(
    words: list[Word],
    word_feats: list[WordFeat],
    frames_p: dict[str, np.ndarray],
    frames_b: dict[str, np.ndarray],
    config_overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Detect time-bounded delivery flaws over word-matched windows.

    Args:
        words: List of Word contract objects.
        word_feats: List of WordFeat contract objects.
        frames_p: Participant frame features dictionary.
        frames_b: Baseline frame features dictionary.
        config_overrides: Optional runtime configuration overrides.

    Returns:
        List of raw flaw dictionaries sorted by start time.
    """
    conf = config.snapshot()
    if config_overrides:
        conf.update(config_overrides)

    win_radius = conf["WIN_RADIUS"]
    z_thresh = conf["Z_THRESH"]
    min_flaw_dur = conf["MIN_FLAW_DUR_S"]
    merge_gap = conf["MERGE_GAP_WORDS"]
    min_voiced = conf["MIN_VOICED_FRAMES"]
    long_pause_s = conf["LONG_PAUSE_S"]
    long_pause_delta_s = conf["LONG_PAUSE_DELTA_S"]
    long_pause_scale = conf["LONG_PAUSE_MAX_SCALE_S"]
    volume_min_drop_db = conf["VOLUME_MIN_DROP_DB"]
    clarity_z_extra = conf["CLARITY_Z_EXTRA"]
    spread_floor = conf["SPREAD_FLOOR"]
    hop_s = conf["HOP_S"]

    # 1. Compute per-word MFCC DTW costs
    compute_word_mfcc_dtw_costs(words, word_feats, frames_p, frames_b, conf)

    n_words = len(words)
    if n_words == 0:
        return []

    # 2. Extract window metrics across all word indices
    # Window for word i is [max(0, i - win_radius), min(n_words - 1, i + win_radius)]
    # Skip window when any word in it is unmatched in either recording
    valid_window_indices: list[int] = []
    m_p_mono: list[float] = []
    m_b_mono: list[float] = []
    mono_indices: list[int] = []

    m_p_pace: list[float] = []
    m_b_pace: list[float] = []
    pace_indices: list[int] = []

    m_p_vol: list[float] = []
    m_b_vol: list[float] = []
    vol_indices: list[int] = []

    m_p_clarity: list[float] = []
    clarity_indices: list[int] = []

    for i in range(n_words):
        w_start_idx = max(0, i - win_radius)
        w_end_idx = min(n_words - 1, i + win_radius)
        window_words = words[w_start_idx : w_end_idx + 1]

        # Skip window when any word in it is unmatched in either recording
        if any(not (w.matched_p and w.matched_b) for w in window_words):
            continue

        valid_window_indices.append(i)

        # Time bounds for window
        tp0 = window_words[0].p_start
        tp1 = window_words[-1].p_end
        tb0 = window_words[0].b_start
        tb1 = window_words[-1].b_end

        if tp0 is None or tp1 is None or tb0 is None or tb1 is None:
            continue

        # Slice frame features
        p_f0 = frames_p["f0_st"][int(round(tp0 / hop_s)) : int(round(tp1 / hop_s))]
        b_f0 = frames_b["f0_st"][int(round(tb0 / hop_s)) : int(round(tb1 / hop_s))]
        p_rms = frames_p["rms_db_rel"][int(round(tp0 / hop_s)) : int(round(tp1 / hop_s))]
        b_rms = frames_b["rms_db_rel"][int(round(tb0 / hop_s)) : int(round(tb1 / hop_s))]

        # MONOTONE: std of f0_st over voiced frames (need >= 15 voiced frames)
        pv = p_f0[~np.isnan(p_f0)]
        bv = b_f0[~np.isnan(b_f0)]
        if len(pv) >= min_voiced and len(bv) >= min_voiced:
            m_p_mono.append(float(np.std(pv)))
            m_b_mono.append(float(np.std(bv)))
            mono_indices.append(i)

        # PACE: sum(syllables) / sum(word durations)
        p_sylls = sum(
            word_feats[k].p.syllables or 1
            for k in range(w_start_idx, w_end_idx + 1)
        )
        p_durs = sum(
            word_feats[k].p.dur or 0.04
            for k in range(w_start_idx, w_end_idx + 1)
        )
        b_sylls = sum(
            word_feats[k].b.syllables or 1
            for k in range(w_start_idx, w_end_idx + 1)
        )
        b_durs = sum(
            word_feats[k].b.dur or 0.04
            for k in range(w_start_idx, w_end_idx + 1)
        )

        if p_durs > 0 and b_durs > 0:
            m_p_pace.append(float(p_sylls / p_durs))
            m_b_pace.append(float(b_sylls / b_durs))
            pace_indices.append(i)

        # LOW_VOLUME: mean rms_db_rel over frames of window words
        if len(p_rms) > 0 and len(b_rms) > 0:
            m_p_vol.append(float(np.mean(p_rms)))
            m_b_vol.append(float(np.mean(b_rms)))
            vol_indices.append(i)

        # CLARITY: mean over words in window of participant DTW cost
        window_dtw_costs = [
            word_feats[k].p.mfcc_dtw_cost
            for k in range(w_start_idx, w_end_idx + 1)
            if word_feats[k].p.mfcc_dtw_cost is not None
        ]
        if window_dtw_costs:
            m_p_clarity.append(float(np.mean(window_dtw_costs)))
            clarity_indices.append(i)

    flagged_by_type: dict[FlawType, list[dict[str, Any]]] = {
        FlawType.MONOTONE: [],
        FlawType.TOO_FAST: [],
        FlawType.TOO_SLOW: [],
        FlawType.LOW_VOLUME: [],
        FlawType.UNCLEAR_ARTICULATION: [],
        FlawType.LONG_PAUSE: [],
    }

    # 3. Flag rule: MONOTONE (d <= -Z)
    if len(m_b_mono) > 0:
        s_b_mono = max(float(np.std(m_b_mono)), spread_floor["MONOTONE"])
        for idx, mp, mb in zip(mono_indices, m_p_mono, m_b_mono):
            d = (mp - mb) / s_b_mono
            if d <= -z_thresh:
                sev = float(np.clip(abs(d) / (2.0 * z_thresh), 0.0, 1.0))
                flagged_by_type[FlawType.MONOTONE].append({
                    "word_idx": idx,
                    "d": d,
                    "severity": sev,
                    "m_p": mp,
                    "m_b": mb,
                })

    # 4. Flag rule: PACE (TOO_FAST: d >= +Z, TOO_SLOW: d <= -Z)
    if len(m_b_pace) > 0:
        s_b_pace = max(float(np.std(m_b_pace)), spread_floor["PACE"])
        for idx, mp, mb in zip(pace_indices, m_p_pace, m_b_pace):
            d = (mp - mb) / s_b_pace
            if d >= z_thresh:
                sev = float(np.clip(abs(d) / (2.0 * z_thresh), 0.0, 1.0))
                flagged_by_type[FlawType.TOO_FAST].append({
                    "word_idx": idx,
                    "d": d,
                    "severity": sev,
                    "m_p": mp,
                    "m_b": mb,
                })
            elif d <= -z_thresh:
                sev = float(np.clip(abs(d) / (2.0 * z_thresh), 0.0, 1.0))
                flagged_by_type[FlawType.TOO_SLOW].append({
                    "word_idx": idx,
                    "d": d,
                    "severity": sev,
                    "m_p": mp,
                    "m_b": mb,
                })

    # 5. Flag rule: LOW_VOLUME (d <= -Z_vol and (m_b - m_p) >= VOLUME_MIN_DROP_DB)
    z_vol_thresh = conf.get("VOLUME_Z_THRESH", z_thresh)
    if len(m_b_vol) > 0:
        s_b_vol = max(float(np.std(m_b_vol)), spread_floor["LOW_VOLUME"])
        for idx, mp, mb in zip(vol_indices, m_p_vol, m_b_vol):
            d = (mp - mb) / s_b_vol
            if d <= -z_vol_thresh and (mb - mp) >= volume_min_drop_db:
                sev = float(np.clip(abs(d) / (2.0 * z_vol_thresh), 0.0, 1.0))
                flagged_by_type[FlawType.LOW_VOLUME].append({
                    "word_idx": idx,
                    "d": d,
                    "severity": sev,
                    "m_p": mp,
                    "m_b": mb,
                })

    # 6. Flag rule: UNCLEAR_ARTICULATION (robust z on participant costs)
    # zc = (cost - median) / (1.4826 * MAD); flagged if zc >= Z + CLARITY_Z_EXTRA
    all_p_costs = [
        wf.p.mfcc_dtw_cost
        for wf in word_feats
        if wf.p.mfcc_dtw_cost is not None
    ]
    if len(all_p_costs) >= 3 and len(m_p_clarity) > 0:
        costs_arr = np.array(all_p_costs, dtype=np.float64)
        med_cost = float(np.median(costs_arr))
        mad = float(np.median(np.abs(costs_arr - med_cost)))
        scale_mad = max(1.4826 * mad, 0.05)
        thresh_clarity = z_thresh + clarity_z_extra

        for idx, mp in zip(clarity_indices, m_p_clarity):
            zc = (mp - med_cost) / scale_mad
            if zc >= thresh_clarity:
                sev = float(np.clip(zc / (2.0 * thresh_clarity), 0.0, 1.0))
                flagged_by_type[FlawType.UNCLEAR_ARTICULATION].append({
                    "word_idx": idx,
                    "d": zc,
                    "severity": sev,
                    "m_p": mp,
                    "m_b": med_cost,
                })

    # 7. Flag rule: LONG_PAUSE (per word gap and smeared word duration)
    for idx in range(1, n_words):
        w_cur = words[idx]
        w_prev = words[idx - 1]
        if not (w_cur.matched_p and w_cur.matched_b and w_prev.matched_p and w_prev.matched_b):
            continue

        gap_p = word_feats[idx].p.pause_before or 0.0
        gap_b = word_feats[idx].b.pause_before or 0.0

        if gap_p >= long_pause_s and (gap_p - gap_b) >= long_pause_delta_s:
            if w_prev.p_end is not None and w_cur.p_start is not None:
                sev = float(np.clip((gap_p - gap_b) / long_pause_scale, 0.0, 1.0))
                d = (gap_p - gap_b) / long_pause_delta_s
                flagged_by_type[FlawType.LONG_PAUSE].append({
                    "word_idx": idx,
                    "word_start_idx": idx - 1,
                    "word_end_idx": idx,
                    "start": float(w_prev.p_end),
                    "end": float(w_cur.p_start),
                    "d": d,
                    "severity": sev,
                    "m_p": gap_p,
                    "m_b": gap_b,
                })

    pause_word_dur_delta = conf.get("LONG_PAUSE_WORD_DUR_DELTA_S", 1.0)
    for idx in range(n_words):
        w_cur = words[idx]
        if not (w_cur.matched_p and w_cur.matched_b):
            continue
        p_dur = word_feats[idx].p.dur or 0.0
        b_dur = word_feats[idx].b.dur or 0.0
        if (p_dur - b_dur) >= pause_word_dur_delta and w_cur.p_start is not None and w_cur.p_end is not None:
            p_i0 = int(round(w_cur.p_start / hop_s))
            p_i1 = int(round(w_cur.p_end / hop_s))
            f0_slice = frames_p["f0_st"][p_i0:p_i1]
            unvoiced_ratio = float(np.mean(np.isnan(f0_slice))) if len(f0_slice) > 0 else 0.0
            if unvoiced_ratio >= 0.3:
                sev = float(np.clip((p_dur - b_dur) / long_pause_scale, 0.0, 1.0))
                d = (p_dur - b_dur) / long_pause_delta_s
                flagged_by_type[FlawType.LONG_PAUSE].append({
                    "word_idx": idx,
                    "word_start_idx": idx,
                    "word_end_idx": idx,
                    "start": float(w_cur.p_start),
                    "end": float(w_cur.p_end),
                    "d": d,
                    "severity": sev,
                    "m_p": p_dur,
                    "m_b": b_dur,
                })

    # 8. Region building: merge flagged words whose indices differ by <= MERGE_GAP_WORDS
    raw_flaws: list[dict[str, Any]] = []

    for flaw_type, items in flagged_by_type.items():
        if not items:
            continue

        if flaw_type == FlawType.LONG_PAUSE:
            # Gaps are already time-bounded intervals
            for it in items:
                raw_flaws.append({
                    "type": flaw_type,
                    "start": round(it["start"], 3),
                    "end": round(it["end"], 3),
                    "word_start_idx": it["word_start_idx"],
                    "word_end_idx": it["word_end_idx"],
                    "severity": round(it["severity"], 3),
                    "z_score": round(it["d"], 2),
                    "participant_value": round(it["m_p"], 2),
                    "baseline_value": round(it["m_b"], 2),
                })
            continue

        # Sort flagged words by word_idx
        items_sorted = sorted(items, key=lambda x: x["word_idx"])
        groups: list[list[dict[str, Any]]] = []
        cur_group: list[dict[str, Any]] = [items_sorted[0]]

        for next_item in items_sorted[1:]:
            # Merge flagged words whose indices differ by <= MERGE_GAP_WORDS (difference <= merge_gap + 1)
            if next_item["word_idx"] - cur_group[-1]["word_idx"] <= (merge_gap + 1):
                cur_group.append(next_item)
            else:
                groups.append(cur_group)
                cur_group = [next_item]
        if cur_group:
            groups.append(cur_group)

        for g in groups:
            w_first_idx = g[0]["word_idx"]
            w_last_idx = g[-1]["word_idx"]
            r_start = words[w_first_idx].p_start
            r_end = words[w_last_idx].p_end

            if r_start is None or r_end is None:
                continue

            r_dur = r_end - r_start
            # Drop regions shorter than MIN_FLAW_DUR_S (except LONG_PAUSE)
            if r_dur < min_flaw_dur:
                continue

            # Severity = mean severity
            sev_mean = float(np.mean([it["severity"] for it in g]))

            # z_score = the most extreme d
            if flaw_type in (FlawType.MONOTONE, FlawType.TOO_SLOW, FlawType.LOW_VOLUME):
                z_extreme = min(it["d"] for it in g)
            else:
                z_extreme = max(it["d"] for it in g)

            # participant_value and baseline_value = mean of window metrics over region
            p_val = float(np.mean([it["m_p"] for it in g]))
            b_val = float(np.mean([it["m_b"] for it in g]))

            raw_flaws.append({
                "type": flaw_type,
                "start": round(float(r_start), 3),
                "end": round(float(r_end), 3),
                "word_start_idx": w_first_idx,
                "word_end_idx": w_last_idx,
                "severity": round(sev_mean, 3),
                "z_score": round(float(z_extreme), 2),
                "participant_value": round(p_val, 2),
                "baseline_value": round(b_val, 2),
            })

    # 9. Sort by start time and number F1..Fn
    raw_flaws.sort(key=lambda f: (f["start"], f["end"]))
    for idx, f in enumerate(raw_flaws, start=1):
        f["id"] = f"F{idx}"

    return raw_flaws
