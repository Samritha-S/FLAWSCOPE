"""Frame-level and word-level acoustic feature extraction for FlawScope."""

import math
import re
from typing import Any
import numpy as np
import librosa

import config

try:
    import parselmouth
    _HAS_PARSELMOUTH = True
except ImportError:
    _HAS_PARSELMOUTH = False


def count_syllables(norm: str) -> int:
    """Estimate syllable count in a normalized word.

    Heuristic: count vowel groups [aeiouy]+, subtract 1 for a trailing silent 'e'
    when count > 1, with a minimum of 1.

    Args:
        norm: Normalized lowercase word.

    Returns:
        Estimated syllable count (>= 1).
    """
    s = norm.lower()
    vowel_groups = re.findall(r"[aeiouy]+", s)
    count = len(vowel_groups)
    if count > 1 and s.endswith("e"):
        count -= 1
    return max(1, count)


def frame_features(y: np.ndarray) -> dict[str, np.ndarray]:
    """Extract frame-level acoustic features on a 100 Hz grid.

    Extracts:
    - f0_st: Pitch in semitones relative to median voiced F0 (NaN when unvoiced).
    - rms_db_rel: Intensity in dB relative to median speech frame intensity.
    - mfcc: 12 mean/var normalized MFCCs (c0 dropped), shape (N, 12).
    - logmel: Log-mel spectrogram, shape (N, 64) for UI visualization.

    All arrays are padded or trimmed to length N = floor(duration / 0.01) + 1.

    Args:
        y: Audio signal array (float32 at config.SR).

    Returns:
        Dict of numpy arrays: {'f0_st', 'rms_db_rel', 'mfcc', 'logmel'}.
    """
    duration_s = float(len(y) / config.SR)
    N = int(math.floor(duration_s / config.HOP_S)) + 1
    grid_times = np.arange(N, dtype=np.float64) * config.HOP_S

    if _HAS_PARSELMOUTH:
        # Parselmouth Praat pitch extraction
        sound = parselmouth.Sound(y.astype(np.float64), config.SR)
        pitch = sound.to_pitch_ac(
            time_step=config.HOP_S,
            pitch_floor=config.F0_MIN,
            pitch_ceiling=config.F0_MAX,
        )
        raw_f0 = np.array([pitch.get_value_at_time(t) for t in grid_times], dtype=np.float64)

        # Intensity extraction
        intensity = sound.to_intensity(
            time_step=config.HOP_S,
            minimum_pitch=config.F0_MIN,
        )
        raw_rms = np.array([intensity.get_value(t) for t in grid_times], dtype=np.float64)
    else:
        # Fallback to librosa.pyin
        f0_pyin, _, _ = librosa.pyin(
            y,
            fmin=config.F0_MIN,
            fmax=config.F0_MAX,
            sr=config.SR,
            hop_length=int(config.SR * config.HOP_S),
        )
        raw_f0 = np.full(N, np.nan, dtype=np.float64)
        m_len = min(N, len(f0_pyin))
        raw_f0[:m_len] = f0_pyin[:m_len]

        rms_spec = librosa.feature.rms(
            y=y,
            frame_length=int(config.SR * 0.025),
            hop_length=int(config.SR * config.HOP_S),
        )[0]
        raw_db = librosa.amplitude_to_db(rms_spec, ref=np.max) + 80.0
        raw_rms = np.full(N, np.nan, dtype=np.float64)
        m_len = min(N, len(raw_db))
        raw_rms[:m_len] = raw_db[:m_len]

    # --- Pitch (f0_st) in semitones relative to median voiced pitch ---
    # NaN for unvoiced frames
    voiced_mask = ~np.isnan(raw_f0) & (raw_f0 > 0)
    f0_st = np.full(N, np.nan, dtype=np.float32)
    if np.any(voiced_mask):
        median_f0 = float(np.median(raw_f0[voiced_mask]))
        if median_f0 > 0:
            f0_st[voiced_mask] = (12.0 * np.log2(raw_f0[voiced_mask] / median_f0)).astype(np.float32)

    # --- Relative Intensity (rms_db_rel) ---
    # Fill boundary NaNs linearly from nearest valid intensity
    nan_mask = np.isnan(raw_rms)
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        if len(valid_indices) > 0:
            raw_rms[nan_mask] = np.interp(
                np.where(nan_mask)[0], valid_indices, raw_rms[valid_indices]
            )
        else:
            raw_rms = np.zeros(N, dtype=np.float64)

    valid_vals = raw_rms[~np.isnan(raw_rms)]
    if len(valid_vals) > 0:
        p95 = float(np.percentile(valid_vals, 95))
        speech_mask = raw_rms > (p95 - 30.0)
        speech_vals = raw_rms[speech_mask]
        ref_intensity = float(np.median(speech_vals)) if len(speech_vals) > 0 else float(np.median(valid_vals))
        rms_db_rel = (raw_rms - ref_intensity).astype(np.float32)
    else:
        rms_db_rel = np.zeros(N, dtype=np.float32)

    # --- MFCCs (12 coefficients, c0 dropped, mean/var normalized) ---
    raw_mfcc = librosa.feature.mfcc(
        y=y,
        sr=config.SR,
        n_mfcc=13,
        n_fft=400,
        hop_length=160,
        n_mels=40,
    )
    # Drop c0 -> 12 coefficients
    mfcc_12 = raw_mfcc[1:, :]
    mean = np.mean(mfcc_12, axis=1, keepdims=True)
    std = np.std(mfcc_12, axis=1, keepdims=True)
    mfcc_norm = (mfcc_12 - mean) / (std + 1e-8)
    mfcc_t = mfcc_norm.T.astype(np.float32)

    if mfcc_t.shape[0] < N:
        mfcc = np.pad(mfcc_t, ((0, N - mfcc_t.shape[0]), (0, 0)), mode="edge")
    else:
        mfcc = mfcc_t[:N, :]

    # --- Log-mel Spectrogram (n_fft=512, hop_length=160, n_mels=64) ---
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=config.SR,
        n_fft=512,
        hop_length=160,
        n_mels=64,
    )
    logmel_raw = librosa.power_to_db(mel_spec, ref=np.max).T.astype(np.float32)
    if logmel_raw.shape[0] < N:
        logmel = np.pad(logmel_raw, ((0, N - logmel_raw.shape[0]), (0, 0)), mode="edge")
    else:
        logmel = logmel_raw[:N, :]

    return {
        "f0_st": f0_st,
        "rms_db_rel": rms_db_rel,
        "mfcc": mfcc,
        "logmel": logmel,
    }


def word_features(
    words: list[dict[str, Any]],
    frames: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Extract word-level acoustic features from frame features within [start, end).

    Args:
        words: List of word dicts with start, end, norm.
        frames: Dict of frame arrays output by frame_features.

    Returns:
        List of dicts containing: dur, f0_mean_st, f0_std_st, rms_db_rel,
        pause_before, syllables, rate_sps.
    """
    feats: list[dict[str, Any]] = []
    N = len(frames["f0_st"])
    prev_end: float | None = None

    for idx, w in enumerate(words):
        start = float(w["start"])
        end = float(w["end"])
        norm = str(w.get("norm", ""))

        dur = max(0.04, end - start)

        # Frame slice [start, end)
        i_start = max(0, min(N - 1, int(round(start / config.HOP_S))))
        i_end = max(i_start + 1, min(N, int(round(end / config.HOP_S))))

        word_f0 = frames["f0_st"][i_start:i_end]
        word_rms = frames["rms_db_rel"][i_start:i_end]

        voiced = word_f0[~np.isnan(word_f0)]
        if len(voiced) >= 5:
            f0_mean_st = float(np.mean(voiced))
            f0_std_st = float(np.std(voiced))
        else:
            f0_mean_st = None
            f0_std_st = None

        rms_mean = float(np.mean(word_rms)) if len(word_rms) > 0 else None

        pause_before = max(0.0, start - prev_end) if prev_end is not None else 0.0
        prev_end = end

        syllables = count_syllables(norm)
        rate_sps = syllables / dur if dur > 0 else 0.0

        feats.append({
            "idx": idx,
            "dur": round(dur, 3),
            "f0_mean_st": round(f0_mean_st, 2) if f0_mean_st is not None else None,
            "f0_std_st": round(f0_std_st, 2) if f0_std_st is not None else None,
            "rms_db_rel": round(rms_mean, 2) if rms_mean is not None else None,
            "pause_before": round(pause_before, 3),
            "syllables": syllables,
            "rate_sps": round(rate_sps, 2),
            "mfcc_dtw_cost": None,
        })

    return feats
