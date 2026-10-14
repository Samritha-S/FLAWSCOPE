"""Hardening and reliability tests for FlawScope signal engine."""

import time
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

import config
from core.audio_io import load_audio
from core.features import frame_features, word_features
from core.detectors import detect_flaws
from core.scoring import compute_scores
from core.warp import build_time_map, warp_baseline
from core.pipeline import analyze


def test_analyze_good_vs_good_zero_flaws():
    """Verify that comparing identical good audio yields zero flaws and overall score >= 99."""
    base_wav = "dataset/raw/clip01.wav"
    transcript_file = "dataset/transcripts/clip01.txt"

    res = analyze(base_wav, base_wav, transcript_file, no_cache=True)
    assert len(res["flaws"]) == 0, f"Expected 0 flaws on good-vs-good, got {len(res['flaws'])}"
    assert res["scores"]["overall"] >= 99.0, f"Expected score >= 99, got {res['scores']['overall']}"


def test_hardening_short_audio(tmp_path):
    """Verify audio under 3 seconds returns a valid result with a warning, not a crash."""
    sr = config.SR
    # 1.5 second sine wave audio
    dur = 1.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    y = 0.3 * np.sin(2 * np.pi * 300 * t)

    short_wav = tmp_path / "short.wav"
    sf.write(str(short_wav), y, sr)

    transcript = "hello world short test"
    res = analyze(str(short_wav), str(short_wav), transcript, no_cache=True)

    assert "meta" in res
    warnings = res["meta"]["warnings"]
    assert any("very short (< 3.0s)" in w for w in warnings)
    assert res["meta"]["duration_b"] == pytest.approx(1.5, abs=0.1)


def test_hardening_transcript_mismatch(tmp_path):
    """Verify that severe transcript mismatch (matched_ratio < 0.5) generates a warning and does not crash."""
    base_wav = "dataset/raw/clip01.wav"
    bogus_transcript = "quantum chromodynamics gravitational singularities astrophysical black hole accretion disk"

    res = analyze(base_wav, base_wav, bogus_transcript, no_cache=True)
    assert res["meta"]["matched_ratio_b"] < 0.5
    warnings = res["meta"]["warnings"]
    assert any("Low alignment match ratio" in w and "< 50%" in w for w in warnings)
    assert "scores" in res
    assert "flaws" in res


def test_hardening_no_voiced_frames():
    """Verify that audio with zero voiced frames (pure silence) runs safely without crashing."""
    dur_s = 4.0
    N = int(dur_s / config.HOP_S) + 1
    silent_y = np.zeros(int(config.SR * dur_s), dtype=np.float32)

    feats = frame_features(silent_y)
    assert np.all(np.isnan(feats["f0_st"]))
    assert len(feats["rms_db_rel"]) == N

    # Feed into warp and detect
    time_map = lambda t: t
    warped = warp_baseline(feats, np.arange(N) * config.HOP_S, time_map)
    assert len(warped["f0_st"]) == N

    flaws = detect_flaws([], [], feats, feats)
    assert flaws == []

    scores = compute_scores([], [])
    assert scores.overall == 100.0


def test_hardening_stereo_and_resampling(tmp_path):
    """Verify loading 44.1 kHz stereo audio converts seamlessly to mono 16 kHz float32."""
    sr_in = 44100
    dur = 2.0
    t = np.linspace(0, dur, int(sr_in * dur), endpoint=False, dtype=np.float32)
    ch1 = 0.2 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.2 * np.cos(2 * np.pi * 880 * t)
    stereo = np.column_stack([ch1, ch2])

    stereo_wav = tmp_path / "stereo_44k.wav"
    sf.write(str(stereo_wav), stereo, sr_in)

    y_out, dur_out = load_audio(stereo_wav)
    assert y_out.ndim == 1  # mono
    assert y_out.dtype == np.float32
    assert len(y_out) == int(config.SR * dur)
    assert dur_out == pytest.approx(dur, rel=1e-2)


def test_hardening_value_error_on_missing_file():
    """Verify that invalid/missing audio paths raise a descriptive ValueError."""
    with pytest.raises(ValueError) as exc_info:
        analyze("non_existent_b.wav", "non_existent_p.wav", "some text")
    assert "not found" in str(exc_info.value).lower()


def test_caching_speed_and_bypass():
    """Verify cached analysis run executes in < 2 seconds and --no-cache bypasses it."""
    base_wav = "dataset/raw/clip01.wav"
    transcript_file = "dataset/transcripts/clip01.txt"

    # First run (populate cache)
    res1 = analyze(base_wav, base_wav, transcript_file)

    # Second run (from cache)
    t0 = time.time()
    res2 = analyze(base_wav, base_wav, transcript_file)
    elapsed_cached = time.time() - t0

    assert elapsed_cached < 2.0, f"Cached run took {elapsed_cached:.3f}s, expected < 2.0s"
    assert res1["scores"]["overall"] == res2["scores"]["overall"]
    assert len(res1["flaws"]) == len(res2["flaws"])

    # Bypass cache with no_cache=True
    res_uncached = analyze(base_wav, base_wav, transcript_file, no_cache=True)
    assert res_uncached["scores"]["overall"] == res1["scores"]["overall"]
