"""Unit tests for frame features, word features, time warping, and pipeline integration."""

import numpy as np
import pytest

import config
from contracts.schema import AnalysisResult, Word
from core.features import count_syllables, frame_features, word_features
from core.pipeline import analyze
from core.warp import build_time_map, rate_curve, warp_baseline


def test_f0_st_octave_jump():
    """Synthetic 200 Hz tone then 400 Hz tone -> f0_st differs by about 12 +/- 0.5 semitones."""
    sr = config.SR
    dur_half = 1.0
    t = np.linspace(0, dur_half, int(sr * dur_half), endpoint=False, dtype=np.float32)

    # 200 Hz tone for 1 second, then 400 Hz tone for 1 second
    y1 = 0.5 * np.sin(2 * np.pi * 200.0 * t)
    y2 = 0.5 * np.sin(2 * np.pi * 400.0 * t)
    y = np.concatenate([y1, y2])

    frames = frame_features(y)
    f0_st = frames["f0_st"]

    # Sample middle of 200 Hz segment (frames 20 to 80)
    voiced_1 = f0_st[20:80]
    voiced_1 = voiced_1[~np.isnan(voiced_1)]
    mean_f0_1 = np.mean(voiced_1)

    # Sample middle of 400 Hz segment (frames 120 to 180)
    voiced_2 = f0_st[120:180]
    voiced_2 = voiced_2[~np.isnan(voiced_2)]
    mean_f0_2 = np.mean(voiced_2)

    delta_semitones = mean_f0_2 - mean_f0_1
    assert pytest.approx(12.0, abs=0.5) == delta_semitones


def test_time_map_identity():
    """Identical baseline and participant words produce an identity time mapping."""
    words = [
        Word(
            idx=0,
            text="hello",
            norm="hello",
            has_punct_after=False,
            b_start=0.2,
            b_end=0.6,
            p_start=0.2,
            p_end=0.6,
            matched_b=True,
            matched_p=True,
        ),
        Word(
            idx=1,
            text="world",
            norm="world",
            has_punct_after=True,
            b_start=0.8,
            b_end=1.4,
            p_start=0.8,
            p_end=1.4,
            matched_b=True,
            matched_p=True,
        ),
    ]

    time_map = build_time_map(words)

    for test_t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.1, 1.4]:
        mapped = time_map(test_t)
        assert pytest.approx(test_t, abs=1e-3) == mapped


def test_time_map_slowdown():
    """A participant speaking 2x slower maps t_p = 2.0 to about t_b = 1.0."""
    words = [
        Word(
            idx=0,
            text="speech",
            norm="speech",
            has_punct_after=False,
            b_start=0.0,
            b_end=1.0,
            p_start=0.0,
            p_end=2.0,
            matched_b=True,
            matched_p=True,
        ),
    ]

    time_map = build_time_map(words)
    mapped = time_map(2.0)
    assert pytest.approx(1.0, abs=0.05) == mapped


def test_syllable_counter():
    """Verify syllable counting heuristic on test cases."""
    cases = {
        "the": 1,
        "speech": 1,
        "beautiful": 3,
        "make": 1,
        "strengths": 1,
    }

    for word, expected in cases.items():
        assert count_syllables(word) == expected, f"Failed on '{word}': got {count_syllables(word)}, expected {expected}"


def test_all_frame_arrays_same_length():
    """Ensure f0_st, rms_db_rel, mfcc, and logmel all share length N = floor(dur / 0.01) + 1."""
    dur = 1.37  # Non-integer second duration
    sr = config.SR
    y = np.random.RandomState(7).randn(int(sr * dur)).astype(np.float32) * 0.1

    frames = frame_features(y)
    expected_N = int(np.floor(dur / config.HOP_S)) + 1

    assert len(frames["f0_st"]) == expected_N
    assert len(frames["rms_db_rel"]) == expected_N
    assert frames["mfcc"].shape == (expected_N, 12)
    assert frames["logmel"].shape == (expected_N, 64)


def test_word_features_and_warp():
    """Test word feature extraction and baseline warping."""
    words = [
        {"idx": 0, "text": "test", "norm": "test", "start": 0.1, "end": 0.4},
        {"idx": 1, "text": "speech", "norm": "speech", "start": 0.5, "end": 0.9},
    ]

    N = 100
    frames = {
        "f0_st": np.full(N, 2.0, dtype=np.float32),
        "rms_db_rel": np.full(N, -10.0, dtype=np.float32),
    }

    feats = word_features(words, frames)
    assert len(feats) == 2
    assert feats[0]["dur"] == pytest.approx(0.3, abs=0.01)
    assert feats[0]["syllables"] == 1
    assert feats[0]["rate_sps"] == pytest.approx(1.0 / 0.3, abs=0.1)
    assert feats[1]["pause_before"] == pytest.approx(0.1, abs=0.01)

    # Test baseline warping
    time_map = lambda t: t
    t_grid = [round(i * 0.01, 2) for i in range(50)]
    warped = warp_baseline(frames, t_grid, time_map)
    assert len(warped["f0_st"]) == 50
    assert len(warped["rms_db_rel"]) == 50


def test_pipeline_analyze_contract():
    """Test full analyze() on clean fixtures and validate against AnalysisResult schema."""
    audio_path = "tests/fixtures/clean_sample.wav"
    transcript_path = "tests/fixtures/clean_sample.txt"

    result_dict = analyze(audio_path, audio_path, transcript_path)
    validated = AnalysisResult.model_validate(result_dict)

    assert validated.meta.duration_b > 0
    assert validated.meta.duration_p > 0
    assert len(validated.words) > 0
    assert len(validated.word_feats) == len(validated.words)
    assert len(validated.curves.t) == len(validated.curves.p.f0_st)
    assert len(validated.curves.t) == len(validated.curves.b.f0_st)
    assert validated.scores.overall == 100.0
    assert len(validated.flaws) == 0
    assert "load" in validated.meta.timings_s
    assert "features" in validated.meta.timings_s
    assert "warp" in validated.meta.timings_s
