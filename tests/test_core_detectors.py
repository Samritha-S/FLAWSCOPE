"""Unit tests for flaw detectors, causal explanations, and rubric scoring."""

import math
import numpy as np
import pytest

import config
from contracts.schema import FlawType, Scores, SideFeat, Word, WordFeat
from core.detectors import detect_flaws
from core.explain import explain_flaws
from core.pipeline import analyze
from core.scoring import compute_scores


def _make_dummy_word_data(n_words: int = 15, base_dur: float = 0.3):
    """Create aligned words and word_feats with identical baseline and participant."""
    words: list[Word] = []
    word_feats: list[WordFeat] = []
    cur_t = 0.2

    for i in range(n_words):
        w_start = round(cur_t, 3)
        w_end = round(cur_t + base_dur, 3)
        cur_t = w_end + 0.05

        word = Word(
            idx=i,
            text=f"word{i}",
            norm=f"word{i}",
            has_punct_after=False,
            b_start=w_start,
            b_end=w_end,
            p_start=w_start,
            p_end=w_end,
            matched_b=True,
            matched_p=True,
        )
        words.append(word)

        p_feat = SideFeat(
            dur=base_dur,
            f0_mean_st=0.0,
            f0_std_st=2.0,
            rms_db_rel=-15.0,
            pause_before=0.05,
            rate_sps=3.3,
            mfcc_dtw_cost=0.15,
            syllables=1,
        )
        b_feat = SideFeat(
            dur=base_dur,
            f0_mean_st=0.0,
            f0_std_st=2.0,
            rms_db_rel=-15.0,
            pause_before=0.05,
            rate_sps=3.3,
            mfcc_dtw_cost=0.15,
            syllables=1,
        )
        word_feats.append(WordFeat(idx=i, p=p_feat, b=b_feat))

    # Frame features
    N = int(math.floor(cur_t / config.HOP_S)) + 50
    frames_p = {
        "f0_st": np.full(N, 1.0, dtype=np.float32),
        "rms_db_rel": np.full(N, -15.0, dtype=np.float32),
        "mfcc": np.zeros((N, 12), dtype=np.float32),
    }
    # Add sinusoidal variation to baseline F0 so std > 0
    t_arr = np.arange(N) * config.HOP_S
    frames_b = {
        "f0_st": (2.0 * np.sin(2 * np.pi * 2.0 * t_arr)).astype(np.float32),
        "rms_db_rel": np.full(N, -15.0, dtype=np.float32),
        "mfcc": np.zeros((N, 12), dtype=np.float32),
    }

    return words, word_feats, frames_p, frames_b


def test_monotone_detector():
    """Verify MONOTONE detector triggers when participant pitch variation is flat."""
    words, word_feats, frames_p, frames_b = _make_dummy_word_data(15, base_dur=0.4)

    # Participant F0 is completely constant across frames (std = 0)
    frames_p["f0_st"][:] = 0.5
    # Baseline F0 has strong pitch oscillations (std ~ 2 semitones)
    t_arr = np.arange(len(frames_b["f0_st"])) * config.HOP_S
    frames_b["f0_st"] = (3.0 * np.sin(2 * np.pi * 3.0 * t_arr)).astype(np.float32)

    flaws = detect_flaws(words, word_feats, frames_p, frames_b)
    types = [f["type"] for f in flaws]
    assert FlawType.MONOTONE in types


def test_too_fast_and_too_slow_detectors():
    """Verify PACE detectors trigger on rushed and dragging tempos."""
    # Fast test
    words_f, word_feats_f, frames_p_f, frames_b_f = _make_dummy_word_data(15, base_dur=0.4)
    for i in range(4, 9):
        # Participant rushes: 4 syllables in short duration
        word_feats_f[i].p.syllables = 4
        word_feats_f[i].p.dur = 0.2
        word_feats_f[i].p.rate_sps = 20.0

    flaws_fast = detect_flaws(words_f, word_feats_f, frames_p_f, frames_b_f)
    assert any(f["type"] == FlawType.TOO_FAST for f in flaws_fast)

    # Slow test
    words_s, word_feats_s, frames_p_s, frames_b_s = _make_dummy_word_data(15, base_dur=0.5)
    for i in range(4, 9):
        # Participant drags: 1 syllable in 1.2s
        word_feats_s[i].p.syllables = 1
        word_feats_s[i].p.dur = 1.2
        word_feats_s[i].p.rate_sps = 0.8
        words_s[i].p_end = words_s[i].p_start + 1.2
    # Adjust following words
    for k in range(9, len(words_s)):
        words_s[k].p_start = words_s[k-1].p_end + 0.05
        words_s[k].p_end = words_s[k].p_start + 0.5

    flaws_slow = detect_flaws(words_s, word_feats_s, frames_p_s, frames_b_s)
    assert any(f["type"] == FlawType.TOO_SLOW for f in flaws_slow)


def test_low_volume_detector():
    """Verify LOW_VOLUME detector flags when intensity drops >= 4 dB with d <= -Z."""
    words, word_feats, frames_p, frames_b = _make_dummy_word_data(15, base_dur=0.4)

    # Baseline intensity around -15 dB
    frames_b["rms_db_rel"][:] = -15.0
    # Participant drops by 10 dB to -25 dB in middle segment
    frames_p["rms_db_rel"][:] = -15.0
    w_start_frame = int(round(words[4].p_start / config.HOP_S))
    w_end_frame = int(round(words[9].p_end / config.HOP_S))
    frames_p["rms_db_rel"][w_start_frame:w_end_frame] = -25.0

    flaws = detect_flaws(words, word_feats, frames_p, frames_b)
    assert any(f["type"] == FlawType.LOW_VOLUME for f in flaws)


def test_long_pause_detector():
    """Verify LONG_PAUSE triggers on gap >= 0.9s with gap delta >= 0.5s."""
    words, word_feats, frames_p, frames_b = _make_dummy_word_data(10, base_dur=0.3)

    # Inject pause before word 5: participant has 1.4s pause, baseline has 0.1s
    words[5].p_start = words[4].p_end + 1.4
    words[5].p_end = words[5].p_start + 0.3
    word_feats[5].p.pause_before = 1.4
    word_feats[5].b.pause_before = 0.1

    flaws = detect_flaws(words, word_feats, frames_p, frames_b)
    pause_flaws = [f for f in flaws if f["type"] == FlawType.LONG_PAUSE]
    assert len(pause_flaws) == 1
    assert pause_flaws[0]["participant_value"] == 1.4
    assert pause_flaws[0]["baseline_value"] == 0.1
    assert pause_flaws[0]["start"] == pytest.approx(words[4].p_end, abs=0.01)
    assert pause_flaws[0]["end"] == pytest.approx(words[5].p_start, abs=0.01)


def test_unclear_articulation_detector():
    """Verify UNCLEAR_ARTICULATION flags on high robust z on participant DTW costs."""
    words, word_feats, frames_p, frames_b = _make_dummy_word_data(15, base_dur=0.4)

    # Base costs are low ~0.15
    for wf in word_feats:
        wf.p.mfcc_dtw_cost = 0.15

    # Outlier costs for words 5..8
    for i in range(5, 9):
        word_feats[i].p.mfcc_dtw_cost = 0.85

    flaws = detect_flaws(words, word_feats, frames_p, frames_b)
    assert any(f["type"] == FlawType.UNCLEAR_ARTICULATION for f in flaws)


def test_region_merging_and_min_duration():
    """Verify adjacent flagged words are merged and short regions (< 0.6s) pruned."""
    words, word_feats, frames_p, frames_b = _make_dummy_word_data(6, base_dur=0.04)

    # 1. Very short duration words: total duration < MIN_FLAW_DUR_S (0.6s)
    for i in range(len(words)):
        words[i].p_start = i * 0.05
        words[i].p_end = (i + 1) * 0.05
        words[i].b_start = i * 0.05
        words[i].b_end = (i + 1) * 0.05
        word_feats[i].p.syllables = 4
        word_feats[i].p.dur = 0.05
        word_feats[i].p.rate_sps = 80.0

    flaws = detect_flaws(words, word_feats, frames_p, frames_b)
    # The short region should be pruned
    assert not any(f["type"] == FlawType.TOO_FAST for f in flaws)

    # 2. Long enough region (base_dur 0.3s, 5 words = 1.5s > 0.6s)
    words2, word_feats2, frames_p2, frames_b2 = _make_dummy_word_data(15, base_dur=0.3)
    for i in range(3, 8):
        word_feats2[i].p.syllables = 4
        word_feats2[i].p.dur = 0.15
        word_feats2[i].p.rate_sps = 26.6

    flaws2 = detect_flaws(words2, word_feats2, frames_p2, frames_b2)
    fast_flaws = [f for f in flaws2 if f["type"] == FlawType.TOO_FAST]
    # Flagged words must be merged into 1 continuous region
    assert len(fast_flaws) == 1
    assert fast_flaws[0]["word_start_idx"] <= 3
    assert fast_flaws[0]["word_end_idx"] >= 7


def test_scoring_monotonicity():
    """Verify that adding flaws or increasing severity strictly decreases or maintains scores."""
    _, word_feats, _, _ = _make_dummy_word_data(10, base_dur=0.4)

    # Clean delivery (0 flaws)
    clean_scores = compute_scores([], word_feats)
    assert clean_scores.overall == 100.0

    # Mild flaw
    from contracts.schema import Flaw
    mild_flaw = Flaw(
        id="F1",
        type=FlawType.MONOTONE,
        start=1.0,
        end=2.0,
        word_start_idx=2,
        word_end_idx=4,
        severity=0.3,
        metric="std_f0_st",
        unit="semitones",
        explanation="test",
        recommendation="test",
    )
    mild_scores = compute_scores([mild_flaw], word_feats)
    assert mild_scores.overall < clean_scores.overall
    assert mild_scores.intonation < 100.0
    assert mild_scores.pace == 100.0  # untouched dimensions remain 100

    # Severe flaw
    severe_flaw = Flaw(
        id="F1",
        type=FlawType.MONOTONE,
        start=1.0,
        end=2.0,
        word_start_idx=2,
        word_end_idx=4,
        severity=0.9,
        metric="std_f0_st",
        unit="semitones",
        explanation="test",
        recommendation="test",
    )
    severe_scores = compute_scores([severe_flaw], word_feats)
    assert severe_scores.overall < mild_scores.overall
    assert severe_scores.intonation < mild_scores.intonation


def test_explanation_contains_all_numbers():
    """Verify explain() output contains all required numerical fields, units, and quotes."""
    words, _, _, _ = _make_dummy_word_data(10, base_dur=0.4)
    raw_flaw = {
        "id": "F1",
        "type": FlawType.MONOTONE,
        "start": 1.2,
        "end": 2.8,
        "word_start_idx": 2,
        "word_end_idx": 5,
        "severity": 0.65,
        "z_score": -2.4,
        "participant_value": 0.8,
        "baseline_value": 2.5,
    }

    flaw = explain_flaws([raw_flaw], words)[0]
    exp = flaw.explanation
    rec = flaw.recommendation

    # Must contain times
    assert "1.2 s" in exp
    assert "2.8 s" in exp
    # Must contain quoted words
    assert "word2" in exp
    assert "word5" in exp
    # Must contain numbers
    assert "0.8 semitones" in exp
    assert "2.5" in exp
    assert "z = -2.4" in exp
    assert "%" in exp  # delta pct
    # Concrete drill recommendation
    assert len(rec) > 20


def test_clean_vs_self_pipeline():
    """Verify analyze() on identical clean audio gives 0 flaws and overall score 100.0."""
    audio_path = "tests/fixtures/clean_sample.wav"
    transcript_path = "tests/fixtures/clean_sample.txt"

    res = analyze(audio_path, audio_path, transcript_path)
    assert len(res["flaws"]) == 0
    assert res["scores"]["overall"] == 100.0
    for dim in ["pace", "intonation", "volume", "pausing", "clarity"]:
        assert res["scores"][dim] == 100.0
