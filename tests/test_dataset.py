"""PyTest suite for dataset generation, segmentation, flaw injection, and validation."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset.inject_flaws import (
    FLAW_RECIPES,
    apply_edge_fades,
    apply_low_volume,
    apply_monotone,
    apply_unclear_articulation,
    time_stretch_audio,
)
from dataset.segmenter import find_cut_points, pick_segments
from dataset.validate_dataset import validate_dataset


def test_baseline_raw_files_exist_and_format():
    """Verify that all baseline WAV files exist, are 16 kHz mono, and between 30 and 60 s."""
    raw_dir = Path("dataset/raw")
    clips = ["clip01.wav", "clip02.wav", "clip03.wav"]
    for clip in clips:
        p = raw_dir / clip
        assert p.exists(), f"Missing baseline file: {p}"
        info = sf.info(str(p))
        assert info.samplerate == 16000
        assert info.channels == 1
        assert 30.0 <= info.duration <= 60.0


def test_transcripts_exist_and_match():
    """Verify transcripts exist for each baseline clip and contain speech text."""
    tr_dir = Path("dataset/transcripts")
    for clip in ["clip01.txt", "clip02.txt", "clip03.txt"]:
        p = tr_dir / clip
        assert p.exists(), f"Missing transcript: {p}"
        text = p.read_text(encoding="utf-8").strip()
        assert len(text) > 50


def test_segmenter_find_cut_points():
    """Verify find_cut_points extracts valid, monotonically increasing cut points."""
    y, sr = sf.read("dataset/raw/clip01.wav", dtype="float32")
    cuts = find_cut_points(y, sr)
    assert len(cuts) >= 5
    for i in range(len(cuts) - 1):
        assert cuts[i] < cuts[i + 1]
    assert cuts[-1] <= len(y) / sr + 0.1


def test_segmenter_pick_segments():
    """Verify pick_segments returns non-overlapping segments between cut points."""
    cuts = [0.5, 3.5, 7.0, 11.5, 16.0, 20.5, 25.0, 29.5, 34.0, 38.5, 43.0]
    segments = pick_segments(cuts, n=4, min_len=2.5, max_len=6.0, min_gap=1.5, seed=7)
    assert len(segments) == 4
    for i in range(len(segments) - 1):
        cur_s, cur_e = segments[i]
        nxt_s, nxt_e = segments[i + 1]
        assert cur_s < cur_e
        assert cur_e <= nxt_s  # non-overlapping


def test_flaw_transformations():
    """Verify each flaw transformation produces valid audio without NaNs or Infs."""
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    # Time stretch fast & slow
    fast = time_stretch_audio(y, sr, 1.45)
    assert len(fast) < len(y)
    assert not np.isnan(fast).any()

    slow = time_stretch_audio(y, sr, 0.68)
    assert len(slow) > len(y)
    assert not np.isnan(slow).any()

    # Low volume
    low_v = apply_low_volume(y, sr, -10.0)
    assert len(low_v) == len(y)
    assert np.sqrt(np.mean(low_v**2)) < np.sqrt(np.mean(y**2))

    # Unclear articulation
    unclear = apply_unclear_articulation(y, sr, 2500.0, seed=7)
    assert len(unclear) == len(y)
    assert not np.isnan(unclear).any()

    # Monotone
    mono = apply_monotone(y, sr, 0.25)
    assert len(mono) == len(y)
    assert not np.isnan(mono).any()

    # Edge fades
    faded = apply_edge_fades(y, sr, 10.0)
    assert len(faded) == len(y)
    assert faded[0] == 0.0


def test_dataset_validation_passes():
    """Run full validate_dataset check and ensure it passes with 0 errors."""
    assert validate_dataset() is True


def test_labels_csv_and_pairs_json_structure():
    """Verify structure and column consistency of labels.csv and pairs.json."""
    labels_path = Path("dataset/labels.csv")
    pairs_path = Path("dataset/pairs.json")

    assert labels_path.exists()
    assert pairs_path.exists()

    with open(labels_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) >= 18
        for row in reader:
            assert row["pair_id"]
            assert row["file"]
            assert row["role"] in ("good", "bad")
            assert row["source"] in ("public", "self_recorded", "synthetic")

    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
        assert len(pairs) >= 18
        for p in pairs:
            assert "pair_id" in p
            assert "baseline" in p
            assert "participant" in p
            assert "transcript" in p
            assert "kind" in p
