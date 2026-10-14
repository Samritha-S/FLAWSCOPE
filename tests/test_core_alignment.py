"""Unit tests for core audio I/O and transcript alignment."""

import json
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

import config
from core.audio_io import file_sha1, load_audio
from core.alignment import (
    align,
    align_transcript,
    tokenize_transcript,
)


def test_tokenize_transcript_punctuation_and_hyphenation():
    """Verify hyphen splitting, inner apostrophe retention, and punct detection."""
    text = "The state-of-the-art don't fail, right? Yes; it's true: 'awesome'!"
    tokens = tokenize_transcript(text)
    norms = [t["norm"] for t in tokens]

    # Hyphenated words must be split into individual tokens
    assert "state" in norms
    assert "of" in norms
    assert "the" in norms
    assert "art" in norms

    # Inner apostrophes must be preserved
    assert "don't" in norms
    assert "it's" in norms

    # Outer quotes and surrounding punctuation must be stripped in norm
    assert "awesome" in norms

    # has_punct_after must be True for tokens ending with . , ; : ? !
    by_norm = {t["norm"]: t for t in tokens}
    assert by_norm["fail"]["has_punct_after"] is True
    assert by_norm["right"]["has_punct_after"] is True
    assert by_norm["yes"]["has_punct_after"] is True
    assert by_norm["true"]["has_punct_after"] is True
    assert by_norm["awesome"]["has_punct_after"] is True

    # Tokens without punctuation must be False
    assert by_norm["state"]["has_punct_after"] is False
    assert by_norm["of"]["has_punct_after"] is False
    assert by_norm["the"]["has_punct_after"] is False
    assert by_norm["art"]["has_punct_after"] is False
    assert by_norm["don't"]["has_punct_after"] is False


def test_align_perfect_match():
    """Verify 100% match when transcript matches ASR output."""
    transcript = "The quick brown fox jumps"
    tokens = tokenize_transcript(transcript)
    fake_asr = [
        {"word": "The", "norm": "the", "start": 0.10, "end": 0.30},
        {"word": "quick", "norm": "quick", "start": 0.35, "end": 0.65},
        {"word": "brown", "norm": "brown", "start": 0.70, "end": 1.00},
        {"word": "fox", "norm": "fox", "start": 1.05, "end": 1.35},
        {"word": "jumps", "norm": "jumps", "start": 1.40, "end": 1.70},
    ]

    words = align_transcript(tokens, fake_asr)
    assert len(words) == 5
    assert all(w["matched"] for w in words)
    assert words[0]["start"] == 0.10
    assert words[2]["norm"] == "brown"
    assert words[2]["start"] == 0.70
    assert words[2]["end"] == 1.00


def test_align_one_missing_word():
    """Verify missing word in ASR is linearly interpolated with matched=False."""
    transcript = "The quick brown fox jumps"
    tokens = tokenize_transcript(transcript)
    # "brown" is missing in ASR
    fake_asr = [
        {"word": "The", "norm": "the", "start": 0.10, "end": 0.30},
        {"word": "quick", "norm": "quick", "start": 0.35, "end": 0.65},
        {"word": "fox", "norm": "fox", "start": 1.25, "end": 1.55},
        {"word": "jumps", "norm": "jumps", "start": 1.60, "end": 1.90},
    ]

    words = align_transcript(tokens, fake_asr)
    assert len(words) == 5

    brown = words[2]
    assert brown["norm"] == "brown"
    assert brown["matched"] is False
    # Interpolated between quick.end (0.65) and fox.start (1.25)
    assert brown["start"] >= 0.65
    assert brown["end"] <= 1.25
    assert brown["end"] >= brown["start"] + 0.04

    # Other words must be matched
    assert words[0]["matched"] is True
    assert words[1]["matched"] is True
    assert words[3]["matched"] is True
    assert words[4]["matched"] is True


def test_align_one_substituted_word():
    """Verify equal-length substitution maps 1:1 with matched=True."""
    transcript = "The quick brown fox jumps"
    tokens = tokenize_transcript(transcript)
    # "brown" substituted by "bronze" in ASR
    fake_asr = [
        {"word": "The", "norm": "the", "start": 0.10, "end": 0.30},
        {"word": "quick", "norm": "quick", "start": 0.35, "end": 0.65},
        {"word": "bronze", "norm": "bronze", "start": 0.70, "end": 1.05},
        {"word": "fox", "norm": "fox", "start": 1.10, "end": 1.40},
        {"word": "jumps", "norm": "jumps", "start": 1.45, "end": 1.75},
    ]

    words = align_transcript(tokens, fake_asr)
    assert len(words) == 5
    brown = words[2]
    assert brown["norm"] == "brown"
    assert brown["matched"] is True
    assert brown["start"] == 0.70
    assert brown["end"] == 1.05


def test_align_inserted_filler():
    """Verify inserted fillers/disfluencies in ASR are ignored."""
    transcript = "The quick brown fox jumps"
    tokens = tokenize_transcript(transcript)
    # ASR has inserted "um" filler
    fake_asr = [
        {"word": "The", "norm": "the", "start": 0.10, "end": 0.30},
        {"word": "quick", "norm": "quick", "start": 0.35, "end": 0.65},
        {"word": "um", "norm": "um", "start": 0.70, "end": 0.90},
        {"word": "brown", "norm": "brown", "start": 0.95, "end": 1.25},
        {"word": "fox", "norm": "fox", "start": 1.30, "end": 1.60},
        {"word": "jumps", "norm": "jumps", "start": 1.65, "end": 1.95},
    ]

    words = align_transcript(tokens, fake_asr)
    assert len(words) == 5
    assert [w["norm"] for w in words] == ["the", "quick", "brown", "fox", "jumps"]
    assert all(w["matched"] for w in words)


def test_align_monotonic_times():
    """Verify post-processing enforces monotonic, non-overlapping times and min duration."""
    tokens = tokenize_transcript("one two three four")
    # Cramped and overlapping ASR words
    fake_asr = [
        {"word": "one", "norm": "one", "start": -0.10, "end": 0.01},  # negative start, tiny duration
        {"word": "two", "norm": "two", "start": 0.005, "end": 0.02},  # overlaps with one
        {"word": "three", "norm": "three", "start": 0.015, "end": 0.025},  # overlaps
        {"word": "four", "norm": "four", "start": 0.020, "end": 0.030},  # overlaps
    ]

    words = align_transcript(tokens, fake_asr)
    assert len(words) == 4

    # First start >= 0
    assert words[0]["start"] >= 0.0

    for i in range(len(words)):
        # Min duration 0.04s
        assert round(words[i]["end"] - words[i]["start"], 3) >= 0.040

        # Monotonic and non-overlapping
        if i > 0:
            assert words[i]["start"] >= words[i - 1]["end"]


def test_audio_io(tmp_path):
    """Test load_audio validation and file_sha1 computation."""
    dummy_wav = tmp_path / "test.wav"

    # Create valid 1s sine wave at 16 kHz
    sr = config.SR
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    y_in = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(dummy_wav), y_in, sr)

    # Test file_sha1
    sha = file_sha1(dummy_wav)
    assert len(sha) == 40
    assert isinstance(sha, str)

    # Test load_audio
    y, dur = load_audio(dummy_wav)
    assert y.dtype == np.float32
    assert pytest.approx(dur, rel=1e-2) == 1.0
    assert len(y) == sr

    # Test non-existent file
    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "non_existent.wav")

    # Test empty file
    empty_wav = tmp_path / "empty.wav"
    sf.write(str(empty_wav), np.array([], dtype=np.float32), sr)
    with pytest.raises(ValueError, match="empty"):
        load_audio(empty_wav)

    # Test file exceeding MAX_AUDIO_S
    long_wav = tmp_path / "too_long.wav"
    long_y = np.zeros(sr * (config.MAX_AUDIO_S + 2), dtype=np.float32)
    sf.write(str(long_wav), long_y, sr)
    with pytest.raises(ValueError, match="MAX_AUDIO_S"):
        load_audio(long_wav)


def test_align_caching_and_warning(tmp_path, monkeypatch):
    """Verify align() caching mechanism and warning emission for low match ratio."""
    fake_asr = [
        {"word": "apple", "norm": "apple", "start": 0.1, "end": 0.4},
        {"word": "banana", "norm": "banana", "start": 0.5, "end": 0.8},
    ]

    call_count = 0

    def mock_transcribe(y):
        nonlocal call_count
        call_count += 1
        return fake_asr

    monkeypatch.setattr("core.alignment.transcribe_words", mock_transcribe)

    y_dummy = np.zeros(16000, dtype=np.float32)
    transcript = "apple banana"
    test_sha1 = "test_sha1_12345"

    # Ensure cache dir under test
    cache_file = Path(".cache") / f"asr_{test_sha1}.json"
    if cache_file.exists():
        cache_file.unlink()

    # First call: invokes transcribe and saves cache
    words1, ratio1, warnings1 = align(y_dummy, transcript, sha1=test_sha1)
    assert call_count == 1
    assert ratio1 == 1.0
    assert len(warnings1) == 0
    assert cache_file.exists()

    # Second call with same sha1: reads from cache without invoking transcribe
    words2, ratio2, warnings2 = align(y_dummy, transcript, sha1=test_sha1)
    assert call_count == 1
    assert ratio2 == 1.0

    # Low match ratio test (< 70% matched)
    low_match_transcript = "apple banana cherry date elderberry fig grape"
    words3, ratio3, warnings3 = align(y_dummy, low_match_transcript, sha1=None)
    assert ratio3 < 0.7
    assert len(warnings3) == 1
    assert "Low alignment match ratio" in warnings3[0]

    # Clean up test cache
    if cache_file.exists():
        cache_file.unlink()
