"""Adapter module transforming contract analysis results and dataset entries into UI structures."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import soundfile as sf
import streamlit as st


def _compute_file_sha1(file_path: str) -> str:
    """Compute SHA1 hash of a file for cache keys."""
    p = Path(file_path)
    if not p.exists():
        return ""
    hasher = hashlib.sha1()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@st.cache_data(show_spinner=False)
def _load_mock_json() -> Dict[str, Any]:
    """Load and cache mock_result.json."""
    mock_file = PROJECT_ROOT / "contracts" / "mock_result.json"
    if not mock_file.exists():
        raise FileNotFoundError(f"Mock result file not found at {mock_file}")
    with open(mock_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_benchmark_mock(raw_mock: Dict[str, Any], transcript: str = "") -> Dict[str, Any]:
    """Adapt raw mock result to benchmark mode contract."""
    res = copy.deepcopy(raw_mock)
    if "meta" not in res:
        res["meta"] = {}
    res["meta"]["mode"] = "benchmark"

    # Auto-transcript
    auto_tx = transcript or " ".join(w.get("text", "") for w in res.get("words", []))
    res["meta"]["transcript_text"] = auto_tx

    # Typical ranges for strong speakers
    res["bands"] = {
        "pitch_variation": [1.4, 3.2],
        "f0_std_roll_st": [1.4, 3.2],
        "loudness": [-4.0, 3.0],
        "rms_db_rel": [-4.0, 3.0],
        "speech_rate": [3.2, 5.2],
        "rate_sps": [3.2, 5.2],
    }

    # Ensure curves.p.f0_std_roll_st exists
    curves = res.setdefault("curves", {})
    p_curves = curves.setdefault("p", {})
    f0_st = p_curves.get("f0_st", [])
    if "f0_std_roll_st" not in p_curves or not p_curves["f0_std_roll_st"]:
        roll: List[Optional[float]] = []
        win = 25
        for i in range(len(f0_st)):
            sub = [v for v in f0_st[max(0, i - win // 2) : min(len(f0_st), i + win // 2 + 1)] if v is not None]
            if len(sub) >= 5:
                mean = sum(sub) / len(sub)
                var = sum((x - mean) ** 2 for x in sub) / len(sub)
                roll.append(round(var ** 0.5, 2))
            else:
                roll.append(round(1.8 + 0.4 * np.sin(i * 0.05), 2) if i < len(f0_st) else None)
        p_curves["f0_std_roll_st"] = roll

    # Populate reference_range on flaws
    default_ranges = {
        "MONOTONE": [1.4, 3.2],
        "TOO_FAST": [3.2, 5.2],
        "TOO_SLOW": [3.2, 5.2],
        "LOW_VOLUME": [-4.0, 3.0],
        "LONG_PAUSE": [0.1, 0.9],
        "UNCLEAR_ARTICULATION": [0.0, 1.8],
    }
    for f in res.get("flaws", []):
        ft = f.get("type", "")
        if "reference_range" not in f or f["reference_range"] is None:
            f["reference_range"] = default_ranges.get(ft, [1.0, 3.0])

    return res


@st.cache_data(show_spinner=False)
def _cached_analyze(
    b_sha1: str,
    p_sha1: str,
    baseline_path: str,
    participant_path: str,
    transcript_text: str,
    overrides_json: str,
) -> Dict[str, Any]:
    """Cached runner for core.pipeline.analyze."""
    from core.pipeline import analyze
    overrides = json.loads(overrides_json) if overrides_json else None
    return analyze(
        baseline_audio=baseline_path,
        participant_audio=participant_path,
        transcript=transcript_text,
        config_overrides=overrides,
    )


def get_result(
    baseline_path: Optional[str],
    participant_path: str,
    transcript_text: str = "",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieve analysis result either from mock data or by executing signal pipeline."""
    is_benchmark = not baseline_path or os.environ.get("FLAWSCOPE_MOCK_MODE") == "benchmark"

    if os.environ.get("FLAWSCOPE_MOCK") == "1" or os.environ.get("FLAWSCOPE_MOCK_MODE") == "benchmark":
        raw = _load_mock_json()
        if is_benchmark:
            return _make_benchmark_mock(raw, transcript_text)
        else:
            res = copy.deepcopy(raw)
            res.setdefault("meta", {})["mode"] = "reference"
            return res

    # If baseline is not provided, evaluate in benchmark mode
    if not baseline_path:
        try:
            from core.pipeline import analyze
            res = analyze(
                baseline_audio=None,  # type: ignore[arg-type]
                participant_audio=participant_path,
                transcript=transcript_text or None,  # type: ignore[arg-type]
                config_overrides=overrides,
            )
            if isinstance(res, dict):
                res.setdefault("meta", {})["mode"] = "benchmark"
                if "bands" not in res:
                    res["bands"] = {
                        "pitch_variation": [1.4, 3.2],
                        "f0_std_roll_st": [1.4, 3.2],
                        "loudness": [-4.0, 3.0],
                        "rms_db_rel": [-4.0, 3.0],
                        "speech_rate": [3.2, 5.2],
                        "rate_sps": [3.2, 5.2],
                    }
                return res
        except Exception:
            raw = _load_mock_json()
            return _make_benchmark_mock(raw, transcript_text)

    # Reference mode with real audio files
    b_sha1 = _compute_file_sha1(baseline_path)
    p_sha1 = _compute_file_sha1(participant_path)
    overrides_json = json.dumps(overrides, sort_keys=True) if overrides else ""

    res = _cached_analyze(
        b_sha1=b_sha1,
        p_sha1=p_sha1,
        baseline_path=baseline_path,
        participant_path=participant_path,
        transcript_text=transcript_text,
        overrides_json=overrides_json,
    )
    if isinstance(res, dict):
        res.setdefault("meta", {})["mode"] = "reference"
    return res


def list_demo_pairs() -> List[Dict[str, Any]]:
    """Read dataset/pairs.json returning list of demo pair dictionaries."""
    pairs_file = PROJECT_ROOT / "dataset" / "pairs.json"
    if not pairs_file.exists() or pairs_file.stat().st_size == 0:
        return []
    try:
        with open(pairs_file, "r", encoding="utf-8") as f:
            pairs = json.load(f)
            return pairs if isinstance(pairs, list) else []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_audio_for_ui(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file as 16 kHz mono float32 for UI display and playback slices."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / path
    if not p.exists():
        return np.zeros(0, dtype=np.float32), 16000

    try:
        y, sr = sf.read(str(p), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
        return y, sr
    except Exception:
        y, sr = librosa.load(str(p), sr=16000, mono=True)
        return y, sr
