"""Adapter module transforming contract analysis results and dataset entries into UI structures."""

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
    baseline_path: str,
    participant_path: str,
    transcript_text: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieve analysis result either from mock data or by executing signal pipeline."""
    if os.environ.get("FLAWSCOPE_MOCK") == "1":
        return _load_mock_json()

    b_sha1 = _compute_file_sha1(baseline_path)
    p_sha1 = _compute_file_sha1(participant_path)
    overrides_json = json.dumps(overrides, sort_keys=True) if overrides else ""

    return _cached_analyze(
        b_sha1=b_sha1,
        p_sha1=p_sha1,
        baseline_path=baseline_path,
        participant_path=participant_path,
        transcript_text=transcript_text,
        overrides_json=overrides_json,
    )


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
