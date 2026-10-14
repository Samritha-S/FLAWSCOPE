"""Caching mechanism for analysis results and extracted frame features."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np

CACHE_DIR = Path(".cache")


def get_cache_dir() -> Path:
    """Ensure and return the root .cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def compute_result_hash(
    sha1_b: str,
    sha1_p: str,
    transcript: str,
    config_snapshot: dict[str, Any],
) -> str:
    """Compute deterministic SHA1 hash for an analysis run.

    Hash = sha1(baseline sha1 + participant sha1 + transcript text + json of config snapshot).

    Args:
        sha1_b: SHA1 digest of baseline audio file.
        sha1_p: SHA1 digest of participant audio file.
        transcript: Reference transcript string.
        config_snapshot: Serialized dictionary of active configuration.

    Returns:
        40-character hexadecimal SHA1 digest string.
    """
    hasher = hashlib.sha1()
    hasher.update(sha1_b.encode("utf-8"))
    hasher.update(sha1_p.encode("utf-8"))
    hasher.update(transcript.encode("utf-8"))
    hasher.update(json.dumps(config_snapshot, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def get_cached_result(result_hash: str) -> dict[str, Any] | None:
    """Load cached analysis result dictionary if present.

    Args:
        result_hash: 40-character SHA1 run hash.

    Returns:
        Cached result dictionary or None if not cached or corrupt.
    """
    cache_file = get_cache_dir() / f"result_{result_hash}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def set_cached_result(result_hash: str, result_dict: dict[str, Any]) -> None:
    """Save analysis result dictionary to cache.

    Args:
        result_hash: 40-character SHA1 run hash.
        result_dict: Complete analysis result matching AnalysisResult schema.
    """
    cache_file = get_cache_dir() / f"result_{result_hash}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
    except Exception:
        pass


def get_cached_features(file_sha1: str) -> dict[str, np.ndarray] | None:
    """Load cached frame features from .npz if present.

    Args:
        file_sha1: SHA1 digest of audio file.

    Returns:
        Dict of numpy arrays: {'f0_st', 'rms_db_rel', 'mfcc', 'logmel'} or None.
    """
    cache_file = get_cache_dir() / f"features_{file_sha1}.npz"
    if cache_file.exists():
        try:
            with np.load(cache_file) as data:
                return {
                    "f0_st": data["f0_st"],
                    "rms_db_rel": data["rms_db_rel"],
                    "mfcc": data["mfcc"],
                    "logmel": data["logmel"],
                }
        except Exception:
            return None
    return None


def set_cached_features(file_sha1: str, features: dict[str, np.ndarray]) -> None:
    """Save frame features to compressed .npz cache.

    Args:
        file_sha1: SHA1 digest of audio file.
        features: Dict of numpy arrays with keys: f0_st, rms_db_rel, mfcc, logmel.
    """
    cache_file = get_cache_dir() / f"features_{file_sha1}.npz"
    try:
        np.savez_compressed(
            cache_file,
            f0_st=features["f0_st"],
            rms_db_rel=features["rms_db_rel"],
            mfcc=features["mfcc"],
            logmel=features["logmel"],
        )
    except Exception:
        pass
