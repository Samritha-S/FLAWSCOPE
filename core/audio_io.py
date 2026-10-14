"""Audio loading, resampling to mono 16 kHz, and content hashing."""

import hashlib
from pathlib import Path
import numpy as np
import librosa
import config


def load_audio(path: str | Path) -> tuple[np.ndarray, float]:
    """Load audio file as mono float32 ndarray resampled to config.SR.

    Args:
        path: Path to audio file (wav, mp3, m4a, flac).

    Returns:
        Tuple of (y: np.ndarray of float32, duration_s: float).

    Raises:
        ValueError: If audio is empty or exceeds config.MAX_AUDIO_S.
        FileNotFoundError: If the specified file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # librosa.load resamples to config.SR and converts to mono
    # timestamps must strictly match original without trimming or normalization
    y, sr = librosa.load(str(file_path), sr=config.SR, mono=True)
    y = y.astype(np.float32)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    duration_s = float(len(y) / config.SR)
    if duration_s > config.MAX_AUDIO_S:
        raise ValueError(
            f"Audio duration {duration_s:.2f}s exceeds MAX_AUDIO_S ({config.MAX_AUDIO_S}s)."
        )

    return y, duration_s


def file_sha1(path: str | Path) -> str:
    """Compute SHA1 hexadecimal digest of a file.

    Args:
        path: Path to file.

    Returns:
        40-character SHA1 hex digest string.
    """
    hasher = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
