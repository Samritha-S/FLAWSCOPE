"""Preprocess raw speech clips to 16 kHz mono WAV and verify audio quality."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def normalize_clip_name(stem: str) -> str:
    """Normalize clip name like 'clip_1' or 'clip1' to 'clip01'."""
    match = re.match(r"^clip[_-]?(\d+)$", stem, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        return f"clip{num:02d}"
    return stem


def load_audio_with_fallback(file_path: Path, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    """Load audio file as mono float32 at target_sr using librosa/soundfile, falling back to ffmpeg."""
    try:
        y, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        return y, sr
    except Exception as exc:
        print(f"Warning: Direct loader failed for {file_path.name} ({exc}), falling back to ffmpeg...", file=sys.stderr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(file_path),
            "-ar",
            str(target_sr),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(tmp_wav),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg decoding failed: {res.stderr}")
        y, sr = sf.read(str(tmp_wav), dtype="float32")
        return y, sr
    finally:
        if tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)


def trim_to_silence(y: np.ndarray, sr: int, max_sec: float) -> np.ndarray:
    """Trim audio at a silence boundary near max_sec if duration exceeds max_sec."""
    duration = len(y) / sr
    if duration <= max_sec:
        return y

    intervals = librosa.effects.split(y, top_db=35)
    if len(intervals) == 0:
        return y[: int(max_sec * sr)]

    # Silence regions are gaps between speech intervals
    cut_candidates: List[float] = []
    for i in range(len(intervals) - 1):
        end_cur = intervals[i][1]
        start_next = intervals[i + 1][0]
        if start_next > end_cur:
            midpoint_s = (end_cur + start_next) / (2.0 * sr)
            cut_candidates.append(midpoint_s)

    # Filter candidates <= max_sec
    valid_cuts = [c for c in cut_candidates if c <= max_sec]
    if valid_cuts:
        chosen_cut = valid_cuts[-1]
    else:
        # Fallback to the first candidate or exact max_sec
        chosen_cut = cut_candidates[0] if cut_candidates else max_sec

    samples_to_keep = int(chosen_cut * sr)
    return y[:samples_to_keep]


def prepare_raw(
    src_dir: Path = Path("dataset/raw_src"),
    dst_dir: Path = Path("dataset/raw"),
    max_sec: float = 60.0,
) -> List[Path]:
    """Convert all clips in src_dir to 16 kHz mono wav in dst_dir."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    supported_exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
    source_files = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in supported_exts])

    if not source_files:
        print(f"No source audio files found in {src_dir}", file=sys.stderr)
        return []

    created_files: List[Path] = []
    print(f"Processing {len(source_files)} source files from {src_dir} to {dst_dir}...")

    for src_file in source_files:
        out_stem = normalize_clip_name(src_file.stem)
        dst_file = dst_dir / f"{out_stem}.wav"

        y, sr = load_audio_with_fallback(src_file, target_sr=TARGET_SR)
        if max_sec > 0:
            y = trim_to_silence(y, sr, max_sec)

        sf.write(str(dst_file), y, sr, subtype="PCM_16")
        dur = len(y) / sr
        print(f"  -> Converted {src_file.name} to {dst_file.name} ({dur:.2f}s, 16kHz mono)")
        created_files.append(dst_file)

    return created_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw speech clips to 16 kHz mono WAV.")
    parser.add_argument("--src-dir", type=Path, default=Path("dataset/raw_src"), help="Source audio directory")
    parser.add_argument("--dst-dir", type=Path, default=Path("dataset/raw"), help="Target WAV directory")
    parser.add_argument("--max-sec", type=float, default=60.0, help="Maximum duration in seconds to trim at silence")
    args = parser.parse_args()

    prepare_raw(src_dir=args.src_dir, dst_dir=args.dst_dir, max_sec=args.max_sec)


if __name__ == "__main__":
    main()
