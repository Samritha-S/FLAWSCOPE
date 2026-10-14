"""Inject reproducible, calibrated flaws into clean baseline speech."""

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf

from dataset.segmenter import find_cut_points, pick_segments

TARGET_SR = 16000

# Severity recipes: [Level 1, Level 2, Level 3, Level 4]
FLAW_RECIPES: Dict[str, List[Any]] = {
    "TOO_FAST": [1.10, 1.25, 1.45, 1.70],
    "TOO_SLOW": [0.92, 0.80, 0.68, 0.55],
    "LONG_PAUSE": [0.6, 1.0, 1.6, 2.4],
    "LOW_VOLUME": [-3.0, -6.0, -10.0, -15.0],
    "MONOTONE": [0.8, 0.5, 0.25, 0.0],
    "UNCLEAR_ARTICULATION": [5000.0, 3500.0, 2500.0, 1500.0],
}

FLAW_ORDER = [
    "MONOTONE",
    "TOO_FAST",
    "TOO_SLOW",
    "LONG_PAUSE",
    "LOW_VOLUME",
    "UNCLEAR_ARTICULATION",
]


def get_ffmpeg_binary() -> str:
    """Locate ffmpeg binary on system or WinGet packages."""
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        winget_path = Path(local_app_data) / "Microsoft/WinGet/Packages"
        matches = list(winget_path.glob("**/ffmpeg.exe"))
        if matches:
            return str(matches[0])
    return "ffmpeg"


def time_stretch_audio(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch audio using ffmpeg atempo or phase vocoder fallback."""
    if abs(rate - 1.0) < 1e-4:
        return y.copy()

    ffmpeg_bin = get_ffmpeg_binary()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        in_p = f_in.name
        out_p = f_out.name

    try:
        sf.write(in_p, y, sr, subtype="PCM_16")
        cmd = [ffmpeg_bin, "-y", "-i", in_p, "-filter:a", f"atempo={rate:.4f}", out_p]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode == 0 and os.path.exists(out_p):
            stretched, _ = sf.read(out_p, dtype="float32")
            return stretched
    except Exception:
        pass
    finally:
        if os.path.exists(in_p):
            os.remove(in_p)
        if os.path.exists(out_p):
            os.remove(out_p)

    # Pure NumPy Phase Vocoder fallback
    n_fft = 2048
    hop_out = 512
    hop_in = int(round(hop_out * rate))
    window = np.hanning(n_fft)

    n_frames = (len(y) - n_fft) // hop_in + 1
    if n_frames <= 0:
        return y.copy()

    frames = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        chunk = y[i * hop_in : i * hop_in + n_fft] * window
        frames[:, i] = np.fft.rfft(chunk)

    phase_adv = np.linspace(0, np.pi * hop_in, n_fft // 2 + 1)
    phase = np.angle(frames[:, 0])
    out_frames = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    out_frames[:, 0] = frames[:, 0]

    for i in range(1, n_frames):
        mag = np.abs(frames[:, i])
        dp = np.angle(frames[:, i]) - np.angle(frames[:, i - 1]) - phase_adv
        dp = dp - 2 * np.pi * np.round(dp / (2 * np.pi))
        phase += phase_adv * (hop_out / hop_in) + dp
        out_frames[:, i] = mag * np.exp(1j * phase)

    out_len = n_fft + (n_frames - 1) * hop_out
    out = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_out
        chunk = np.fft.irfft(out_frames[:, i], n=n_fft) * window
        out[start : start + n_fft] += chunk
        win_sum[start : start + n_fft] += window**2

    mask = win_sum > 1e-4
    out[mask] /= win_sum[mask]
    return out


def apply_low_volume(y: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    """Apply gain drop with 50 ms linear fades at boundaries."""
    gain_lin = 10.0 ** (gain_db / 20.0)
    fade_len = min(int(0.05 * sr), len(y) // 2)
    env = np.ones(len(y), dtype=np.float32) * gain_lin
    if fade_len > 0:
        env[:fade_len] = np.linspace(1.0, gain_lin, fade_len)
        env[-fade_len:] = np.linspace(gain_lin, 1.0, fade_len)
    return (y * env).astype(np.float32)


def apply_unclear_articulation(y: np.ndarray, sr: int, cutoff_hz: float, seed: int = 7) -> np.ndarray:
    """Apply Butterworth low-pass filter plus faint Gaussian noise at 30 dB SNR."""
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    gain = 1.0 / np.sqrt(1.0 + (freqs / cutoff_hz) ** 8)
    filtered = np.fft.irfft(np.fft.rfft(y) * gain, n=n).astype(np.float32)

    sig_rms = float(np.sqrt(np.mean(filtered**2))) + 1e-9
    noise_rms = sig_rms / (10.0 ** (30.0 / 20.0))

    rng = np.random.RandomState(seed)
    noise = rng.randn(n).astype(np.float32)
    current_noise_rms = float(np.sqrt(np.mean(noise**2))) + 1e-9
    scaled_noise = noise * (noise_rms / current_noise_rms)

    return np.clip(filtered + scaled_noise, -1.0, 1.0)


def apply_monotone(y: np.ndarray, sr: int, fraction: float) -> np.ndarray:
    """Flatten pitch contour towards piece median in log domain via Parselmouth."""
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    try:
        manipulation = call(sound, "To Manipulation", 0.01, 75.0, 500.0)
        pitch_tier = call(manipulation, "Extract pitch tier")
        num_points = call(pitch_tier, "Get number of points")
        if num_points <= 0:
            return y.copy()

        times = [call(pitch_tier, "Get time from index", i + 1) for i in range(num_points)]
        freqs = [call(pitch_tier, "Get value at index", i + 1) for i in range(num_points)]

        log_f = np.log(freqs)
        med = float(np.median(log_f))
        new_f = np.exp(med + fraction * (log_f - med))

        new_tier = call("Create PitchTier", "monotone", 0.0, sound.duration)
        for t_val, f_val in zip(times, new_f):
            call(new_tier, "Add point", t_val, f_val)

        call([manipulation, new_tier], "Replace pitch tier")
        resynth = call(manipulation, "Get resynthesis (overlap-add)")
        out_y = resynth.values[0]
        if len(out_y) > len(y):
            out_y = out_y[: len(y)]
        elif len(out_y) < len(y):
            out_y = np.pad(out_y, (0, len(y) - len(out_y)))
        return out_y.astype(np.float32)
    except Exception:
        return y.copy()


def apply_edge_fades(piece: np.ndarray, sr: int = TARGET_SR, fade_ms: float = 10.0) -> np.ndarray:
    """Apply 10 ms linear fade-in and fade-out at piece boundaries."""
    fade_len = min(int((fade_ms / 1000.0) * sr), len(piece) // 2)
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        piece = piece.copy()
        piece[:fade_len] *= fade_in
        piece[-fade_len:] *= fade_out
    return piece


def transform_segment(
    piece: np.ndarray,
    sr: int,
    flaw_type: str,
    level: int,
    seed: int,
) -> Tuple[np.ndarray, str]:
    """Transform an audio segment according to flaw type and severity level."""
    idx = level - 1
    val = FLAW_RECIPES[flaw_type][idx]

    if flaw_type == "TOO_FAST":
        res = time_stretch_audio(piece, sr, rate=float(val))
        param = f"rate={val:.2f}"
    elif flaw_type == "TOO_SLOW":
        res = time_stretch_audio(piece, sr, rate=float(val))
        param = f"rate={val:.2f}"
    elif flaw_type == "LOW_VOLUME":
        res = apply_low_volume(piece, sr, gain_db=float(val))
        param = f"gain_db={val:.1f}"
    elif flaw_type == "MONOTONE":
        res = apply_monotone(piece, sr, fraction=float(val))
        param = f"fraction={val:.2f}"
    elif flaw_type == "UNCLEAR_ARTICULATION":
        res = apply_unclear_articulation(piece, sr, cutoff_hz=float(val), seed=seed)
        param = f"cutoff_hz={int(val)}"
    else:
        res = piece.copy()
        param = ""

    return res, param


def generate_flawed_audio(
    y: np.ndarray,
    sr: int,
    flaw_specs: List[Dict[str, Any]],
    seed: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Assemble output audio by splitting, transforming flaw pieces, and concatenating with 10ms fades.

    flaw_specs: list of {start_cut, end_cut, flaw_type, level}
    """
    duration_s = len(y) / sr
    # Sort specs by start_cut
    flaw_specs = sorted(flaw_specs, key=lambda s: s["start_cut"])

    # Create timeline boundaries
    boundaries: List[float] = [0.0]
    for spec in flaw_specs:
        boundaries.append(spec["start_cut"])
        boundaries.append(spec["end_cut"])
    boundaries.append(duration_s)
    boundaries = sorted(list(dict.fromkeys(boundaries)))

    output_pieces: List[np.ndarray] = []
    recorded_labels: List[Dict[str, Any]] = []
    cum_s = 0.0

    spec_idx = 0
    num_specs = len(flaw_specs)

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]
        if t_end <= t_start:
            continue

        start_samp = int(round(t_start * sr))
        end_samp = int(round(t_end * sr))
        piece = y[start_samp:end_samp].astype(np.float32)

        # Check if this slice matches the current flaw spec
        is_flawed = False
        curr_spec = None
        if spec_idx < num_specs:
            s_spec = flaw_specs[spec_idx]
            if abs(t_start - s_spec["start_cut"]) < 1e-3 and abs(t_end - s_spec["end_cut"]) < 1e-3:
                is_flawed = True
                curr_spec = s_spec
                spec_idx += 1

        if is_flawed and curr_spec:
            flaw_type = curr_spec["flaw_type"]
            level = curr_spec["level"]

            if flaw_type == "LONG_PAUSE":
                pause_s = float(FLAW_RECIPES["LONG_PAUSE"][level - 1])
                silence_piece = np.zeros(int(round(pause_s * sr)), dtype=np.float32)

                flaw_start = round(cum_s, 3)
                flaw_end = round(cum_s + pause_s, 3)
                recorded_labels.append({
                    "flaw_type": flaw_type,
                    "start_s": flaw_start,
                    "end_s": flaw_end,
                    "severity_level": level,
                    "param": f"pause_s={pause_s:.1f}",
                    "notes": f"Injected LONG_PAUSE level {level}",
                })

                output_pieces.append(apply_edge_fades(silence_piece, sr=sr, fade_ms=10.0))
                cum_s += pause_s

                # Normal speech piece follows
                clean_piece = apply_edge_fades(piece, sr=sr, fade_ms=10.0)
                output_pieces.append(clean_piece)
                cum_s += len(clean_piece) / sr
            else:
                transformed_piece, param = transform_segment(piece, sr, flaw_type, level, seed)
                transformed_piece = apply_edge_fades(transformed_piece, sr=sr, fade_ms=10.0)

                flaw_start = round(cum_s, 3)
                dur = len(transformed_piece) / sr
                flaw_end = round(cum_s + dur, 3)

                recorded_labels.append({
                    "flaw_type": flaw_type,
                    "start_s": flaw_start,
                    "end_s": flaw_end,
                    "severity_level": level,
                    "param": param,
                    "notes": f"Injected {flaw_type} level {level}",
                })

                output_pieces.append(transformed_piece)
                cum_s += dur
        else:
            clean_piece = apply_edge_fades(piece, sr=sr, fade_ms=10.0)
            output_pieces.append(clean_piece)
            cum_s += len(clean_piece) / sr

    if output_pieces:
        full_audio = np.concatenate(output_pieces, axis=0)
    else:
        full_audio = y.copy()

    return full_audio, recorded_labels


def process_clip(
    clip_id: str,
    raw_dir: Path,
    out_dir: Path,
    seed: int,
    is_clip01: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process a single clip, generating mixed files and single files (for clip01)."""
    wav_path = raw_dir / f"{clip_id}.wav"
    transcript_path = Path("dataset/transcripts") / f"{clip_id}.txt"
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing audio: {wav_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing transcript: {transcript_path}")

    y, sr = sf.read(str(wav_path), dtype="float32")
    cut_points = find_cut_points(y, sr)

    labels_rows: List[Dict[str, Any]] = []
    pairs_entries: List[Dict[str, Any]] = []

    # Cycle order for mixed flaws
    rng = random.Random(seed)
    shuffled_flaws = list(FLAW_ORDER)
    rng.shuffle(shuffled_flaws)

    # (a) Four MIXED files: L1 (1 flaw), L2 (2 flaws), L3 (4 flaws), L4 (6 flaws)
    level_flaw_counts = [(1, 1), (2, 2), (3, 4), (4, 6)]
    for level, count in level_flaw_counts:
        pair_id = f"{clip_id}_L{level}_mixed"
        out_filename = f"{pair_id}.wav"
        out_path = out_dir / out_filename

        # Pick count segments
        segments = pick_segments(cut_points, n=count, min_len=2.5, max_len=6.0, min_gap=1.5, seed=seed + level)
        flaw_specs: List[Dict[str, Any]] = []
        for i, (seg_s, seg_e) in enumerate(segments):
            f_type = shuffled_flaws[i % len(shuffled_flaws)]
            flaw_specs.append({
                "start_cut": seg_s,
                "end_cut": seg_e,
                "flaw_type": f_type,
                "level": level,
            })

        out_audio, labels = generate_flawed_audio(y, sr, flaw_specs, seed=seed + level)
        sf.write(str(out_path), out_audio, sr, subtype="PCM_16")

        for lbl in labels:
            labels_rows.append({
                "pair_id": pair_id,
                "file": f"dataset/generated/{out_filename}",
                "role": "bad",
                "source": "synthetic",
                "baseline_file": f"dataset/raw/{clip_id}.wav",
                "transcript_file": f"dataset/transcripts/{clip_id}.txt",
                "flaw_type": lbl["flaw_type"],
                "start_s": f"{lbl['start_s']:.3f}",
                "end_s": f"{lbl['end_s']:.3f}",
                "severity_level": lbl["severity_level"],
                "param": lbl["param"],
                "notes": lbl["notes"],
            })

        pairs_entries.append({
            "pair_id": pair_id,
            "baseline": f"dataset/raw/{clip_id}.wav",
            "participant": f"dataset/generated/{out_filename}",
            "transcript": f"dataset/transcripts/{clip_id}.txt",
            "level": level,
            "kind": "mixed",
            "flaw_type": None,
        })

    # (b) Six SINGLE-type level-3 files for clip01
    if is_clip01:
        for f_type in FLAW_ORDER:
            pair_id = f"{clip_id}_L3_{f_type}"
            out_filename = f"{pair_id}.wav"
            out_path = out_dir / out_filename

            segments = pick_segments(cut_points, n=1, min_len=2.5, max_len=6.0, min_gap=1.5, seed=seed)
            if not segments:
                segments = [(cut_points[0], cut_points[1])]
            seg_s, seg_e = segments[0]

            flaw_specs = [{
                "start_cut": seg_s,
                "end_cut": seg_e,
                "flaw_type": f_type,
                "level": 3,
            }]

            out_audio, labels = generate_flawed_audio(y, sr, flaw_specs, seed=seed)
            sf.write(str(out_path), out_audio, sr, subtype="PCM_16")

            for lbl in labels:
                labels_rows.append({
                    "pair_id": pair_id,
                    "file": f"dataset/generated/{out_filename}",
                    "role": "bad",
                    "source": "synthetic",
                    "baseline_file": f"dataset/raw/{clip_id}.wav",
                    "transcript_file": f"dataset/transcripts/{clip_id}.txt",
                    "flaw_type": lbl["flaw_type"],
                    "start_s": f"{lbl['start_s']:.3f}",
                    "end_s": f"{lbl['end_s']:.3f}",
                    "severity_level": lbl["severity_level"],
                    "param": lbl["param"],
                    "notes": lbl["notes"],
                })

            pairs_entries.append({
                "pair_id": pair_id,
                "baseline": f"dataset/raw/{clip_id}.wav",
                "participant": f"dataset/generated/{out_filename}",
                "transcript": f"dataset/transcripts/{clip_id}.txt",
                "level": 3,
                "kind": "single",
                "flaw_type": f_type,
            })

    return labels_rows, pairs_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject reproducible flaws into baseline audio.")
    parser.add_argument("--clip", type=str, default=None, help="Clip ID to process (e.g. clip01)")
    parser.add_argument("--all", action="store_true", help="Process all baseline clips in dataset/raw")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--raw-dir", type=Path, default=Path("dataset/raw"), help="Directory with baseline audio")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset/generated"), help="Output directory")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        clips = sorted([p.stem for p in args.raw_dir.glob("clip*.wav") if not p.stem.endswith("_mixed")])
    elif args.clip:
        clips = [args.clip]
    else:
        print("Please specify --clip <clip_id> or --all.")
        return

    print(f"Injecting flaws for clips: {clips} (seed={args.seed})...")

    all_labels_rows: List[Dict[str, Any]] = []
    all_pairs_entries: List[Dict[str, Any]] = []

    # Also record baseline files themselves
    for clip_id in clips:
        baseline_file = f"dataset/raw/{clip_id}.wav"
        transcript_file = f"dataset/transcripts/{clip_id}.txt"
        all_labels_rows.append({
            "pair_id": f"{clip_id}_good",
            "file": baseline_file,
            "role": "good",
            "source": "public",
            "baseline_file": baseline_file,
            "transcript_file": transcript_file,
            "flaw_type": "NONE",
            "start_s": "0.000",
            "end_s": "0.000",
            "severity_level": "0",
            "param": "",
            "notes": "Good baseline recording",
        })

    for clip_id in clips:
        is_clip01 = (clip_id == "clip01")
        l_rows, p_entries = process_clip(
            clip_id=clip_id,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            seed=args.seed,
            is_clip01=is_clip01,
        )
        all_labels_rows.extend(l_rows)
        all_pairs_entries.extend(p_entries)

    # Write dataset/labels.csv
    labels_csv_path = Path("dataset/labels.csv")
    csv_columns = [
        "pair_id",
        "file",
        "role",
        "source",
        "baseline_file",
        "transcript_file",
        "flaw_type",
        "start_s",
        "end_s",
        "severity_level",
        "param",
        "notes",
    ]

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, lineterminator="\n")
        writer.writeheader()
        for row in all_labels_rows:
            writer.writerow(row)

    print(f"Wrote {len(all_labels_rows)} rows to {labels_csv_path}")

    # Write dataset/pairs.json
    pairs_json_path = Path("dataset/pairs.json")
    with open(pairs_json_path, "w", encoding="utf-8") as f:
        json.dump(all_pairs_entries, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(all_pairs_entries)} entries to {pairs_json_path}")


if __name__ == "__main__":
    main()
