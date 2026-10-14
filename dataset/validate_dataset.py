"""Validate dataset files, audio formatting, label intervals, and schema compliance."""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import soundfile as sf

TARGET_SR = 16000


def validate_dataset(
    labels_csv_path: Path = Path("dataset/labels.csv"),
    pairs_json_path: Path = Path("dataset/pairs.json"),
) -> bool:
    """Run all dataset integrity validations and print summary table."""
    errors: List[str] = []

    if not labels_csv_path.exists():
        print(f"Error: {labels_csv_path} does not exist", file=sys.stderr)
        return False
    if not pairs_json_path.exists():
        print(f"Error: {pairs_json_path} does not exist", file=sys.stderr)
        return False

    with open(labels_csv_path, "r", encoding="utf-8") as f:
        labels = list(csv.DictReader(f))

    with open(pairs_json_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # Cache audio durations and specs
    audio_info: Dict[str, Tuple[float, int, int]] = {}

    def check_audio(rel_path: str) -> None:
        if not rel_path or rel_path in audio_info:
            return
        p = Path(rel_path)
        if not p.exists():
            errors.append(f"Audio file missing: {rel_path}")
            return
        try:
            info = sf.info(str(p))
            if info.samplerate != TARGET_SR:
                errors.append(f"Wrong sample rate ({info.samplerate} != {TARGET_SR}): {rel_path}")
            if info.channels != 1:
                errors.append(f"Not mono ({info.channels} channels): {rel_path}")
            audio_info[rel_path] = (info.duration, info.samplerate, info.channels)
        except Exception as exc:
            errors.append(f"Could not read audio {rel_path}: {exc}")

    # Validate file existence in pairs.json
    for p in pairs:
        pair_id = p.get("pair_id", "")
        baseline = p.get("baseline", "")
        participant = p.get("participant", "")
        transcript = p.get("transcript", "")

        check_audio(baseline)
        check_audio(participant)

        if not Path(transcript).exists():
            errors.append(f"Transcript missing for {pair_id}: {transcript}")

    # Group labels by file
    labels_by_file: Dict[str, List[Dict[str, Any]]] = {}
    bad_files: set[str] = set()

    for row in labels:
        f_path = row["file"]
        role = row["role"]
        if role == "bad":
            bad_files.add(f_path)
        labels_by_file.setdefault(f_path, []).append(row)

        check_audio(f_path)
        check_audio(row.get("baseline_file", ""))
        tr_file = row.get("transcript_file", "")
        if tr_file and not Path(tr_file).exists():
            errors.append(f"Transcript missing: {tr_file}")

        # Check start_s and end_s
        if role == "bad":
            try:
                start_s = float(row["start_s"])
                end_s = float(row["end_s"])
            except ValueError:
                errors.append(f"Invalid timestamp in {f_path}: start={row.get('start_s')}, end={row.get('end_s')}")
                continue

            if start_s < 0.0 or end_s <= start_s:
                errors.append(f"Invalid label interval [{start_s}, {end_s}] in {f_path}")

            if f_path in audio_info:
                dur = audio_info[f_path][0]
                if end_s > dur + 0.1:  # 100ms tolerance for sample rounding
                    errors.append(f"Label interval [{start_s}, {end_s}] exceeds file duration ({dur:.2f}s) in {f_path}")

    # Check that every non-good file has at least 1 label
    for bf in bad_files:
        lbls = labels_by_file.get(bf, [])
        if not lbls:
            errors.append(f"Non-good file {bf} has no labels in labels.csv")

    # Check no two labels of the same type overlap
    for f_path, file_labels in labels_by_file.items():
        by_type: Dict[str, List[Tuple[float, float]]] = {}
        for row in file_labels:
            if row["role"] != "bad":
                continue
            ft = row["flaw_type"]
            by_type.setdefault(ft, []).append((float(row["start_s"]), float(row["end_s"])))

        for ft, intervals in by_type.items():
            sorted_ints = sorted(intervals, key=lambda x: x[0])
            for i in range(len(sorted_ints) - 1):
                cur_end = sorted_ints[i][1]
                nxt_start = sorted_ints[i + 1][0]
                if cur_end > nxt_start + 1e-3:
                    errors.append(
                        f"Overlapping labels of type {ft} in {f_path}: "
                        f"[{sorted_ints[i][0]:.3f}, {cur_end:.3f}] and [{nxt_start:.3f}, {sorted_ints[i+1][1]:.3f}]"
                    )

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Clip / Pair ID':<32} {'Level':<8} {'N Labels':<10} {'Duration (s)':<12}")
    print("-" * 70)

    # Summary by pair in pairs.json
    for p in pairs:
        pair_id = p["pair_id"]
        level = str(p.get("level", "-"))
        part_file = p["participant"]
        n_lbls = len([r for r in labels if r.get("pair_id") == pair_id and r["role"] == "bad"])
        dur_str = f"{audio_info.get(part_file, (0.0,))[0]:.2f}"
        print(f"{pair_id:<32} {level:<8} {n_lbls:<10} {dur_str:<12}")

    print("=" * 70 + "\n")

    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    print("Dataset validation PASSED! All files exist, audio specs match, and intervals are strictly valid.")
    return True


def main() -> None:
    success = validate_dataset()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
