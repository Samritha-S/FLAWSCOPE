"""CLI tool to append manual label rows to labels.csv and pairs.json."""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_COLUMNS = [
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


def add_label(
    pair_id: str,
    file_path: str,
    role: str = "bad",
    source: str = "self_recorded",
    baseline_file: str = "",
    transcript_file: str = "",
    flaw_type: str = "UNCLEAR_ARTICULATION",
    start_s: float = 0.0,
    end_s: float = 0.0,
    severity_level: int = 3,
    param: str = "",
    notes: str = "",
    level: int = 3,
    kind: str = "self",
) -> None:
    labels_csv_path = Path("dataset/labels.csv")
    pairs_json_path = Path("dataset/pairs.json")

    # Read existing rows or create header
    existing_rows = []
    if labels_csv_path.exists():
        with open(labels_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    new_row = {
        "pair_id": pair_id,
        "file": file_path.replace("\\", "/"),
        "role": role,
        "source": source,
        "baseline_file": baseline_file.replace("\\", "/"),
        "transcript_file": transcript_file.replace("\\", "/"),
        "flaw_type": flaw_type,
        "start_s": f"{start_s:.3f}",
        "end_s": f"{end_s:.3f}",
        "severity_level": str(severity_level),
        "param": param,
        "notes": notes,
    }
    existing_rows.append(new_row)

    with open(labels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in existing_rows:
            writer.writerow(r)

    print(f"Appended label for {pair_id} ({flaw_type} [{start_s:.3f} - {end_s:.3f}]) to {labels_csv_path}")

    # Update pairs.json if not present
    pairs = []
    if pairs_json_path.exists() and pairs_json_path.stat().st_size > 0:
        try:
            with open(pairs_json_path, "r", encoding="utf-8") as f:
                pairs = json.load(f)
        except Exception:
            pairs = []

    if not any(p.get("pair_id") == pair_id for p in pairs):
        pairs.append({
            "pair_id": pair_id,
            "baseline": baseline_file.replace("\\", "/"),
            "participant": file_path.replace("\\", "/"),
            "transcript": transcript_file.replace("\\", "/"),
            "level": level,
            "kind": kind,
            "flaw_type": flaw_type if kind == "single" else None,
        })
        with open(pairs_json_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)
            f.write("\n")
        print(f"Added entry for {pair_id} to {pairs_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a manual label to labels.csv and pairs.json.")
    parser.add_argument("--pair-id", required=True, help="Pair identifier, e.g. clip01_self_A")
    parser.add_argument("--file", required=True, help="Path to participant audio file")
    parser.add_argument("--role", default="bad", choices=["good", "bad"], help="Audio role")
    parser.add_argument("--source", default="self_recorded", choices=["public", "self_recorded", "synthetic"], help="Source type")
    parser.add_argument("--baseline-file", required=True, help="Path to baseline audio file")
    parser.add_argument("--transcript-file", required=True, help="Path to transcript text file")
    parser.add_argument("--flaw-type", required=True, help="Flaw category name")
    parser.add_argument("--start-s", type=float, required=True, help="Start time in seconds")
    parser.add_argument("--end-s", type=float, required=True, help="End time in seconds")
    parser.add_argument("--severity-level", type=int, default=3, help="Severity level 1-4")
    parser.add_argument("--param", default="", help="Parameter string, e.g. 'rate=1.45'")
    parser.add_argument("--notes", default="", help="Descriptive notes")
    parser.add_argument("--level", type=int, default=3, help="Overall severity level for pairs.json")
    parser.add_argument("--kind", default="self", choices=["mixed", "single", "self"], help="Pair kind")
    args = parser.parse_args()

    add_label(
        pair_id=args.pair_id,
        file_path=args.file,
        role=args.role,
        source=args.source,
        baseline_file=args.baseline_file,
        transcript_file=args.transcript_file,
        flaw_type=args.flaw_type,
        start_s=args.start_s,
        end_s=args.end_s,
        severity_level=args.severity_level,
        param=args.param,
        notes=args.notes,
        level=args.level,
        kind=args.kind,
    )


if __name__ == "__main__":
    main()
