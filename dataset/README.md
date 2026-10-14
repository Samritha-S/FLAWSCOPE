# FlawScope Dataset

This directory contains reference baseline speech recordings, verified transcriptions, generated flaw ladders, and ground-truth bounding annotations.

## Directory Structure

- `raw_src/`: Original unedited source audio (historical speeches, unnormalized).
- `raw/`: Standardized baseline clips (16 kHz mono WAV, 30-60 s).
- `transcripts/`: Verified text transcripts (`.txt`) corresponding to each baseline clip.
- `generated/`: Synthetically injected flawed speech files (L1 to L4 severity ladders and single-type isolates).
- `labels.csv`: Ground-truth bounding intervals and flaw annotations.
- `pairs.json`: Mapping of baseline-to-participant audio pairs.

---

## Sources Table

| clip_id | speaker | source URL | licence | date | notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `clip01` | Franklin D. Roosevelt | [FDR Library / National Archives](https://www.fdrlibrary.org) | Public Domain (US Gov) | 1933-05-07 | Fireside Chat 2: Outlining New Deal Program |
| `clip02` | Franklin D. Roosevelt | [FDR Library / National Archives](https://www.fdrlibrary.org) | Public Domain (US Gov) | 1933-03-12 | Fireside Chat 1: On the Bank Crisis |
| `clip03` | Franklin D. Roosevelt | [FDR Library / National Archives](https://www.fdrlibrary.org) | Public Domain (US Gov) | 1933-05-07 | Fireside Chat 2: Outlining New Deal Program |
| `clip04` | *(Placeholder)* | *(Fill in URL)* | *(Public Domain / CC BY)* | *(YYYY-MM-DD)* | *(Placeholder for future addition)* |
| `clip05` | *(Placeholder)* | *(Fill in URL)* | *(Public Domain / CC BY)* | *(YYYY-MM-DD)* | *(Placeholder for future addition)* |

---

## Calibrated Flaw Injection Recipes

Flaws are injected by splitting good baseline audio at silence cut points (midpoints of non-speech gaps $\ge$ 120 ms), transforming the flawed pieces, and concatenating with 10 ms edge fades. Cumulative timeline tracking ensures exact ground-truth timestamps.

| Flaw Type | Description & Implementation | Level 1 (Near-perfect) | Level 2 (Mild) | Level 3 (Moderate) | Level 4 (Egregious) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `TOO_FAST` | Time-stretching speedup via phase vocoder / `atempo` | 1.10 rate | 1.25 rate | 1.45 rate | 1.70 rate |
| `TOO_SLOW` | Time-stretching slowdown via phase vocoder / `atempo` | 0.92 rate | 0.80 rate | 0.68 rate | 0.55 rate |
| `LONG_PAUSE` | Silence duration inserted at cut point | 0.6 s | 1.0 s | 1.6 s | 2.4 s |
| `LOW_VOLUME` | Gain attenuation with 50 ms edge fades | -3 dB | -6 dB | -10 dB | -15 dB |
| `MONOTONE` | Pitch contour compression towards log-median via Parselmouth | 0.80 fraction | 0.50 fraction | 0.25 fraction | 0.00 fraction |
| `UNCLEAR_ARTICULATION` | 4th-order Butterworth low-pass filter + 30 dB SNR noise | 5000 Hz cutoff | 3500 Hz cutoff | 2500 Hz cutoff | 1500 Hz cutoff |

### Mixed Flaw Distribution per Level

- **Level 1 (`<clip>_L1_mixed.wav`)**: 1 flawed segment
- **Level 2 (`<clip>_L2_mixed.wav`)**: 2 flawed segments
- **Level 3 (`<clip>_L3_mixed.wav`)**: 4 flawed segments
- **Level 4 (`<clip>_L4_mixed.wav`)**: 6 flawed segments (covers all six flaw types)
- **Single-flaw isolates (`clip01_L3_<FLAW_TYPE>.wav`)**: 1 isolated flaw of that specific type at Level 3

---

## Dataset Schema

### `labels.csv`
- `pair_id`: Identifier of audio pair (e.g., `clip01_L1_mixed`)
- `file`: Path to participant audio
- `role`: `good` (baseline) or `bad` (participant)
- `source`: `public`, `self_recorded`, or `synthetic`
- `baseline_file`: Path to baseline audio
- `transcript_file`: Path to transcript text
- `flaw_type`: Flaw category or `NONE`
- `start_s`: Flaw start in seconds on output timeline
- `end_s`: Flaw end in seconds on output timeline
- `severity_level`: Integer severity (1 to 4, 0 for good)
- `param`: Exact transformation parameter
- `notes`: Human-readable description

### `pairs.json`
List of objects:
```json
{
  "pair_id": "clip01_L1_mixed",
  "baseline": "dataset/raw/clip01.wav",
  "participant": "dataset/generated/clip01_L1_mixed.wav",
  "transcript": "dataset/transcripts/clip01.txt",
  "level": 1,
  "kind": "mixed",
  "flaw_type": null
}
```

---

## Reproducibility & Commands

- **Generate baseline audio**:
  ```bash
  python dataset/prepare_raw.py --max-sec 60
  ```
- **Inject flaws (all clips)**:
  ```bash
  python dataset/inject_flaws.py --all --seed 7
  ```
- **Validate dataset**:
  ```bash
  python dataset/validate_dataset.py
  ```
- **Append manual self-recorded label**:
  ```bash
  python dataset/add_label.py --pair-id clip01_self_A --file dataset/raw/clip01_self_A.wav --baseline-file dataset/raw/clip01.wav --transcript-file dataset/transcripts/clip01.txt --flaw-type UNCLEAR_ARTICULATION --start-s 5.2 --end-s 8.1
  ```
