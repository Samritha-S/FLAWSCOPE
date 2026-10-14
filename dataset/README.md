# FlawScope Dataset

This directory contains reference baseline speech recordings, verified transcriptions, generated flaw ladders, and ground-truth bounding annotations.

## Directory Structure

- `raw_src/`: Original unedited source audio (ignored by git).
- `raw/`: Standardized baseline clips (16 kHz mono WAV, 30-60 s).
- `transcripts/`: Verified text transcripts (`.txt`) corresponding to each baseline clip.
- `generated/`: Synthetically injected flawed speech files (L1 to L4 severity ladders).
- `labels.csv`: Ground-truth bounding intervals and flaw annotations.
- `pairs.json`: Mapping of baseline-to-participant audio pairs.

## Licensing Rules

- All source audio used for synthetic modification must be public-domain or permissively licensed (e.g., CC BY).
- Audio under restrictive licenses such as CC BY-NC-ND (including TED talks) must NOT be used for derivative flaw injection.
- Detailed clip attribution and licensing terms are recorded in `labels.csv` and root documentation.
