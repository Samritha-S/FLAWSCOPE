# AGENTS.md - Workspace Rules and Engineering Conventions

## Product Summary

FlawScope evaluates a spoken performance against an ideal "baseline" delivery of the SAME transcript.
It performs the following sequence:
1. Aligns the transcript to both recordings at word level.
2. Extracts acoustic features (F0, intensity, MFCC, spectrum, speech rate, pauses).
3. Detects time-bounded delivery flaws by comparing participant vs baseline over word-matched windows.
4. Explains each flaw with exact acoustic numbers and deltas.
5. Scores five rubric dimensions (pace, intonation, volume, pausing, clarity) and an overall score.
6. Displays analysis in an interactive Streamlit dashboard.

Stack: Python 3.10+, Streamlit, Plotly, NumPy, SciPy, Pandas, Librosa, SoundFile, Praat-Parselmouth, faster-whisper (base.en, CPU int8), Pydantic v2, PyTest. Everything must remain simple, deterministic, and CPU-only.

---

## Repository Tree

```text
flawscope/
  AGENTS.md
  README.md
  requirements.txt
  config.py
  .gitignore
  .streamlit/
    config.toml
  contracts/
    schema.py
    make_mock.py
    mock_result.json
  core/
    __init__.py
    audio_io.py
    alignment.py
    features.py
    warp.py
    detectors.py
    explain.py
    scoring.py
    cache.py
    pipeline.py
  app/
    main.py
    ui/
      __init__.py
      theme.py
      adapter.py
      charts.py
      components.py
  dataset/
    raw_src/
    raw/
    transcripts/
    generated/
    prepare_raw.py
    segmenter.py
    inject_flaws.py
    add_label.py
    validate_dataset.py
    labels.csv
    pairs.json
    README.md
  eval/
    run_eval.py
  tests/
    test_core_contracts.py
  docs/
    FRD.md
    tuning_log.md
```

---

## Ownership Matrix

| Engineer | Owned Paths & Modules |
| :--- | :--- |
| **Teammate A (Signal Engine)** | `core/`, `config.py`, `contracts/`, `tests/test_core_*` |
| **Teammate B (Data and Dashboard)** | `dataset/`, `eval/`, `app/`, `tests/test_dataset.py`, `.streamlit/` |

**Rule**: Never edit files outside your assigned ownership. If changes in the other subsystem are required, ask the human coordinator instead.

---

## Contract and Threshold Rules

1. **Additive Contract Changes**: Schema or enum modifications in `contracts/schema.py` must be strictly additive and immediately announced to the partner engineer.
2. **Centralized Configuration**: All thresholds, window sizes, weights, and acoustic constants must live ONLY in `config.py`. Never hardcode detection thresholds or rubric weights in individual modules.

---

## Engineering Conventions

- **Type Hints & Docstrings**: All functions and classes must include clear type hints and docstrings.
- **Pure Functions**: Write pure, stateless signal processing functions where possible.
- **Determinism**: Fixed random seeds everywhere (`seed=7` default). No randomness without an explicit seed. Temperature set to 0.
- **ASCII Only**: Use standard ASCII characters in source code, comments, and strings.
- **Testing**: Maintain small, focused `pytest` tests for every core feature.
- **Network Isolation**: Zero network access at runtime, except for the one-off Whisper model download.
- **CPU Bound**: All signal processing and inference must run efficiently on a standard laptop CPU.

---

## UI Color Palette

- **Primary Teal**: `#0F9D8A`
- **Amber**: `#F5A623`
- **Coral**: `#FF6B5B`
- **Sky**: `#2D9CDB`
- **Green**: `#3BB273`
- **Brown**: `#8D6E63`
- **Ink**: `#1F2A37`
- **Soft Background**: `#F4FAF9`

### Flaw Category Colors

- `MONOTONE`: Coral (`#FF6B5B`)
- `TOO_FAST`: Amber (`#F5A623`)
- `TOO_SLOW`: Sky (`#2D9CDB`)
- `LONG_PAUSE`: Teal (`#0F9D8A`)
- `LOW_VOLUME`: Brown (`#8D6E63`)
- `UNCLEAR_ARTICULATION`: Green (`#3BB273`)

### Curve Styling

- Baseline curve: Dashed grey (`#8A94A6`)
- Participant curve: Solid ink (`#1F2A37`)

---

## Git and Reporting Workflow

- **Commit Messages**: Every commit message must be prefixed by the engineer identifier, e.g., `A: ...` or `B: ...`.
- **Done Summary**: Every completed task must conclude with a structured DONE summary detailing:
  1. Files changed or created
  2. Commands executed and their status
  3. Verification results
  4. Known limitations or pending follow-ups
