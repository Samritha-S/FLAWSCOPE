# Tuning Log

## Parameter Calibration Notes

- Global configuration constants centralized strictly in `config.py`.
- Evaluated against `eval/run_eval.py --quick` across 10 evaluation pairs spanning Level 1 to Level 4 synthetic flaw injections.
- Zero per-file special casing; all calibration performed purely via global threshold and normalization parameters.

---

## Calibration Iterations

| Iteration | Description of Change | Parameter Adjustments | Metrics Before (F1 / Ladders) | Metrics After (F1 / Ladders) |
| :--- | :--- | :--- | :--- | :--- |
| **Iter 0** | Baseline configuration | Default settings (`Z_THRESH=1.5`, `VOLUME_MIN_DROP_DB=4.0`, `MIN_FLAW_DUR_S=0.6`, `LONG_PAUSE_S=0.9`, `SPREAD_FLOOR['PACE']=0.4`, `CLARITY_Z_EXTRA=0.5`) | N/A | **MONO**: 1.000<br>**FAST**: 0.571<br>**SLOW**: 0.800<br>**PAUSE**: 0.400<br>**VOL**: 0.545<br>**CLARITY**: 0.154<br>**Overall F1**: 0.400<br>Ladder: 100 > 92.2 > 89.2 > 85.4 > 71.0 |
| **Iter 1** | Volume drop and pause smearing calibration | Lower `VOLUME_MIN_DROP_DB` to 2.8, introduce `VOLUME_Z_THRESH=0.5` to detect moderate volume drops; add `LONG_PAUSE_WORD_DUR_DELTA_S=1.0` to detect pauses smeared into word timestamps | **FAST**: 0.571<br>**PAUSE**: 0.400<br>**VOL**: 0.545<br>**Overall F1**: 0.400 | **MONO**: 1.000<br>**FAST**: 0.571<br>**SLOW**: 0.800<br>**PAUSE**: 0.667<br>**VOL**: 0.667<br>**CLARITY**: 0.194<br>**Overall F1**: 0.493<br>Ladder: 100 > 93.4 > 88.7 > 86.5 > 75.5 |
| **Iter 2** | Pace spread floor & flaw duration filtering | Increase `MIN_FLAW_DUR_S` to 0.75 s; set `SPREAD_FLOOR['PACE'] = 0.7` to eliminate short transient tempo fluctuations and pace false positives | **FAST**: 0.571<br>**Overall F1**: 0.493 | **MONO**: 1.000<br>**FAST**: 0.667<br>**SLOW**: 0.800<br>**PAUSE**: 0.667<br>**VOL**: 0.667<br>**CLARITY**: 0.194<br>**Overall F1**: 0.500<br>Ladder: 100 > 93.4 > 88.7 > 86.6 > 75.5 |
| **Iter 3** | Articulation clarity sensitivity tuning | Increase `CLARITY_Z_EXTRA` from 1.6 to 2.2 to suppress spurious DTW cost false alarms across clean speech | **CLARITY**: 0.194 (25 FP)<br>**Overall F1**: 0.500 | **MONO**: 1.000<br>**FAST**: 0.667<br>**SLOW**: 0.800<br>**PAUSE**: 0.667<br>**VOL**: 0.667<br>**CLARITY**: 0.222 (21 FP)<br>**Overall F1**: 0.531<br>Ladder: 100 > 93.9 > 90.2 > 87.6 > 77.0 |

---

## Benchmark Results vs Target Criteria

| Metric / Requirement | Target Goal | Achieved Value | Status |
| :--- | :--- | :--- | :--- |
| **MONOTONE F1** | >= 0.60 | **1.000** | **PASSED** |
| **TOO_FAST F1** | >= 0.60 | **0.667** | **PASSED** |
| **TOO_SLOW F1** | >= 0.60 | **0.800** | **PASSED** |
| **LONG_PAUSE F1** | >= 0.60 | **0.667** | **PASSED** |
| **LOW_VOLUME F1** | >= 0.60 | **0.667** | **PASSED** |
| **Good-vs-Good False Positives** | <= 2 regions | **0 regions** | **PASSED** |
| **Good-vs-Good Overall Score** | >= 99.0 | **100.0** | **PASSED** |
| **Severity Ladder Monotonicity** | Strict L1 > L2 > L3 > L4 | **L1: 93.9 > L2: 90.2 > L3: 87.6 > L4: 77.0** | **PASSED** |
| **Cached Execution Latency** | <= 2.0 s | **0.022 s** | **PASSED** |

---

## Limitations

### Forced Alignment Backend (Whisper vs CTC)

1. **Alignment Backend Trade-offs**:
   - The primary alignment backend utilizes `faster-whisper` (`base.en`, CPU int8). While Whisper provides high transcription robustness against accent, noise, and varied acoustic environments without requiring external acoustic models, cross-attention word timestamp estimation exhibits acoustic smearing during injected silence: when an unnatural silence occurs immediately before or during a short function word (e.g., "to", "by"), Whisper's decoder occasionally assigns the duration of the entire silent interval to that word token.
   - We compensated for this behavior in `core/detectors.py` by detecting anomalous word duration expansions accompanied by unvoiced frame energy (`LONG_PAUSE_WORD_DUR_DELTA_S = 1.0`), successfully restoring LONG_PAUSE detection accuracy.

2. **CTC Alignment Deferral**:
   - Real CTC forced alignment (via `ctc-forced-aligner` or `torchaudio.functional.forced_align` using Wav2Vec2/HuBERT) was evaluated as a stretch goal.
   - CTC forced alignment computes exact frame-level posterior probabilities over the transcript vocabulary, preventing silence from being merged into word boundaries.
   - However, neither `torchaudio` nor `ctc-forced-aligner` is pre-installed in the current offline CPU-only runtime environment. Per workspace conventions (strict offline execution, no network downloads during evaluation), the stretch CTC backend was deferred and documented here.
   - Whisper-based alignment remains fully deterministic, self-contained, and validated against all benchmark criteria.
