FlawScope - Functional Requirements Document (FRD)

Project: FlawScope - Counterpointing Speech Analytics & Temporal Flaw Founding (Statement 1)
Constraints: 4 hours, 2 engineers, working prototype, CPU only, English only

1. Purpose and scope

FlawScope is a web prototype that evaluates a spoken performance against a reference (baseline) delivery of the same text. It locates when the delivery deviates, explains why using the underlying acoustic numbers, scores it on a fixed rubric, and ships with a contrastive dataset and a validation run that prove the detectors work.

2. Traceability to the problem statement
Problem statement item	What it asks	Covered by
1a Good baseline	Source and transcribe recordings of highly effective speakers	FR-D1
1b Bad spectrum	Same text, intentionally flawed delivery, near-perfect to egregious	FR-D2, FR-D3
1c Labeling	Accurately label and temporally bound paired good/bad recordings	FR-D4
2a Forced alignment	Map the transcript onto the audio with word timestamps	FR-A1
2b Feature extraction	FFT, MFCCs, pitch, F0 contours, speech rate, pause intervals, clarity	FR-A2
3a Temporal grounding	Start and end timestamps of flaw regions vs baseline expectation	FR-A3
3b Causal explainability	Raw mathematical delta turned into a human-readable cause	FR-A4, FR-A5
4a Frontend	Interactive UI accepting audio and transcript uploads	FR-U1
4b Time-series overlay	Baseline vs participant features with flaw regions and explanations	FR-U2, FR-U3, FR-U5
Deliverable	Working dashboard: upload, feature extraction, visual and textual representation	FR-U1 to FR-U8
3. Functional requirements

Owner A = Signal Engine, B = Data and Dashboard. MUST items define the minimum demo. If time slips, cut in this order: FR-A7, FR-U9, FR-U6, FR-D3, FR-U8, FR-U7.

ID	Owner	Priority	Requirement
FR-D1	B	MUST	At least 3 good baseline clips (30-60 s, single speaker, English, public-domain or properly licensed) each with a verified transcript .txt.
FR-D2	B	MUST	For every good clip, a severity ladder of flawed versions of the exact same text (L1 near-perfect, L2 mild, L3 moderate, L4 egregious) made by a seeded, reproducible flaw-injection pipeline.
FR-D3	B	SHOULD	At least 2 human self-recorded flawed reads of the same transcript (a real-world, non-synthetic test).
FR-D4	B	MUST	labels.csv and pairs.json with temporally bounded ground truth (flaw type, start, end, level) for every injected flaw.
FR-A1	A	MUST	Word-level alignment of the transcript to each recording (start and end seconds); ASR mismatches resolved by sequence alignment; unmatched words interpolated and flagged.
FR-A2	A	MUST	Features per frame (100 Hz) and per word: F0 in semitones, intensity in dB, MFCC (12), log-mel spectrum (FFT), speech rate, pause durations, syllable counts.
FR-A3	A	MUST	Detect six flaw types by comparing participant to baseline over word-matched windows; output start/end seconds and word range on the participant timeline.
FR-A4	A	MUST	Per flaw: metric, participant value, baseline value, delta %, z-score, plain-English cause and a recommendation. Template-based and deterministic.
FR-A5	A	MUST	Rubric scores 0-100 for pace, intonation, volume, pausing, clarity plus an overall score using the formula in section 6; reproducible.
FR-A6	A	SHOULD	Caching by file hash and config hash; analysis of a 60 s pair in 60 s or less on a laptop CPU; cached rerun in 2 s or less.
FR-A7	A	COULD	True CTC forced-alignment backend behind a config flag, with fallback.
FR-U1	B	MUST	Upload baseline audio, baseline transcript and participant audio (optional separate participant transcript), or pick a demo pair from the dataset.
FR-U2	B	MUST	Time-series overlay of participant vs time-warped baseline for pitch, loudness and speech rate with shaded, colour-coded flaw regions and word hover.
FR-U3	B	MUST	Flaw panel: list and cards with time range, type, severity, explanation, and playback of the flawed segment and its baseline counterpart; selecting a flaw zooms the timeline.
FR-U4	B	MUST	Score cards and a radar chart for the five rubric dimensions plus overall.
FR-U5	B	MUST	Aligned transcript strip with flawed words highlighted by type.
FR-U6	B	SHOULD	Spectrogram and MFCC views with flaw boxes.
FR-U7	B	SHOULD	Validation tab: detected vs ground truth, precision/recall per flaw type, and the score-vs-severity ladder.
FR-U8	B	SHOULD	Method tab: the formulas and thresholds behind every detector (mathematical rationale), read live from config.py.
FR-U9	B	COULD	Sensitivity slider that re-runs detection (cached features) with a new Z threshold.
4. Architecture and repository
Stage	What happens	Module
1 Load	Mono 16 kHz audio, hashes for caching	core/audio_io
2 Align	Whisper word timestamps matched to the transcript with difflib; unmatched words interpolated	core/alignment
3 Features	F0 (semitones), intensity, MFCC, log-mel (FFT), per-word rate, pauses	core/features
4 Warp	Baseline time-warped onto the participant timeline via matched word anchors	core/warp
5 Detect	Word-matched window deviation d = (m_p - m_b)/s_b against the Z threshold; region merging	core/detectors
6 Explain + score	Deterministic templates; rubric scores 0-100	core/explain, core/scoring
7 Dashboard	Streamlit: overview, timeline overlay, flaw cards, spectrogram, validation, method	app/
text
flawscope/
  AGENTS.md  README.md  requirements.txt  config.py            (A)
  contracts/   schema.py  make_mock.py  mock_result.json       (A)
  core/        audio_io alignment features warp detectors
               explain scoring cache pipeline                  (A)
  app/         main.py  ui/ (theme adapter charts tab_*.py)    (B)
  dataset/     raw/ transcripts/ generated/ labels.csv
               pairs.json inject_flaws.py ...                  (B)
  eval/        run_eval.py  results.json                       (B)
  tests/       test_core_*.py (A)    test_dataset.py (B)
  docs/        tuning_log.md                                   (A)
5. Data contract

Everything the dashboard shows comes from one JSON object. Times are seconds on the participant timeline unless a field starts with b_. Schema changes must be additive and announced to the other engineer.

text
analyze(baseline_audio, participant_audio, transcript, participant_transcript=None,
        config_overrides=None) -> dict          # validates against contracts.schema.AnalysisResult

AnalysisResult
  meta        baseline_file, participant_file, duration_b, duration_p, align_backend,
              matched_ratio_b/p, config{}, timings_s{}, warnings[]
  words[]     idx, text, norm, has_punct_after, b_start, b_end, p_start, p_end, matched_b, matched_p
  word_feats[] idx, p{dur,f0_mean_st,f0_std_st,rms_db_rel,pause_before,rate_sps,syllables,
              mfcc_dtw_cost}, b{...same...}
  curves      t[] (participant grid, 100 Hz)
              p{f0_st[], rms_db_rel[], rate_sps[]}   b{...}  (baseline time-warped onto t)
  flaws[]     id, type, start, end, word_start_idx, word_end_idx, severity(0..1), metric, unit,
              participant_value, baseline_value, delta_pct, z_score, explanation, recommendation
  scores      overall, pace, intonation, volume, pausing, clarity   (0..100)

FlawType = MONOTONE | TOO_FAST | TOO_SLOW | LONG_PAUSE | LOW_VOLUME | UNCLEAR_ARTICULATION
6. Flaw taxonomy, detection rules and scoring

Window = the word plus two words either side. Baseline and participant are compared on the same words, so the delta is meaningful even when the recordings differ in length. Default Z = 1.5 (config.py).

Flaw	Metric	Flag rule	Why it matters
MONOTONE	std of F0 (semitones) in a 5-word window	d = (m_p - m_b)/s_b  <=  -Z	Flat pitch removes emphasis and intent cues
TOO_FAST	syllables per second of speech time	d  >=  +Z	Rushed delivery; listeners cannot process the content
TOO_SLOW	syllables per second of speech time	d  <=  -Z	Dragging pace loses energy and attention
LONG_PAUSE	silence between adjacent words	gap_p >= 0.9 s and gap_p - gap_b >= 0.5 s	Hesitation or lost thread
LOW_VOLUME	mean loudness (dB) vs own median speech level	d <= -Z and drop >= 4 dB	Trailing off; the message is not carried to the back of the room
UNCLEAR_ARTICULATION	DTW cost between word-matched MFCC sequences	robust z (median/MAD) >= Z + 0.5	Mumbled or smeared consonants
Severity of a flagged window = clip(|d| / (2Z), 0, 1); for LONG_PAUSE = clip((gap_p - gap_b) / 2.5, 0, 1).
Regions: flagged words at most 1 word apart are merged; regions shorter than 0.6 s are dropped (pauses excepted).
Dimension score = 100 x exp(-3 x sum(severity x duration) / T_speech). Pace = TOO_FAST + TOO_SLOW; Intonation = MONOTONE; Volume = LOW_VOLUME; Pausing = LONG_PAUSE; Clarity = UNCLEAR_ARTICULATION.
Overall = 0.25 pace + 0.25 intonation + 0.15 volume + 0.15 pausing + 0.20 clarity. Deterministic: fixed seeds, temperature 0, no randomness.
Limitation: pace and pitch comparisons assume the same or a similar speaker; articulation uses a robust z-score on the participant's own costs because speakers differ.
7. Dataset specification
Item	Specification
Good clips	3 clips, 30-60 s, one speaker, clean audio, public-domain or properly licensed; transcript verified by listening
Severity ladder	Per clip: L1 (near-perfect), L2, L3, L4 (egregious) mixed-flaw versions of the exact same text; 1 / 2 / 4 / 6 flawed pieces
Single-flaw set	clip01 only: six level-3 files, one flaw type each
Self-recorded	2 human flawed reads of one transcript, labelled by hand with add_label.py
Labels	labels.csv (pair_id, file, role, source, baseline_file, transcript_file, flaw_type, start_s, end_s, severity_level, param, notes); pairs.json lists every baseline/participant pair
Licensing	Modified audio must come from public-domain or permissively licensed sources. TED talks are CC BY-NC-ND (no derivatives): not for the modified set. Log source and licence per clip
8. Dashboard specification
Sidebar: demo-pair picker or three uploads (baseline audio, participant audio, transcript), sensitivity slider, Analyze.
Overview: overall + five dimension score cards, radar chart, flaw counts by type, aligned transcript strip with flawed words highlighted, top 3 flaws.
Timeline: three stacked rows (pitch, loudness, speech rate), participant solid vs baseline dashed, colour-coded flaw regions, word hover, focus-on-flaw zoom with audio playback.
Flaws: sortable table and expandable cards with explanation, numbers, recommendation and playback; JSON/CSV download.
Spectrogram, Validation (detected vs truth, precision/recall, severity ladder) and Method (formulas and live thresholds).
Palette: teal 
#0F9D8A, amber 
#F5A623, coral 
#FF6B5B, sky 
#2D9CDB, green 
#3BB273, brown 
#8D6E63, ink 
#1F2A37.
9. Non-functional requirements, scope limits, acceptance
Python 3.10+, CPU only, offline after the one-off Whisper model download; a 60 s pair analysed in 60 s or less, cached rerun 2 s or less.
Deterministic and reproducible: identical inputs and config give identical JSON. Thresholds live only in config.py.
Out of scope: other languages, live streaming, accounts, deployment, speaker identification, scoring the content or argument itself.
Acceptance: (1) fresh clone + pip install + streamlit run works; (2) the demo pair shows flaw regions that match what you hear; (3) every flaw has numbers and a plain-English cause; (4) good-vs-good returns no flaws and about 100; (5) the Validation tab shows real precision/recall and a score that falls from L1 to L4.
