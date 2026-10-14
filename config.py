"""Single source of truth for all configuration and detection thresholds in FlawScope."""

SR = 16000
HOP_S = 0.01
F0_MIN = 75
F0_MAX = 500
MAX_AUDIO_S = 120

ASR_MODEL = "base.en"
ALIGN_BACKEND = "whisper"  # "whisper" | "ctc"

WIN_RADIUS = 2  # window for word i = words i-2 .. i+2
Z_THRESH = 1.5
MIN_FLAW_DUR_S = 0.6
MERGE_GAP_WORDS = 1

MIN_VOICED_FRAMES = 15
DTW_MIN_FRAMES = 3

LONG_PAUSE_S = 0.9
LONG_PAUSE_DELTA_S = 0.5
LONG_PAUSE_MAX_SCALE_S = 2.5
VOLUME_MIN_DROP_DB = 4.0
CLARITY_Z_EXTRA = 0.5

SPREAD_FLOOR = {
    "MONOTONE": 0.5,
    "PACE": 0.4,
    "LOW_VOLUME": 1.5,
}

PENALTY_K = 3.0

WEIGHTS = {
    "pace": 0.25,
    "intonation": 0.25,
    "volume": 0.15,
    "pausing": 0.15,
    "clarity": 0.20,
}

FLAW_TYPES = [
    "MONOTONE",
    "TOO_FAST",
    "TOO_SLOW",
    "LONG_PAUSE",
    "LOW_VOLUME",
    "UNCLEAR_ARTICULATION",
]

FLAW_DIMENSION = {
    "TOO_FAST": "pace",
    "TOO_SLOW": "pace",
    "MONOTONE": "intonation",
    "LOW_VOLUME": "volume",
    "LONG_PAUSE": "pausing",
    "UNCLEAR_ARTICULATION": "clarity",
}


def snapshot() -> dict:
    """Return all global configuration constants as a serializable dictionary."""
    return {
        "SR": SR,
        "HOP_S": HOP_S,
        "F0_MIN": F0_MIN,
        "F0_MAX": F0_MAX,
        "MAX_AUDIO_S": MAX_AUDIO_S,
        "ASR_MODEL": ASR_MODEL,
        "ALIGN_BACKEND": ALIGN_BACKEND,
        "WIN_RADIUS": WIN_RADIUS,
        "Z_THRESH": Z_THRESH,
        "MIN_FLAW_DUR_S": MIN_FLAW_DUR_S,
        "MERGE_GAP_WORDS": MERGE_GAP_WORDS,
        "MIN_VOICED_FRAMES": MIN_VOICED_FRAMES,
        "DTW_MIN_FRAMES": DTW_MIN_FRAMES,
        "LONG_PAUSE_S": LONG_PAUSE_S,
        "LONG_PAUSE_DELTA_S": LONG_PAUSE_DELTA_S,
        "LONG_PAUSE_MAX_SCALE_S": LONG_PAUSE_MAX_SCALE_S,
        "VOLUME_MIN_DROP_DB": VOLUME_MIN_DROP_DB,
        "CLARITY_Z_EXTRA": CLARITY_Z_EXTRA,
        "SPREAD_FLOOR": dict(SPREAD_FLOOR),
        "PENALTY_K": PENALTY_K,
        "WEIGHTS": dict(WEIGHTS),
        "FLAW_TYPES": list(FLAW_TYPES),
        "FLAW_DIMENSION": dict(FLAW_DIMENSION),
    }
