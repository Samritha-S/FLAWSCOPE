"""Main execution pipeline for FlawScope audio alignment and flaw analysis."""

import json
import os
from pathlib import Path
from contracts.schema import AnalysisResult


def analyze(
    baseline_audio: str,
    participant_audio: str,
    transcript: str,
    participant_transcript: str | None = None,
    config_overrides: dict | None = None,
) -> dict:
    """Analyze spoken performance against a baseline reference.

    Args:
        baseline_audio: Path to reference baseline audio file.
        participant_audio: Path to participant audio file.
        transcript: Reference transcript text.
        participant_transcript: Optional participant transcript text.
        config_overrides: Optional dictionary of configuration overrides.

    Returns:
        JSON-serializable dictionary matching contracts.schema.AnalysisResult.
    """
    if os.environ.get("FLAWSCOPE_MOCK") == "1":
        mock_path = Path(__file__).resolve().parent.parent / "contracts" / "mock_result.json"
        with open(mock_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        # Validate against AnalysisResult schema before returning
        validated = AnalysisResult.model_validate(raw_data)
        return validated.model_dump(mode="json")

    raise NotImplementedError(
        "Full signal engine analysis is not yet implemented. Set FLAWSCOPE_MOCK=1 to use mock data."
    )
