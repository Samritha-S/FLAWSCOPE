"""Contract and pipeline smoke tests for FlawScope."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contracts.schema import AnalysisResult
from core.pipeline import analyze


def test_mock_result_validates():
    mock_file = Path(__file__).resolve().parent.parent / "contracts" / "mock_result.json"
    assert mock_file.exists(), "mock_result.json must exist"
    with open(mock_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = AnalysisResult.model_validate(data)
    assert len(result.words) == 60
    assert len(result.flaws) == 6
    assert result.scores.overall > 0


def test_analyze_import():
    assert callable(analyze)
