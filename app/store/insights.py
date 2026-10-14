"""Deterministic rule-based progress insights generator from speaker sessions."""

from collections import Counter
import json
from typing import Any, Dict, List, Optional

DIMENSIONS = ["pace", "intonation", "volume", "pausing", "clarity"]


def generate_insights_text(sessions: List[Dict[str, Any]]) -> str:
    """Generate concise text summary of trends across speaker sessions.

    Args:
        sessions: List of session dictionaries sorted oldest to latest.

    Returns:
        Deterministic insight message.
    """
    if len(sessions) < 2:
        return "Complete one more session to see your trend."

    data = generate_insights(sessions)
    bullets = data.get("bullets", [])
    return "\n".join(bullets)


def generate_insights(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute progress insights comparing first to latest session.

    Analyzes:
    - Overall score change from first to latest session
    - Most improved rubric dimension
    - Weakest rubric dimension in latest session
    - Most frequent delivery flaw and number of sessions it appeared in

    Args:
        sessions: List of session dictionaries sorted oldest to latest.

    Returns:
        Dictionary containing insight fields and formatted bullet strings.
    """
    if len(sessions) < 2:
        msg = "Complete one more session to see your trend."
        return {
            "ready": False,
            "message": msg,
            "text": msg,
            "bullets": [msg],
        }

    first = sessions[0]
    latest = sessions[-1]
    n_sessions = len(sessions)

    # 1. Overall change first to latest
    first_overall = float(first.get("overall", 0.0))
    latest_overall = float(latest.get("overall", 0.0))
    diff_overall = latest_overall - first_overall

    if diff_overall > 0.05:
        overall_sentence = (
            f"Overall score improved by {diff_overall:+.1f} points "
            f"(from {first_overall:.1f} to {latest_overall:.1f})."
        )
    elif diff_overall < -0.05:
        overall_sentence = (
            f"Overall score decreased by {abs(diff_overall):.1f} points "
            f"(from {first_overall:.1f} to {latest_overall:.1f})."
        )
    else:
        overall_sentence = f"Overall score remained steady at {latest_overall:.1f} points."

    # 2. Most improved dimension
    dim_diffs = {
        dim: float(latest.get(dim, 0.0)) - float(first.get(dim, 0.0))
        for dim in DIMENSIONS
    }
    best_dim = max(DIMENSIONS, key=lambda d: dim_diffs[d])
    best_gain = dim_diffs[best_dim]

    if best_gain > 0:
        improved_sentence = (
            f"Most improved dimension: {best_dim.title()} "
            f"({best_gain:+.1f} points, from {float(first.get(best_dim, 0.0)):.1f} to {float(latest.get(best_dim, 0.0)):.1f})."
        )
    else:
        improved_sentence = (
            f"Most improved dimension: None (all dimensions flat or decreased; best was {best_dim.title()} with {best_gain:+.1f} points)."
        )

    # 3. Weakest dimension on latest session
    latest_dims = {dim: float(latest.get(dim, 0.0)) for dim in DIMENSIONS}
    weakest_dim = min(DIMENSIONS, key=lambda d: latest_dims[d])
    weakest_score = latest_dims[weakest_dim]
    weakest_sentence = f"Weakest dimension: {weakest_dim.title()} ({weakest_score:.1f} points in your latest session)."

    # 4. Most frequent flaw and in how many sessions
    flaw_session_counts: Counter[str] = Counter()
    for s in sessions:
        raw_counts = s.get("flaw_counts")
        if isinstance(raw_counts, str):
            try:
                raw_counts = json.loads(raw_counts)
            except Exception:
                raw_counts = {}
        elif not isinstance(raw_counts, dict):
            raw_json = s.get("flaw_counts_json")
            if raw_json:
                try:
                    raw_counts = json.loads(raw_json)
                except Exception:
                    raw_counts = {}
            else:
                raw_counts = {}

        for flaw_type, count in raw_counts.items():
            if count > 0:
                flaw_session_counts[flaw_type] += 1

    if flaw_session_counts:
        top_flaw, session_count = flaw_session_counts.most_common(1)[0]
        flaw_name = top_flaw.replace("_", " ").title()
        flaw_sentence = f"Most frequent flaw: {flaw_name} (appeared in {session_count} of {n_sessions} sessions)."
    else:
        flaw_sentence = "Most frequent flaw: None (zero delivery flaws recorded across sessions)."

    bullets = [
        overall_sentence,
        improved_sentence,
        weakest_sentence,
        flaw_sentence,
    ]

    return {
        "ready": True,
        "first_overall": first_overall,
        "latest_overall": latest_overall,
        "diff_overall": round(diff_overall, 1),
        "overall_sentence": overall_sentence,
        "most_improved_dim": best_dim,
        "best_gain": round(best_gain, 1),
        "improved_sentence": improved_sentence,
        "weakest_dim": weakest_dim,
        "weakest_score": round(weakest_score, 1),
        "weakest_sentence": weakest_sentence,
        "most_frequent_flaw": flaw_sentence,
        "bullets": bullets,
        "text": "\n".join(bullets),
    }
