"""Adapter module transforming contract analysis results into presentation structures."""

from typing import Any
import pandas as pd


def flaws_to_dataframe(flaws: list[Any]) -> pd.DataFrame:
    """Convert list of Flaw objects/dicts to a pandas DataFrame."""
    rows = []
    for f in flaws:
        if isinstance(f, dict):
            rows.append({
                "ID": f.get("id"),
                "Type": f.get("type"),
                "Start (s)": round(float(f.get("start", 0)), 2),
                "End (s)": round(float(f.get("end", 0)), 2),
                "Duration (s)": round(float(f.get("end", 0)) - float(f.get("start", 0)), 2),
                "Severity": round(float(f.get("severity", 0)), 2),
                "Participant": f.get("participant_value"),
                "Baseline": f.get("baseline_value"),
                "Delta %": f.get("delta_pct"),
                "Z-Score": f.get("z_score"),
                "Explanation": f.get("explanation"),
                "Recommendation": f.get("recommendation"),
            })
        else:
            rows.append({
                "ID": f.id,
                "Type": f.type.value if hasattr(f.type, "value") else str(f.type),
                "Start (s)": round(float(f.start), 2),
                "End (s)": round(float(f.end), 2),
                "Duration (s)": round(float(f.end - f.start), 2),
                "Severity": round(float(f.severity), 2),
                "Participant": f.participant_value,
                "Baseline": f.baseline_value,
                "Delta %": f.delta_pct,
                "Z-Score": f.z_score,
                "Explanation": f.explanation,
                "Recommendation": f.recommendation,
            })
    return pd.DataFrame(rows)


def words_to_dataframe(words: list[Any]) -> pd.DataFrame:
    """Convert list of Word objects/dicts to a pandas DataFrame."""
    rows = []
    for w in words:
        if isinstance(w, dict):
            rows.append({
                "Idx": w.get("idx"),
                "Text": w.get("text"),
                "Norm": w.get("norm"),
                "P Start": w.get("p_start"),
                "P End": w.get("p_end"),
                "B Start": w.get("b_start"),
                "B End": w.get("b_end"),
                "Matched P": w.get("matched_p"),
                "Matched B": w.get("matched_b"),
            })
        else:
            rows.append({
                "Idx": w.idx,
                "Text": w.text,
                "Norm": w.norm,
                "P Start": w.p_start,
                "P End": w.p_end,
                "B Start": w.b_start,
                "B End": w.b_end,
                "Matched P": w.matched_p,
                "Matched B": w.matched_b,
            })
    return pd.DataFrame(rows)
