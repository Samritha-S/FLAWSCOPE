"""Method tab module placeholder."""

from typing import Any, Dict
import streamlit as st


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Method tab interface."""
    try:
        st.subheader("Methodology & Mathematical Rationale")
        st.markdown(
            """
            <div class="flawscope-card">
                <h4>Algorithm & Metric Formulations (Phase B5)</h4>
                <p>Mathematical formulas, DTW time-warping rationale, scoring weights, and active detection
                thresholds loaded live from <code>config.py</code>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:
        st.error(f"Error rendering Method tab: {exc}")
