"""Flaws panel tab module placeholder."""

from typing import Any, Dict
import streamlit as st


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Flaws tab interface."""
    st.subheader("Detected Delivery Flaws")
    st.markdown(
        """
        <div class="flawscope-card">
            <h4>Flaws Detail Panel (Phase B4)</h4>
            <p>Sortable table and expandable cards detailing exact acoustic metrics, z-scores,
            deterministic explanations, actionable tips, and side-by-side audio playback excerpts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
