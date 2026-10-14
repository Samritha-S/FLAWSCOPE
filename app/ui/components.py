"""Reusable Streamlit dashboard UI components (score cards, flaw cards, transcript strip)."""

from typing import Any
import streamlit as st

from app.ui.theme import FLAW_COLORS, PRIMARY_TEAL


def render_score_cards(scores: Any) -> None:
    """Render overall score card and 5 dimension metrics."""
    overall = scores.overall if hasattr(scores, "overall") else scores.get("overall", 100.0)
    pace = scores.pace if hasattr(scores, "pace") else scores.get("pace", 100.0)
    intonation = scores.intonation if hasattr(scores, "intonation") else scores.get("intonation", 100.0)
    volume = scores.volume if hasattr(scores, "volume") else scores.get("volume", 100.0)
    pausing = scores.pausing if hasattr(scores, "pausing") else scores.get("pausing", 100.0)
    clarity = scores.clarity if hasattr(scores, "clarity") else scores.get("clarity", 100.0)

    # Color code overall badge
    if overall >= 85:
        score_color = "#3BB273"  # green
        score_label = "Effective Delivery"
    elif overall >= 70:
        score_color = "#F5A623"  # amber
        score_label = "Moderate Delivery"
    else:
        score_color = "#FF6B5B"  # coral
        score_label = "Flawed Delivery"

    c_ov, c_sub = st.columns([1, 2.5])
    with c_ov:
        st.markdown(
            f"""
            <div style="background-color: #FFFFFF; border-radius: 10px; padding: 18px; border-left: 6px solid {score_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <div style="font-size: 13px; color: #6B7280; text-transform: uppercase; font-weight: bold;">Overall Score</div>
                <div style="font-size: 42px; font-weight: bold; color: {score_color}; line-height: 1.1;">{overall:.1f} <span style="font-size: 20px; color: #9CA3AF;">/ 100</span></div>
                <div style="font-size: 12px; color: #4B5563; margin-top: 4px;">{score_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_sub:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Pace", f"{pace:.1f}")
        col2.metric("Intonation", f"{intonation:.1f}")
        col3.metric("Volume", f"{volume:.1f}")
        col4.metric("Pausing", f"{pausing:.1f}")
        col5.metric("Clarity", f"{clarity:.1f}")


def render_transcript_strip(words: list[Any], flaws: list[Any]) -> None:
    """Render aligned transcript strip with words highlighted by flaw category."""
    st.markdown("##### Aligned Transcript Strip")

    # Map word index to flaw category
    word_flaw_map: dict[int, str] = {}
    for f in flaws:
        ftype = f.type.value if hasattr(f.type, "value") else str(f.type)
        w_start = int(f.word_start_idx if hasattr(f, "word_start_idx") else f.get("word_start_idx", 0))
        w_end = int(f.word_end_idx if hasattr(f, "word_end_idx") else f.get("word_end_idx", 0))
        for idx in range(w_start, w_end + 1):
            if idx not in word_flaw_map:
                word_flaw_map[idx] = ftype

    spans = []
    for w in words:
        idx = w.idx if hasattr(w, "idx") else w.get("idx", 0)
        text = w.text if hasattr(w, "text") else w.get("text", "")
        ftype = word_flaw_map.get(idx)

        if ftype:
            bg_col = FLAW_COLORS.get(ftype, PRIMARY_TEAL)
            span_html = (
                f"<span title='{ftype}' style='background-color: {bg_col}33; "
                f"border-bottom: 2px solid {bg_col}; padding: 2px 4px; margin: 2px; "
                f"border-radius: 3px; font-weight: 500;'>{text}</span>"
            )
        else:
            span_html = f"<span style='padding: 2px 3px; margin: 2px;'>{text}</span>"
        spans.append(span_html)

    strip_html = f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 14px; line-height: 2.2; font-size: 15px; margin-bottom: 15px;">
        {' '.join(spans)}
    </div>
    """
    st.markdown(strip_html, unsafe_allow_html=True)


def render_flaw_cards(flaws: list[Any]) -> None:
    """Render interactive cards with explanations and drill recommendations."""
    if not flaws:
        st.success("No delivery flaws detected! Delivery matches the baseline reference closely.")
        return

    st.markdown(f"##### Detected Flaws ({len(flaws)})")

    for f in flaws:
        ftype = f.type.value if hasattr(f.type, "value") else str(f.type)
        fid = f.id if hasattr(f, "id") else f.get("id", "F")
        f_start = float(f.start if hasattr(f, "start") else f.get("start", 0))
        f_end = float(f.end if hasattr(f, "end") else f.get("end", 0))
        sev = float(f.severity if hasattr(f, "severity") else f.get("severity", 0))
        exp = f.explanation if hasattr(f, "explanation") else f.get("explanation", "")
        rec = f.recommendation if hasattr(f, "recommendation") else f.get("recommendation", "")
        p_val = f.participant_value if hasattr(f, "participant_value") else f.get("participant_value")
        b_val = f.baseline_value if hasattr(f, "baseline_value") else f.get("baseline_value")
        unit = f.unit if hasattr(f, "unit") else f.get("unit", "")
        z = f.z_score if hasattr(f, "z_score") else f.get("z_score")

        color = FLAW_COLORS.get(ftype, PRIMARY_TEAL)

        with st.expander(f"[{fid}] {ftype} ({f_start:.1f}s – {f_end:.1f}s) — Severity: {sev:.0%}", expanded=True):
            col_a, col_b = st.columns([3, 1.2])
            with col_a:
                st.markdown(f"**Explanation:** {exp}")
                st.markdown(f"💡 **Recommendation:** {rec}")
            with col_b:
                st.markdown(f"**Metric:** `{f.metric if hasattr(f, 'metric') else f.get('metric', '')}`")
                st.markdown(f"**Participant:** `{p_val} {unit}`")
                st.markdown(f"**Baseline:** `{b_val} {unit}`")
                if z is not None:
                    st.markdown(f"**Z-Score:** `{z}`")
                st.progress(sev)
