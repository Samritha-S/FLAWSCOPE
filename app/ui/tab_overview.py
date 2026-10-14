"""Overview tab rendering rubric score cards, radar chart, flaw counts, and aligned transcript."""

from collections import Counter
from typing import Any, Dict, List

import html
import plotly.graph_objects as go
import streamlit as st

from app.ui.theme import FLAW_COLORS, PALETTE, get_score_color


def render_score_cards(scores: Dict[str, float]) -> None:
    """Render overall score card prominently alongside 5 rubric dimension cards."""
    col_overall, col_dims = st.columns([1.1, 2.9])

    overall_val = scores.get("overall", 0.0)
    overall_color = get_score_color(overall_val)

    with col_overall:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 0.9rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Overall Delivery</div>
                <div class="score-large" style="color: {overall_color};">{overall_val:.1f}</div>
                <div style="font-size: 0.85rem; font-weight: 600; color: {overall_color};">
                    {"Strong Delivery" if overall_val >= 80 else "Needs Refinement" if overall_val >= 60 else "Critical Flaws"}
                </div>
                <div class="score-bar-bg" style="margin-top: 0.8rem;">
                    <div class="score-bar-fill" style="width: {min(max(overall_val, 0.0), 100.0)}%; background-color: {overall_color};"></div>
                </div>
                <div class="caption-sm" style="margin-top: 0.5rem;">Weighted across 5 rubric dimensions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    dimensions = [
        ("pace", "Pace", scores.get("pace", 0.0)),
        ("intonation", "Intonation", scores.get("intonation", 0.0)),
        ("volume", "Volume", scores.get("volume", 0.0)),
        ("pausing", "Pausing", scores.get("pausing", 0.0)),
        ("clarity", "Clarity", scores.get("clarity", 0.0)),
    ]

    with col_dims:
        dim_cols = st.columns(5)
        for i, (key, label, val) in enumerate(dimensions):
            c_color = get_score_color(val)
            with dim_cols[i]:
                st.markdown(
                    f"""
                    <div class="flawscope-card" style="padding: 0.9rem 0.6rem; text-align: center;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #4B5563; text-transform: uppercase;">{label}</div>
                        <div class="score-sub" style="color: {c_color}; margin-top: 0.3rem;">{val:.1f}</div>
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width: {min(max(val, 0.0), 100.0)}%; background-color: {c_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_radar_chart(scores: Dict[str, float]) -> go.Figure:
    """Create a filled Plotly radar chart for the 5 rubric dimensions."""
    categories = ["Pace", "Intonation", "Volume", "Pausing", "Clarity"]
    values = [
        scores.get("pace", 0.0),
        scores.get("intonation", 0.0),
        scores.get("volume", 0.0),
        scores.get("pausing", 0.0),
        scores.get("clarity", 0.0),
    ]

    # Close polygon loop
    r_vals = values + [values[0]]
    theta_vals = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill="toself",
            name="Participant",
            line=dict(color=PALETTE["teal"], width=2.5),
            fillcolor="rgba(15, 157, 138, 0.25)",
            marker=dict(size=6, color=PALETTE["teal"]),
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#6B7280"),
                gridcolor="#E5E7EB",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#1F2A37", family="sans-serif"),
                gridcolor="#E5E7EB",
            ),
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        showlegend=False,
        margin=dict(l=35, r=35, t=25, b=25),
        height=280,
    )
    return fig


def render_flaw_distribution_chart(flaws: List[Dict[str, Any]]) -> go.Figure:
    """Create horizontal bar chart of flaw counts per category."""
    counts = Counter(f["type"] for f in flaws)
    flaw_types = list(FLAW_COLORS.keys())

    y_labels = [ft.replace("_", " ").title() for ft in flaw_types]
    x_counts = [counts.get(ft, 0) for ft in flaw_types]
    colors = [FLAW_COLORS[ft] for ft in flaw_types]

    fig = go.Figure(
        go.Bar(
            x=x_counts,
            y=y_labels,
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=x_counts,
            textposition="auto",
        )
    )

    fig.update_layout(
        margin=dict(l=10, r=20, t=10, b=10),
        height=280,
        xaxis=dict(
            title="Detected Occurrences",
            dtick=1,
            gridcolor="#E5E7EB",
            zeroline=False,
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_aligned_transcript_strip(words: List[Dict[str, Any]], flaws: List[Dict[str, Any]]) -> None:
    """Render aligned transcript with words highlighted by flaw category and hover explanations."""
    # Build word-to-flaw map
    word_flaw_map: Dict[int, Dict[str, Any]] = {}
    for flaw in flaws:
        w_start = flaw.get("word_start_idx", -1)
        w_end = flaw.get("word_end_idx", -1)
        if w_start >= 0 and w_end >= 0:
            for idx in range(w_start, w_end + 1):
                word_flaw_map[idx] = flaw

    # Render Legend
    legend_html = '<div style="margin-bottom: 0.6rem;">'
    for ft, color in FLAW_COLORS.items():
        label = ft.replace("_", " ").title()
        legend_html += f"""
        <span class="legend-item">
            <span class="legend-dot" style="background-color: {color};"></span>
            {label}
        </span>
        """
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # Render words
    words_html = ['<div class="transcript-container">']
    for w in words:
        idx = w.get("idx", 0)
        raw_text = html.escape(w.get("text", ""))
        flaw = word_flaw_map.get(idx)

        if flaw:
            ft = flaw["type"]
            f_color = FLAW_COLORS.get(ft, "#FF6B5B")
            explanation = html.escape(flaw.get("explanation", f"{ft} delivery flaw"))
            bg_color = f"{f_color}25"  # 15% opacity hex
            word_span = (
                f'<span class="transcript-word word-flawed" style="background-color: {bg_color}; '
                f'border-bottom: 2px solid {f_color};" title="{ft}: {explanation}">{raw_text}</span>'
            )
        else:
            word_span = f'<span class="transcript-word">{raw_text}</span>'

        words_html.append(word_span)
        if w.get("has_punct_after", False):
            words_html.append(" ")
        else:
            words_html.append(" ")

    words_html.append("</div>")
    st.markdown("".join(words_html), unsafe_allow_html=True)


def render_top_flaws(flaws: List[Dict[str, Any]]) -> None:
    """Render top 3 critical flaws sorted by severity."""
    sorted_flaws = sorted(flaws, key=lambda f: f.get("severity", 0.0), reverse=True)
    top_3 = sorted_flaws[:3]

    if not top_3:
        st.info("No flaws detected in this performance.")
        return

    cols = st.columns(len(top_3))
    for i, flaw in enumerate(top_3):
        ft = flaw.get("type", "UNKNOWN")
        sev = flaw.get("severity", 0.0)
        s_time = flaw.get("start", 0.0)
        e_time = flaw.get("end", 0.0)
        expl = flaw.get("explanation", "")
        rec = flaw.get("recommendation", "")

        with cols[i]:
            st.markdown(
                f"""
                <div class="flawscope-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span class="flaw-pill pill-{ft}">{ft.replace('_', ' ')}</span>
                        <span style="font-weight: 700; font-size: 0.85rem; color: #374151;">Sev {sev:.2f}</span>
                    </div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #4B5563; margin-bottom: 0.4rem;">
                        {s_time:.1f}s - {e_time:.1f}s ({e_time - s_time:.1f}s)
                    </div>
                    <div style="font-size: 0.85rem; line-height: 1.4; color: #1F2A37; margin-bottom: 0.5rem;">
                        {expl}
                    </div>
                    <div class="caption-sm" style="font-style: italic; border-top: 1px dashed #E5E7EB; padding-top: 0.4rem;">
                        Tip: {rec}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Main render function for the Overview tab."""
    scores = result.get("scores", {})
    words = result.get("words", [])
    flaws = result.get("flaws", [])

    st.subheader("Performance Scorecard")
    render_score_cards(scores)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        st.markdown(
            """
            <div class="flawscope-card" style="height: 100%;">
                <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem; color: #1F2A37;">
                    Rubric Dimension Radar
                </div>
            """,
            unsafe_allow_html=True,
        )
        radar_fig = render_radar_chart(scores)
        st.plotly_chart(radar_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart2:
        st.markdown(
            """
            <div class="flawscope-card" style="height: 100%;">
                <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.5rem; color: #1F2A37;">
                    Where It Went Wrong (Flaw Frequency)
                </div>
            """,
            unsafe_allow_html=True,
        )
        dist_fig = render_flaw_distribution_chart(flaws)
        st.plotly_chart(dist_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("### Top Critical Delivery Flaws")
    render_top_flaws(flaws)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("### Aligned Transcript Strip")
    render_aligned_transcript_strip(words, flaws)
