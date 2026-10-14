"""Plotly charting helpers for synchronized time-series overlays, radar charts, and acoustic visualizers."""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui.theme import BASELINE_COLOR, FLAW_COLORS, PALETTE, PARTICIPANT_COLOR


def build_timeline_figure(
    result: Dict[str, Any],
    x_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Build synchronized 3-row time-series figure comparing participant vs baseline."""
    curves = result.get("curves", {})
    t = curves.get("t", [])
    p_curves = curves.get("p", {})
    b_curves = curves.get("b", {})

    p_f0 = p_curves.get("f0_st", [])
    b_f0 = b_curves.get("f0_st", [])
    p_rms = p_curves.get("rms_db_rel", [])
    b_rms = b_curves.get("rms_db_rel", [])
    p_rate = p_curves.get("rate_sps", [])
    b_rate = b_curves.get("rate_sps", [])

    words = result.get("words", [])
    flaws = result.get("flaws", [])

    # Use WebGL if array is large
    use_gl = len(t) > 5000
    ScatterClass = go.Scattergl if use_gl else go.Scatter

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Pitch (F0, semitones vs speaker median)",
            "Loudness (RMS intensity, dB rel)",
            "Speech Rate (syllables/s, step cadence)",
        ),
    )

    # --- Row 1: Pitch F0 ---
    fig.add_trace(
        ScatterClass(
            x=t,
            y=p_f0,
            name="Participant",
            line=dict(color=PARTICIPANT_COLOR, width=1.6),
            connectgaps=False,
            showlegend=True,
            hovertemplate="Participant F0: %{y:.1f} st<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        ScatterClass(
            x=t,
            y=b_f0,
            name="Baseline (Warped)",
            line=dict(color=BASELINE_COLOR, width=1.6, dash="dash"),
            connectgaps=False,
            showlegend=True,
            hovertemplate="Baseline F0: %{y:.1f} st<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Word center hover anchors on Row 1
    word_x = []
    word_text = []
    for w in words:
        p_s = w.get("p_start")
        p_e = w.get("p_end")
        if p_s is not None and p_e is not None:
            mid = (p_s + p_e) / 2.0
            word_x.append(mid)
            word_text.append(f"Word: {w.get('text', '')} ({p_s:.2f}s - {p_e:.2f}s)")

    if word_x:
        fig.add_trace(
            go.Scatter(
                x=word_x,
                y=[0.0] * len(word_x),
                text=word_text,
                mode="markers",
                marker=dict(opacity=0.0, size=1),
                hoverinfo="text",
                name="Words",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # --- Row 2: Loudness RMS ---
    fig.add_trace(
        ScatterClass(
            x=t,
            y=p_rms,
            name="Participant",
            line=dict(color=PARTICIPANT_COLOR, width=1.6),
            connectgaps=False,
            showlegend=False,
            hovertemplate="Participant RMS: %{y:.1f} dB<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        ScatterClass(
            x=t,
            y=b_rms,
            name="Baseline (Warped)",
            line=dict(color=BASELINE_COLOR, width=1.6, dash="dash"),
            connectgaps=False,
            showlegend=False,
            hovertemplate="Baseline RMS: %{y:.1f} dB<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # --- Row 3: Speech Rate ---
    fig.add_trace(
        ScatterClass(
            x=t,
            y=p_rate,
            name="Participant",
            line=dict(color=PARTICIPANT_COLOR, width=1.6, shape="hv"),
            connectgaps=False,
            showlegend=False,
            hovertemplate="Participant Rate: %{y:.1f} sps<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        ScatterClass(
            x=t,
            y=b_rate,
            name="Baseline (Warped)",
            line=dict(color=BASELINE_COLOR, width=1.6, dash="dash", shape="hv"),
            connectgaps=False,
            showlegend=False,
            hovertemplate="Baseline Rate: %{y:.1f} sps<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # --- Flaw regions as translucent vrects spanning all rows ---
    for flaw in flaws:
        ft = flaw.get("type", "UNKNOWN")
        color = FLAW_COLORS.get(ft, PALETTE["coral"])
        s = float(flaw.get("start", 0.0))
        e = float(flaw.get("end", 0.0))

        # Ensure minimum visual width for short pauses
        if ft == "LONG_PAUSE" and (e - s) < 0.15:
            e = s + 0.15

        label_txt = ft.replace("_", " ").title()
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=color,
            opacity=0.22,
            line_width=0,
            annotation_text=label_txt,
            annotation_position="top left",
            annotation=dict(font_size=10, font_color=PALETTE["ink"]),
        )

    # Configure range slider on bottom axis
    fig.update_xaxes(
        row=3,
        col=1,
        rangeslider=dict(visible=True, thickness=0.07, bgcolor="#F4FAF9"),
        title_text="Time on Participant Timeline (seconds)",
    )

    # Grid and axis styling
    for r in (1, 2, 3):
        fig.update_xaxes(
            row=r,
            col=1,
            gridcolor="#E5E7EB",
            zeroline=False,
        )
        fig.update_yaxes(
            row=r,
            col=1,
            gridcolor="#E5E7EB",
            zeroline=False,
        )

    # Specific y-axis labels
    fig.update_yaxes(title_text="Semitones", row=1, col=1)
    fig.update_yaxes(title_text="Rel dB", row=2, col=1)
    fig.update_yaxes(title_text="Syllables/s", row=3, col=1)

    # Zoom window if provided
    if x_range is not None:
        fig.update_xaxes(range=[max(0.0, x_range[0]), x_range[1]])
    elif t:
        fig.update_xaxes(range=[0.0, t[-1]])

    fig.update_layout(
        height=640,
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=11, color="#1F2A37"),
        ),
        margin=dict(l=50, r=30, t=50, b=30),
    )

    return fig
