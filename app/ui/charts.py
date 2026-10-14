"""Plotly charting helpers for synchronized time-series overlays, radar charts, and acoustic visualizers."""

from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.ui.theme import BASELINE_COLOR, FLAW_COLORS, PALETTE, PARTICIPANT_COLOR


def build_timeline_figure(
    result: Dict[str, Any],
    x_range: Optional[Tuple[float, float]] = None,
) -> go.Figure:
    """Build synchronized 3-row time-series figure for benchmark or reference mode."""
    meta = result.get("meta", {})
    is_benchmark = (meta.get("mode") == "benchmark")

    curves = result.get("curves", {})
    t = curves.get("t", [])
    p_curves = curves.get("p", {})
    b_curves = curves.get("b", {})

    p_f0 = p_curves.get("f0_st", [])
    p_f0_var = p_curves.get("f0_std_roll_st") or p_f0
    b_f0 = b_curves.get("f0_st", [])
    p_rms = p_curves.get("rms_db_rel", [])
    b_rms = b_curves.get("rms_db_rel", [])
    p_rate = p_curves.get("rate_sps", [])
    b_rate = b_curves.get("rate_sps", [])

    words = result.get("words", [])
    flaws = result.get("flaws", [])
    bands = result.get("bands", {})

    # Use WebGL if array is large
    use_gl = len(t) > 5000
    ScatterClass = go.Scattergl if use_gl else go.Scatter

    participant_name = "Your delivery" if is_benchmark else "Participant"

    subplot_titles = (
        ("Pitch variation (curves.p.f0_std_roll_st)", "Loudness (rms_db_rel)", "Speech rate")
        if is_benchmark
        else (
            "Pitch (F0, semitones vs speaker median)",
            "Loudness (RMS intensity, dB rel)",
            "Speech Rate (syllables/s, step cadence)",
        )
    )

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=subplot_titles,
    )

    # --- Row 1: Pitch / Pitch Variation ---
    r1_y = p_f0_var if is_benchmark else p_f0
    r1_hover = (
        "Pitch Std: %{y:.2f} st<extra></extra>"
        if is_benchmark
        else "Participant F0: %{y:.1f} st<extra></extra>"
    )

    fig.add_trace(
        ScatterClass(
            x=t,
            y=r1_y,
            name=participant_name,
            line=dict(color=PARTICIPANT_COLOR, width=1.6),
            connectgaps=False,
            showlegend=True,
            hovertemplate=r1_hover,
        ),
        row=1,
        col=1,
    )

    if not is_benchmark:
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
            name=participant_name,
            line=dict(color=PARTICIPANT_COLOR, width=1.6),
            connectgaps=False,
            showlegend=False,
            hovertemplate=f"{participant_name} RMS: %{{y:.1f}} dB<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if not is_benchmark:
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
            name=participant_name,
            line=dict(color=PARTICIPANT_COLOR, width=1.6, shape="hv"),
            connectgaps=False,
            showlegend=False,
            hovertemplate=f"{participant_name} Rate: %{{y:.1f}} sps<extra></extra>",
        ),
        row=3,
        col=1,
    )

    if not is_benchmark:
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

    # --- Benchmark Mode: Shaded Typical Bands (green #3BB273 at 12% opacity) ---
    if is_benchmark:
        pitch_band = bands.get("pitch_variation") or bands.get("f0_std_roll_st") or [1.4, 3.2]
        loud_band = bands.get("loudness") or bands.get("rms_db_rel") or [-4.0, 3.0]
        rate_band = bands.get("speech_rate") or bands.get("rate_sps") or [3.2, 5.2]

        # Row 1 Shaded Band
        fig.add_hrect(
            y0=pitch_band[0],
            y1=pitch_band[1],
            fillcolor="#3BB273",
            opacity=0.12,
            line_width=0,
            row=1,
            col=1,
        )
        # Row 2 Shaded Band
        fig.add_hrect(
            y0=loud_band[0],
            y1=loud_band[1],
            fillcolor="#3BB273",
            opacity=0.12,
            line_width=0,
            row=2,
            col=1,
        )
        # Row 3 Shaded Band
        fig.add_hrect(
            y0=rate_band[0],
            y1=rate_band[1],
            fillcolor="#3BB273",
            opacity=0.12,
            line_width=0,
            row=3,
            col=1,
        )

        # Legend entry for Typical Range
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name="Typical range (strong speakers)",
                marker=dict(size=10, color="rgba(59, 178, 115, 0.4)", symbol="square"),
                showlegend=True,
            ),
            row=1,
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
    fig.update_yaxes(title_text="Variation (st)" if is_benchmark else "Semitones", row=1, col=1)
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
