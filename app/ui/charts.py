"""Plotly charting helpers for time-series overlays, radar charts, and spectrograms."""

from typing import Any
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa

from app.ui.theme import (
    BASELINE_COLOR,
    FLAW_COLORS,
    PARTICIPANT_COLOR,
    PRIMARY_TEAL,
)


def create_timeline_figure(
    curves: Any,
    flaws: list[Any],
    words: list[Any] | None = None,
    height: int = 650,
) -> go.Figure:
    """Create a 3-tier synchronized time-series overlay of Pitch, Loudness, and Rate."""
    t_grid = curves.t if hasattr(curves, "t") else curves.get("t", [])
    p_curves = curves.p if hasattr(curves, "p") else curves.get("p", {})
    b_curves = curves.b if hasattr(curves, "b") else curves.get("b", {})

    p_f0 = getattr(p_curves, "f0_st", None) or p_curves.get("f0_st", [])
    b_f0 = getattr(b_curves, "f0_st", None) or b_curves.get("f0_st", [])
    p_rms = getattr(p_curves, "rms_db_rel", None) or p_curves.get("rms_db_rel", [])
    b_rms = getattr(b_curves, "rms_db_rel", None) or b_curves.get("rms_db_rel", [])
    p_rate = getattr(p_curves, "rate_sps", None) or p_curves.get("rate_sps", [])
    b_rate = getattr(b_curves, "rate_sps", None) or b_curves.get("rate_sps", [])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[
            "<b>Pitch</b> (F0 in semitones relative to median)",
            "<b>Loudness</b> (RMS intensity dB relative to speech level)",
            "<b>Speech Rate</b> (syllables per second)",
        ],
    )

    # 1. Pitch
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=b_f0,
            mode="lines",
            name="Baseline F0",
            line=dict(color=BASELINE_COLOR, dash="dash", width=1.5),
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=p_f0,
            mode="lines",
            name="Participant F0",
            line=dict(color=PARTICIPANT_COLOR, width=2.0),
            hoverinfo="x+y+name",
        ),
        row=1,
        col=1,
    )

    # 2. Loudness
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=b_rms,
            mode="lines",
            name="Baseline Loudness",
            line=dict(color=BASELINE_COLOR, dash="dash", width=1.5),
            showlegend=False,
            hoverinfo="x+y+name",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=p_rms,
            mode="lines",
            name="Participant Loudness",
            line=dict(color=PARTICIPANT_COLOR, width=2.0),
            showlegend=False,
            hoverinfo="x+y+name",
        ),
        row=2,
        col=1,
    )

    # 3. Speech Rate
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=b_rate,
            mode="lines",
            line_shape="hv",
            name="Baseline Rate",
            line=dict(color=BASELINE_COLOR, dash="dash", width=1.5),
            showlegend=False,
            hoverinfo="x+y+name",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=p_rate,
            mode="lines",
            line_shape="hv",
            name="Participant Rate",
            line=dict(color=PARTICIPANT_COLOR, width=2.0),
            showlegend=False,
            hoverinfo="x+y+name",
        ),
        row=3,
        col=1,
    )

    # Shaded Flaw Regions
    for f in flaws:
        ftype = f.type.value if hasattr(f.type, "value") else str(f.type)
        f_start = float(f.start if hasattr(f, "start") else f.get("start", 0))
        f_end = float(f.end if hasattr(f, "end") else f.get("end", 0))
        fid = f.id if hasattr(f, "id") else f.get("id", "")
        f_col = FLAW_COLORS.get(ftype, PRIMARY_TEAL)

        # Highlight across all 3 subplots
        for r in [1, 2, 3]:
            fig.add_vrect(
                x0=f_start,
                x1=f_end,
                fillcolor=f_col,
                opacity=0.20,
                layer="below",
                line=dict(color=f_col, width=1, dash="dot"),
                row=r,
                col=1,
                annotation_text=f"{fid}: {ftype}" if r == 1 else "",
                annotation_position="top left",
                annotation_font=dict(size=10, color=f_col),
            )

    fig.update_layout(
        height=height,
        margin=dict(l=50, r=30, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F4FAF9",
    )
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Semitones", row=1, col=1)
    fig.update_yaxes(title_text="dB", row=2, col=1)
    fig.update_yaxes(title_text="Syllables/s", row=3, col=1)

    return fig


def create_radar_chart(scores: Any) -> go.Figure:
    """Create a 5-dimension rubric radar chart."""
    categories = ["Pace", "Intonation", "Volume", "Pausing", "Clarity"]

    if hasattr(scores, "pace"):
        vals = [scores.pace, scores.intonation, scores.volume, scores.pausing, scores.clarity]
    else:
        vals = [
            scores.get("pace", 100),
            scores.get("intonation", 100),
            scores.get("volume", 100),
            scores.get("pausing", 100),
            scores.get("clarity", 100),
        ]

    # Close the radar loop
    r_vals = vals + [vals[0]]
    theta_cats = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_vals,
            theta=theta_cats,
            fill="toself",
            fillcolor="rgba(15, 157, 138, 0.25)",
            line=dict(color=PRIMARY_TEAL, width=2.5),
            name="Rubric Performance",
            hoverinfo="r+theta",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                linecolor="#CCCCCC",
                gridcolor="#E5E7EB",
            ),
            angularaxis=dict(gridcolor="#E5E7EB", linecolor="#CCCCCC"),
        ),
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def create_spectrogram_figure(y: np.ndarray, sr: int, flaws: list[Any]) -> go.Figure:
    """Generate a log-mel spectrogram visualization with flaw region boundaries."""
    # Compute mel spectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=64)
    log_mel = librosa.power_to_db(mels, ref=np.max)

    duration = len(y) / sr
    t_axis = np.linspace(0, duration, log_mel.shape[1])
    mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=0, fmax=sr // 2)

    fig = go.Figure(
        data=go.Heatmap(
            z=log_mel,
            x=t_axis,
            y=mel_freqs,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )

    for f in flaws:
        ftype = f.type.value if hasattr(f.type, "value") else str(f.type)
        f_start = float(f.start if hasattr(f, "start") else f.get("start", 0))
        f_end = float(f.end if hasattr(f, "end") else f.get("end", 0))
        fid = f.id if hasattr(f, "id") else f.get("id", "")
        f_col = FLAW_COLORS.get(ftype, PRIMARY_TEAL)

        fig.add_vrect(
            x0=f_start,
            x1=f_end,
            line=dict(color=f_col, width=2, dash="dash"),
            annotation_text=f"{fid}",
            annotation_position="top left",
            annotation_font=dict(color="#FFFFFF", size=11),
        )

    fig.update_layout(
        title="Participant Log-Mel Spectrogram",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        height=380,
        margin=dict(l=50, r=30, t=50, b=40),
        paper_bgcolor="#F4FAF9",
    )
    return fig
