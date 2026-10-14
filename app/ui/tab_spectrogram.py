"""Spectrogram and MFCC tab module providing time-frequency and phonetic visual evidence."""

from typing import Any, Dict, List

import librosa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.ui import adapter
from app.ui.theme import FLAW_COLORS, PALETTE


def render_spectrogram_heatmap(
    log_mel: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    flaws: List[Dict[str, Any]],
) -> go.Figure:
    """Render log-mel time-frequency spectrogram heatmap with overlaid flaw bounding boxes."""
    fig = go.Figure(
        data=go.Heatmap(
            z=log_mel,
            x=times,
            y=freqs,
            colorscale="Cividis",
            colorbar=dict(title="dB", thickness=12, len=0.85),
            hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Intensity: %{z:.1f} dB<extra></extra>",
        )
    )

    # Add flaw bounding boxes
    for flaw in flaws:
        ft = flaw.get("type", "UNKNOWN")
        color = FLAW_COLORS.get(ft, PALETTE["coral"])
        s = float(flaw.get("start", 0.0))
        e = float(flaw.get("end", 0.0))
        if ft == "LONG_PAUSE" and (e - s) < 0.15:
            e = s + 0.15

        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=color,
            opacity=0.25,
            line=dict(color=color, width=1.5),
            annotation_text=ft.replace("_", " ").title(),
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#FFFFFF"),
        )

    fig.update_layout(
        title=dict(text="Participant Log-Mel Spectrogram (64 Bands)", font=dict(size=14, color="#1F2A37")),
        xaxis=dict(title="Time on Participant Timeline (seconds)", gridcolor="#E5E7EB"),
        yaxis=dict(title="Frequency (Hz)", gridcolor="#E5E7EB"),
        height=320,
        margin=dict(l=50, r=30, t=40, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def render_mfcc_heatmap(
    mfccs: np.ndarray,
    times: np.ndarray,
    flaws: List[Dict[str, Any]],
) -> go.Figure:
    """Render 12-band MFCC (c1-c12) coefficient heatmap."""
    coeff_labels = [f"c{i}" for i in range(1, 13)]

    fig = go.Figure(
        data=go.Heatmap(
            z=mfccs,
            x=times,
            y=coeff_labels,
            colorscale="RdYlBu",
            colorbar=dict(title="Coeff", thickness=12, len=0.85),
            hovertemplate="Time: %{x:.2f}s<br>Coeff: %{y}<br>Value: %{z:.2f}<extra></extra>",
        )
    )

    # Add flaw bounding boxes
    for flaw in flaws:
        ft = flaw.get("type", "UNKNOWN")
        color = FLAW_COLORS.get(ft, PALETTE["coral"])
        s = float(flaw.get("start", 0.0))
        e = float(flaw.get("end", 0.0))
        if ft == "LONG_PAUSE" and (e - s) < 0.15:
            e = s + 0.15

        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=color,
            opacity=0.22,
            line=dict(color=color, width=1.2),
        )

    fig.update_layout(
        title=dict(text="Participant MFCC Trajectories (c1 to c12)", font=dict(size=14, color="#1F2A37")),
        xaxis=dict(title="Time on Participant Timeline (seconds)", gridcolor="#E5E7EB"),
        yaxis=dict(title="MFCC Coefficient", gridcolor="#E5E7EB"),
        height=260,
        margin=dict(l=50, r=30, t=40, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Spectrogram and MFCC tab interface."""
    try:
        flaws = result.get("flaws", [])
        p_audio = ctx.get("participant_audio")

        # Load audio if not in context
        y = None
        sr = 16000
        if p_audio and len(p_audio[0]) > 0:
            y, sr = p_audio
        else:
            p_file = result.get("meta", {}).get("participant_file", "")
            if p_file:
                y, sr = adapter.load_audio_for_ui(p_file)

        # Fallback harmonic synthesis for mock test if no audio available
        if y is None or len(y) == 0:
            dur = float(result.get("meta", {}).get("duration_p", 24.0))
            t_synth = np.linspace(0, dur, int(dur * sr), endpoint=False)
            y = (0.3 * np.sin(2 * np.pi * 180 * t_synth) + 0.15 * np.sin(2 * np.pi * 360 * t_synth)).astype(np.float32)

        st.subheader("Time-Frequency & MFCC Decomposition")
        st.markdown(
            "Spectral heatmaps capture acoustic energy distributions and phonetic transitions, revealing delivery flaws directly in the frequency domain."
        )

        n_fft = 512
        hop_length = 160
        n_mels = 64

        # 1. Log-Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        log_mel = librosa.power_to_db(melspec, ref=np.max)
        t_mels = np.arange(log_mel.shape[1]) * (hop_length / sr)
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)

        fig_spec = render_spectrogram_heatmap(log_mel, t_mels, mel_freqs, flaws)
        st.plotly_chart(fig_spec, use_container_width=True)

        # 2. MFCC Features (c1 to c12)
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
        )[1:13]  # Coefficients 1 to 12
        t_mfcc = np.arange(mfccs.shape[1]) * (hop_length / sr)

        fig_mfcc = render_mfcc_heatmap(mfccs, t_mfcc, flaws)
        st.plotly_chart(fig_mfcc, use_container_width=True)

        # 3. Visual Acoustic Guidance Caption
        st.markdown(
            """
            <div class="flawscope-card" style="border-top: 3px solid #0F9D8A; padding: 1rem;">
                <div style="font-weight: 700; color: #1F2A37; margin-bottom: 0.4rem;">
                    Visual Acoustic Patterns to Observe:
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.88rem; line-height: 1.6; color: #4B5563;">
                    <li><strong>Monotone Delivery:</strong> Pitch harmonics form rigidly horizontal, unvarying bands across time instead of natural musical curves.</li>
                    <li><strong>Low Volume:</strong> Visible dark valleys with low spectral energy across all frequency bands.</li>
                    <li><strong>Unclear Articulation:</strong> Upper formants and high-frequency energy (> 2.5 kHz) appear washed out or blurred by ambient noise.</li>
                    <li><strong>Rushed / Sluggish Cadence:</strong> Formant transitions and syllable bursts appear either densely compressed (Too Fast) or overly prolonged (Too Slow).</li>
                    <li><strong>Long Pauses:</strong> Extended dead-silence intervals with zero harmonic structure between speech bursts.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as exc:
        st.error(f"Error rendering Spectrogram tab: {exc}")
