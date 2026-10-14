"""Timeline tab module featuring synchronized acoustic overlays, flaw zoom, and audio excerpt comparison."""

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import streamlit as st

from app.ui.charts import build_timeline_figure
from app.ui.theme import FLAW_COLORS


def slice_to_wav_bytes(y: np.ndarray, sr: int, start_s: float, end_s: float) -> bytes:
    """Slice an audio array between start_s and end_s and return WAV file bytes."""
    s_idx = max(0, int(start_s * sr))
    e_idx = min(len(y), int(end_s * sr))
    if e_idx <= s_idx:
        chunk = np.zeros(int(0.2 * sr), dtype=np.float32)
    else:
        chunk = y[s_idx:e_idx]

    bio = io.BytesIO()
    sf.write(bio, chunk, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def find_baseline_interval(flaw: Dict[str, Any], words: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Retrieve corresponding baseline word boundaries for a given flaw."""
    w_start = flaw.get("word_start_idx", -1)
    w_end = flaw.get("word_end_idx", -1)

    b_s = None
    b_e = None

    if w_start >= 0 and w_end >= 0:
        for idx in range(w_start, w_end + 1):
            if idx < len(words):
                w = words[idx]
                if w.get("b_start") is not None and b_s is None:
                    b_s = float(w["b_start"])
                if w.get("b_end") is not None:
                    b_e = float(w["b_end"])

    if b_s is not None and b_e is not None:
        return b_s, b_e

    # Fallback to participant timestamps
    return float(flaw.get("start", 0.0)), float(flaw.get("end", 0.0))


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Timeline tab interface."""
    try:
        flaws = result.get("flaws", [])
        words = result.get("words", [])

        # Retrieve audio arrays from context
        p_audio = ctx.get("participant_audio")
        b_audio = ctx.get("baseline_audio")

        # Full-length Audio Players
        st.markdown("### Full Performance Playback")
        col_full_p, col_full_b = st.columns(2)

        with col_full_p:
            st.markdown("**Participant Delivery (Evaluated Recording)**")
            if p_audio and len(p_audio[0]) > 0:
                p_wav = slice_to_wav_bytes(p_audio[0], p_audio[1], 0.0, len(p_audio[0]) / p_audio[1])
                st.audio(p_wav, format="audio/wav")
            else:
                st.caption("Participant audio waveform loaded from metadata.")

        with col_full_b:
            st.markdown("**Baseline Counterpoint (Reference Delivery)**")
            if b_audio and len(b_audio[0]) > 0:
                b_wav = slice_to_wav_bytes(b_audio[0], b_audio[1], 0.0, len(b_audio[0]) / b_audio[1])
                st.audio(b_wav, format="audio/wav")
            else:
                st.caption("Baseline audio waveform loaded from metadata.")

        st.markdown("---")

        # Flaw focus selector
        col_select, col_info = st.columns([1.8, 1.2])

        flaw_options = ["None (Full Timeline)"]
        flaw_map = {}
        for f in flaws:
            fid = f.get("id", "F")
            ft = f.get("type", "UNKNOWN")
            s = float(f.get("start", 0.0))
            e = float(f.get("end", 0.0))
            sev = float(f.get("severity", 0.0))
            label = f"{fid} | {ft} | {s:.1f}-{e:.1f} s | severity {sev:.2f}"
            flaw_options.append(label)
            flaw_map[label] = f

        with col_select:
            selected_label = st.selectbox(
                "🔍 Focus on a Flaw (Zooms Timeline & Excerpt Playback)",
                options=flaw_options,
                index=0,
                help="Select a specific flaw to zoom into its time region and hear matching audio clips.",
            )

        selected_flaw = flaw_map.get(selected_label)
        x_range = None
        if selected_flaw:
            f_s = float(selected_flaw.get("start", 0.0))
            f_e = float(selected_flaw.get("end", 0.0))
            x_range = (f_s - 1.5, f_e + 1.5)

        # Build and render Plotly figure
        fig = build_timeline_figure(result, x_range=x_range)
        st.plotly_chart(fig, use_container_width=True)

        # Flaw Detail Card & Excerpt Playback if focused
        if selected_flaw:
            ft = selected_flaw.get("type", "UNKNOWN")
            s = float(selected_flaw.get("start", 0.0))
            e = float(selected_flaw.get("end", 0.0))
            sev = float(selected_flaw.get("severity", 0.0))
            expl = selected_flaw.get("explanation", "")
            rec = selected_flaw.get("recommendation", "")
            metric = selected_flaw.get("metric", "")
            unit = selected_flaw.get("unit", "")
            p_val = selected_flaw.get("participant_value")
            b_val = selected_flaw.get("baseline_value")
            z = selected_flaw.get("z_score")

            st.markdown(
                f"""
                <div class="flawscope-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div>
                            <span class="flaw-pill pill-{ft}">{ft.replace('_', ' ')}</span>
                            <span style="font-weight: 700; color: #1F2A37; font-size: 1rem;">{selected_flaw.get('id', '')}</span>
                        </div>
                        <span style="font-weight: 700; font-size: 0.9rem; color: #4B5563;">Severity: {sev:.2f} / 1.00</span>
                    </div>
                    <div style="font-size: 0.9rem; font-weight: 600; color: #374151; margin-bottom: 0.4rem;">
                        Time Interval: {s:.2f}s - {e:.2f}s (Duration: {e - s:.2f}s)
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.5; color: #1F2A37; margin-bottom: 0.5rem;">
                        {expl}
                    </div>
                    <div class="caption-sm" style="font-style: italic; border-top: 1px dashed #E5E7EB; padding-top: 0.4rem; color: #0F9D8A;">
                        Tip: {rec}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Sliced Excerpt Audio Players
            st.markdown("#### Side-by-Side Excerpt Audio Counterpoint")
            col_clip_p, col_clip_b = st.columns(2)

            # Participant slice: [s - 0.3, e + 0.3]
            p_s = max(0.0, s - 0.3)
            p_e = e + 0.3

            # Baseline slice: [b_s - 0.3, b_e + 0.3]
            b_start, b_end = find_baseline_interval(selected_flaw, words)
            b_s = max(0.0, b_start - 0.3)
            b_e = b_end + 0.3

            with col_clip_p:
                st.markdown(f"**Participant Flawed Segment** ({p_s:.1f}s - {p_e:.1f}s)")
                if p_audio and len(p_audio[0]) > 0:
                    clip_bytes = slice_to_wav_bytes(p_audio[0], p_audio[1], p_s, p_e)
                    st.audio(clip_bytes, format="audio/wav")
                else:
                    st.info("Upload or select real audio to hear live sliced playback.")

            with col_clip_b:
                st.markdown(f"**Matching Baseline Excerpt** ({b_s:.1f}s - {b_e:.1f}s)")
                if b_audio and len(b_audio[0]) > 0:
                    clip_bytes_b = slice_to_wav_bytes(b_audio[0], b_audio[1], b_s, b_e)
                    st.audio(clip_bytes_b, format="audio/wav")
                else:
                    st.info("Upload or select real audio to hear live sliced playback.")

    except Exception as exc:
        st.error(f"Error rendering Timeline tab: {exc}")
