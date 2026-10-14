"""Flaws panel tab module featuring sortable table, expandable cards, metric comparisons, audio clips, and data export."""

import io
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import soundfile as sf
import streamlit as st

from app.ui.tab_timeline import find_baseline_interval, slice_to_wav_bytes
from app.ui.theme import FLAW_COLORS


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Flaws tab interface."""
    try:
        flaws = result.get("flaws", [])
        words = result.get("words", [])
        p_audio = ctx.get("participant_audio")
        b_audio = ctx.get("baseline_audio")

        st.subheader("Detected Delivery Flaws & Actionable Guidance")
        st.markdown(
            "Sort, inspect, and listen to every temporally bounded flaw detected against baseline counterpoints."
        )

        if not flaws:
            st.info("No delivery flaws detected in this performance! Excellent delivery.")
            return

        # Prepare Table Data
        table_rows = []
        for f in flaws:
            table_rows.append({
                "ID": f.get("id", ""),
                "Type": f.get("type", ""),
                "Start (s)": round(float(f.get("start", 0.0)), 2),
                "End (s)": round(float(f.get("end", 0.0)), 2),
                "Severity": round(float(f.get("severity", 0.0)), 2),
                "Metric": f.get("metric", ""),
                "Participant": f.get("participant_value"),
                "Baseline": f.get("baseline_value"),
                "Delta (%)": round(float(f.get("delta_pct", 0.0)), 1) if f.get("delta_pct") is not None else None,
                "Z-score": round(float(f.get("z_score", 0.0)), 2) if f.get("z_score") is not None else None,
            })

        df = pd.DataFrame(table_rows)

        # Sortable Dataframe
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

        # Download Buttons
        col_dl1, col_dl2 = st.columns([1, 1])
        with col_dl1:
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Full Report (JSON)",
                data=json_str,
                file_name="flawscope_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Flaws Table (CSV)",
                data=csv_str,
                file_name="flawscope_flaws.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("---")
        st.markdown("### Flaw Deep-Dive & Excerpt Playback")

        # Per flaw expanders
        for f in flaws:
            fid = f.get("id", "F")
            ft = f.get("type", "UNKNOWN")
            s = float(f.get("start", 0.0))
            e = float(f.get("end", 0.0))
            sev = float(f.get("severity", 0.0))
            expl = f.get("explanation", "")
            rec = f.get("recommendation", "")
            metric = f.get("metric", "")
            unit = f.get("unit", "")
            p_val = f.get("participant_value")
            b_val = f.get("baseline_value")
            delta_pct = f.get("delta_pct")
            z = f.get("z_score")

            expander_title = f"{fid} | {ft.replace('_', ' ')} [{s:.2f}s - {e:.2f}s] - Severity: {sev:.2f}"

            with st.expander(expander_title, expanded=False):
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.6rem;">
                        <span class="flaw-pill pill-{ft}">{ft.replace('_', ' ')}</span>
                        <strong>Time Interval:</strong> {s:.2f}s - {e:.2f}s (Duration: {e - s:.2f}s)
                    </div>
                    <div style="font-size: 0.95rem; line-height: 1.5; color: #1F2A37; margin-bottom: 0.6rem;">
                        <strong>Explanation:</strong> {expl}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metric table
                metric_df = pd.DataFrame([{
                    "Metric Name": f"{metric} ({unit})",
                    "Participant Value": p_val if p_val is not None else "N/A",
                    "Baseline Value": b_val if b_val is not None else "N/A",
                    "Delta (%)": f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A",
                    "Z-score": f"{z:+.2f}" if z is not None else "N/A",
                }])
                st.table(metric_df)

                # Actionable Recommendation
                st.markdown(
                    f"""
                    <div style="background-color: #F4FAF9; border-left: 3px solid #0F9D8A; padding: 0.6rem 0.8rem; border-radius: 4px; margin-bottom: 0.8rem;">
                        <strong style="color: #0F9D8A;">Coaching Recommendation:</strong> {rec}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Audio players for this flaw
                col_ex_p, col_ex_b = st.columns(2)

                p_s = max(0.0, s - 0.3)
                p_e = e + 0.3
                b_start, b_end = find_baseline_interval(f, words)
                b_s = max(0.0, b_start - 0.3)
                b_e = b_end + 0.3

                with col_ex_p:
                    st.caption(f"**Participant Excerpt** [{p_s:.1f}s - {p_e:.1f}s]")
                    if p_audio and len(p_audio[0]) > 0:
                        wav_bytes = slice_to_wav_bytes(p_audio[0], p_audio[1], p_s, p_e)
                        st.audio(wav_bytes, format="audio/wav")
                    else:
                        st.caption("Audio array available when evaluated with media.")

                with col_ex_b:
                    st.caption(f"**Baseline Excerpt** [{b_s:.1f}s - {b_e:.1f}s]")
                    if b_audio and len(b_audio[0]) > 0:
                        wav_bytes_b = slice_to_wav_bytes(b_audio[0], b_audio[1], b_s, b_e)
                        st.audio(wav_bytes_b, format="audio/wav")
                    else:
                        st.caption("Audio array available when evaluated with media.")

    except Exception as exc:
        st.error(f"Error rendering Flaws tab: {exc}")
