"""Streamlit web dashboard entry point for FlawScope."""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

try:
    import config
    DEFAULT_Z_THRESH = getattr(config, "Z_THRESH", 1.5)
except Exception:
    DEFAULT_Z_THRESH = 1.5

from app.ui import (
    adapter,
    tab_flaws,
    tab_method,
    tab_overview,
    tab_spectrogram,
    tab_timeline,
    tab_validation,
    theme,
)


def init_session_state() -> None:
    """Initialize session state keys to maintain analysis results across reruns."""
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "baseline_audio" not in st.session_state:
        st.session_state["baseline_audio"] = None
    if "participant_audio" not in st.session_state:
        st.session_state["participant_audio"] = None
    if "active_pair_id" not in st.session_state:
        st.session_state["active_pair_id"] = None
    if "comparison_mode" not in st.session_state:
        st.session_state["comparison_mode"] = "Strict (same speaker or ideal reference)"
    if "re_analyze" not in st.session_state:
        st.session_state["re_analyze"] = False
    if "last_inputs" not in st.session_state:
        st.session_state["last_inputs"] = {}


def _switch_to_style_tolerant() -> None:
    """Callback to switch comparison mode to style-tolerant and trigger re-analysis."""
    st.session_state["comparison_mode"] = "Style-tolerant (different speaker)"
    st.session_state["pending_mode_change"] = "Style-tolerant (different speaker)"
    st.session_state["re_analyze"] = True


def save_uploaded_file(uploaded_file, suffix: str = ".wav") -> str:
    """Save an uploaded file to a temporary location and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def main() -> None:
    st.set_page_config(
        page_title="FlawScope - Spoken Delivery Analytics",
        page_icon=":studio_microphone:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    theme.inject_css()
    init_session_state()

    if "pending_mode_change" in st.session_state:
        st.session_state["comparison_mode"] = st.session_state.pop("pending_mode_change")

    is_mock_mode = os.environ.get("FLAWSCOPE_MOCK") == "1"

    # --- Sidebar Controls ---
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
                <h2 style="margin: 0; font-size: 1.4rem; color: {theme.PALETTE['ink']};">FlawScope</h2>
                {f'<span class="mock-chip">MOCK DATA</span>' if is_mock_mode else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Audio Counterpoint Selection")

        demo_pairs = adapter.list_demo_pairs()
        pair_options = ["Upload my own"]
        if demo_pairs:
            pair_options = [p["pair_id"] for p in demo_pairs] + ["Upload my own"]
        else:
            st.info("No demo pairs found in dataset/pairs.json.")

        selected_pair_option = st.selectbox(
            "Select Evaluation Target",
            options=pair_options,
            index=0 if demo_pairs else 0,
            help="Choose a pre-generated contrastive flaw pair or upload your own recordings.",
        )

        baseline_path: Optional[str] = None
        participant_path: Optional[str] = None
        transcript_text: str = ""

        if selected_pair_option != "Upload my own" and demo_pairs:
            matched = next((p for p in demo_pairs if p["pair_id"] == selected_pair_option), None)
            if matched:
                baseline_path = str(PROJECT_ROOT / matched["baseline"])
                participant_path = str(PROJECT_ROOT / matched["participant"])
                tr_file = PROJECT_ROOT / matched["transcript"]
                if tr_file.exists():
                    transcript_text = tr_file.read_text(encoding="utf-8").strip()

                kind = matched.get("kind", "mixed").upper()
                level = matched.get("level", "-")
                st.caption(f"**Kind**: `{kind}` | **Severity**: `Level {level}`")
        else:
            st.markdown("#### Upload Audio and Transcript")
            b_upload = st.file_uploader("Baseline Audio (Clean Reference)", type=["wav", "mp3", "m4a"], key="b_up")
            p_upload = st.file_uploader("Participant Audio (Target)", type=["wav", "mp3", "m4a"], key="p_up")
            t_upload = st.file_uploader("Transcript Text File (.txt)", type=["txt"], key="t_up")
            t_input = st.text_area("Or Paste Transcript Here", height=90, key="t_in")

            if b_upload:
                baseline_path = save_uploaded_file(b_upload, suffix=Path(b_upload.name).suffix)
            if p_upload:
                participant_path = save_uploaded_file(p_upload, suffix=Path(p_upload.name).suffix)
            if t_upload:
                transcript_text = t_upload.getvalue().decode("utf-8", errors="ignore").strip()
            elif t_input:
                transcript_text = t_input.strip()

            st.caption("For best results, read the same transcript as the reference.")

        # Advanced Settings Expander
        with st.expander("Advanced Detection Settings", expanded=False):
            z_threshold = st.slider(
                "Sensitivity Threshold (Z-score)",
                min_value=1.0,
                max_value=3.0,
                value=float(DEFAULT_Z_THRESH),
                step=0.1,
                help="Lower values increase detector sensitivity (more flaws detected); higher values require larger deviations.",
            )

            comp_mode = st.radio(
                "Comparison mode",
                options=[
                    "Strict (same speaker or ideal reference)",
                    "Style-tolerant (different speaker)",
                ],
                key="comparison_mode",
                help="Tolerant mode ignores overall speed and pitch-range differences between speakers, so a flaw spread over the whole recording is harder to see.",
            )

        is_speaker_tolerant = comp_mode == "Style-tolerant (different speaker)"

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Spoken Delivery", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
            <div class="caption-sm" style="text-align: center;">
                FlawScope Prototype | Phase B2<br/>
                Signal Processing and Temporal Grounding
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trigger analysis on button click or if re-analysis requested
    re_analyze = st.session_state.pop("re_analyze", False)
    if analyze_clicked or re_analyze:
        # If re-analyzing and inputs are empty, restore from last_inputs
        last_inputs = st.session_state.get("last_inputs", {})
        if not baseline_path and last_inputs.get("baseline_path"):
            baseline_path = last_inputs["baseline_path"]
        if not participant_path and last_inputs.get("participant_path"):
            participant_path = last_inputs["participant_path"]
        if not transcript_text and last_inputs.get("transcript_text"):
            transcript_text = last_inputs["transcript_text"]

        if not is_mock_mode and (not baseline_path or not participant_path or not transcript_text):
            st.error("Please provide baseline audio, participant audio, and transcript before analyzing.")
        else:
            with st.spinner("Aligning audio waveforms and analyzing delivery metrics..."):
                try:
                    overrides = {
                        "Z_THRESH": z_threshold,
                        "SPEAKER_TOLERANT": is_speaker_tolerant,
                    }
                    result = adapter.get_result(
                        baseline_path=baseline_path or "",
                        participant_path=participant_path or "",
                        transcript_text=transcript_text,
                        overrides=overrides,
                    )
                    st.session_state["analysis_result"] = result
                    st.session_state["active_pair_id"] = selected_pair_option
                    st.session_state["last_inputs"] = {
                        "baseline_path": baseline_path,
                        "participant_path": participant_path,
                        "transcript_text": transcript_text,
                    }

                    # Load audio arrays for playback and timeline caching
                    if participant_path and Path(participant_path).exists():
                        y_p, sr_p = adapter.load_audio_for_ui(participant_path)
                        st.session_state["participant_audio"] = (y_p, sr_p)
                    if baseline_path and Path(baseline_path).exists():
                        y_b, sr_b = adapter.load_audio_for_ui(baseline_path)
                        st.session_state["baseline_audio"] = (y_b, sr_b)

                    st.toast("Analysis completed successfully!")
                except Exception as exc:
                    st.error(
                        f"Analysis failed: {exc}\n\n"
                        "Hint: Check audio file formats or run with FLAWSCOPE_MOCK=1 to verify UI rendering."
                    )

    # --- Header ---
    st.markdown(
        f"""
        <div style="margin-bottom: 1.2rem;">
            <h1 style="margin: 0; font-size: 2.1rem; color: {theme.PALETTE['ink']};">
                FlawScope
            </h1>
            <div style="font-size: 1.05rem; color: #4B5563; margin-top: 0.2rem;">
                Counterpointing speech analytics and temporal flaw grounding against reference spoken deliveries.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Main Content Tabs ---
    result = st.session_state.get("analysis_result")
    ctx = {
        "baseline_audio": st.session_state.get("baseline_audio"),
        "participant_audio": st.session_state.get("participant_audio"),
        "pair_id": st.session_state.get("active_pair_id"),
        "is_mock": is_mock_mode,
    }

    if result is None:
        st.markdown(
            """
            <div class="flawscope-card" style="text-align: center; padding: 3rem 1rem;">
                <h3 style="margin-bottom: 0.4rem; color: #1F2A37;">Ready for Speech Performance Analysis</h3>
                <p style="color: #6B7280; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                    Select a contrastive evaluation pair or upload your own spoken audio in the sidebar,
                    then click <strong>Analyze Spoken Delivery</strong> to extract pitch, cadence, pausing, and clarity metrics.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Check for speaker warning in result.meta.warnings
    meta = result.get("meta", {})
    warnings = meta.get("warnings", []) if isinstance(meta, dict) else getattr(meta, "warnings", [])
    speaker_warning = next(
        (w for w in warnings if "different speaker likely" in str(w).lower()),
        None,
    )
    if speaker_warning:
        col_warn, col_btn = st.columns([3.6, 1.4])
        with col_warn:
            st.markdown('<div class="coral-alert">', unsafe_allow_html=True)
            st.warning(str(speaker_warning))
            st.markdown("</div>", unsafe_allow_html=True)
        with col_btn:
            st.markdown("<div style='height: 0.35rem;'></div>", unsafe_allow_html=True)
            st.button(
                "Re-run in style-tolerant mode",
                type="primary",
                on_click=_switch_to_style_tolerant,
                use_container_width=True,
            )

    # Tabs container
    tabs = st.tabs([
        "Overview",
        "Timeline",
        "Flaws",
        "Spectrogram",
        "Validation",
        "Method",
    ])

    with tabs[0]:
        tab_overview.render(result, ctx)
    with tabs[1]:
        tab_timeline.render(result, ctx)
    with tabs[2]:
        tab_flaws.render(result, ctx)
    with tabs[3]:
        tab_spectrogram.render(result, ctx)
    with tabs[4]:
        tab_validation.render(result, ctx)
    with tabs[5]:
        tab_method.render(result, ctx)


if __name__ == "__main__":
    main()
