"""Streamlit web dashboard entry point for FlawScope supporting benchmark and reference modes."""

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
    if "transcript_override" not in st.session_state:
        st.session_state["transcript_override"] = None


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

    is_mock_mode = (
        os.environ.get("FLAWSCOPE_MOCK") == "1"
        or os.environ.get("FLAWSCOPE_MOCK_MODE") == "benchmark"
    )

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

        # 1. Top Section "Your recording"
        st.markdown("### Your recording")

        recorded_audio = None
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Record now")

        p_upload = st.file_uploader(
            "Upload audio (WAV, MP3, M4A)",
            type=["wav", "mp3", "m4a"],
            key="p_up",
            help="Upload your spoken delivery to analyze against benchmarks or a reference.",
        )

        participant_path: Optional[str] = None
        baseline_path: Optional[str] = None
        transcript_text: str = ""

        if recorded_audio:
            participant_path = save_uploaded_file(recorded_audio, suffix=".wav")
        elif p_upload:
            participant_path = save_uploaded_file(p_upload, suffix=Path(p_upload.name).suffix)

        # Compact player after upload
        if participant_path and Path(participant_path).exists():
            st.audio(participant_path)

        # 2. Transcript Expander (shown after analysis)
        prev_result = st.session_state.get("analysis_result")
        if prev_result is not None:
            prev_meta = prev_result.get("meta", {})
            auto_tx = (
                prev_meta.get("transcript_text")
                or " ".join(w.get("text", "") for w in prev_result.get("words", []))
            )
            with st.expander("Transcript (auto-generated) - edit if anything is wrong", expanded=False):
                edited_tx = st.text_area(
                    "Edit transcript text",
                    value=auto_tx,
                    height=110,
                    key="tx_edit_input",
                )
                if st.button("Re-analyse with this transcript", type="secondary", use_container_width=True):
                    st.session_state["transcript_override"] = edited_tx
                    st.session_state["re_analyze"] = True
                    st.rerun()

        # 3. Collapsed expander "Advanced: compare against a reference"
        demo_selected = False
        selected_demo = "None (benchmark mode)"

        with st.expander("Advanced: compare against a reference", expanded=False):
            st.markdown("#### Reference Baseline Audio")
            b_upload = st.file_uploader(
                "Reference baseline audio (optional)",
                type=["wav", "mp3", "m4a"],
                key="b_up",
                help="Upload a reference baseline audio of the exact same transcript to evaluate in Reference Mode.",
            )
            if b_upload:
                baseline_path = save_uploaded_file(b_upload, suffix=Path(b_upload.name).suffix)
                st.audio(baseline_path)

            st.markdown("#### Reference Transcript")
            t_upload = st.file_uploader("Reference transcript file (.txt) (optional)", type=["txt"], key="t_up")
            t_input = st.text_area("Or paste reference transcript", height=80, key="t_in")
            if t_upload:
                transcript_text = t_upload.getvalue().decode("utf-8", errors="ignore").strip()
            elif t_input:
                transcript_text = t_input.strip()

            st.caption("For best results, read the same transcript as the reference.")

            st.markdown("---")
            st.markdown("#### Demo / validation pairs")
            demo_pairs = adapter.list_demo_pairs()
            demo_options = ["None (benchmark mode)"] + [p["pair_id"] for p in demo_pairs]
            selected_demo = st.selectbox(
                "Demo / validation pairs",
                options=demo_options,
                index=0,
                help="Choose a pre-generated contrastive flaw pair to evaluate in reference mode.",
            )

            if selected_demo != "None (benchmark mode)":
                demo_selected = True
                matched = next((p for p in demo_pairs if p["pair_id"] == selected_demo), None)
                if matched:
                    baseline_path = str(PROJECT_ROOT / matched["baseline"])
                    if not participant_path:
                        participant_path = str(PROJECT_ROOT / matched["participant"])
                    tr_file = PROJECT_ROOT / matched["transcript"]
                    if tr_file.exists() and not transcript_text:
                        transcript_text = tr_file.read_text(encoding="utf-8").strip()

                    kind = matched.get("kind", "mixed").upper()
                    level = matched.get("level", "-")
                    st.caption(f"**Kind**: `{kind}` | **Severity**: `Level {level}`")

            st.markdown("---")
            st.markdown("#### Detection Tuning")
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

        # 4. Mode Chip beside Analyze
        is_reference_mode = bool(baseline_path)
        chip_text = (
            "Reference mode - compared with your reference recording"
            if is_reference_mode
            else "Benchmark mode - judged against strong speakers"
        )
        chip_class = "mode-reference" if is_reference_mode else "mode-benchmark"

        st.markdown(
            f"""
            <div style="margin-top: 0.8rem; margin-bottom: 0.5rem;">
                <span class="mode-chip {chip_class}">
                    {chip_text}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button("Analyze Spoken Delivery", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
            <div class="caption-sm" style="text-align: center;">
                FlawScope Prototype | Benchmark & Reference Modes<br/>
                Signal Processing and Acoustic Grounding
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Trigger analysis on button click or if re-analysis requested
    re_analyze = st.session_state.pop("re_analyze", False)
    if analyze_clicked or re_analyze:
        # Check if transcript override is present from editor
        if st.session_state.get("transcript_override"):
            transcript_text = st.session_state.pop("transcript_override")

        # If re-analyzing and inputs are empty, restore from last_inputs
        last_inputs = st.session_state.get("last_inputs", {})
        if not baseline_path and last_inputs.get("baseline_path"):
            baseline_path = last_inputs["baseline_path"]
        if not participant_path and last_inputs.get("participant_path"):
            participant_path = last_inputs["participant_path"]
        if not transcript_text and last_inputs.get("transcript_text"):
            transcript_text = last_inputs["transcript_text"]

        if not is_mock_mode and not participant_path:
            st.error("Please record or upload your audio before analyzing.")
        else:
            with st.spinner("Analyzing spoken delivery..."):
                try:
                    overrides = {
                        "Z_THRESH": z_threshold,
                        "SPEAKER_TOLERANT": is_speaker_tolerant,
                    }
                    result = adapter.get_result(
                        baseline_path=baseline_path,
                        participant_path=participant_path or "",
                        transcript_text=transcript_text,
                        overrides=overrides,
                    )
                    st.session_state["analysis_result"] = result
                    st.session_state["active_pair_id"] = selected_demo if demo_selected else None
                    st.session_state["last_inputs"] = {
                        "baseline_path": baseline_path,
                        "participant_path": participant_path,
                        "transcript_text": transcript_text,
                    }

                    # Load audio arrays for playback and timeline caching
                    if participant_path and Path(participant_path).exists():
                        y_p, sr_p = adapter.load_audio_for_ui(participant_path)
                        st.session_state["participant_audio"] = (y_p, sr_p)
                    else:
                        st.session_state["participant_audio"] = None

                    if baseline_path and Path(baseline_path).exists():
                        y_b, sr_b = adapter.load_audio_for_ui(baseline_path)
                        st.session_state["baseline_audio"] = (y_b, sr_b)
                    else:
                        st.session_state["baseline_audio"] = None

                    # Automatically persist to profile store
                    try:
                        from app.store.db import create_profile, list_profiles, save_session
                        profiles = list_profiles()
                        pid = profiles[0]["id"] if profiles else create_profile("Default Speaker")
                        save_session(
                            profile_id=pid,
                            result=result,
                            ctx={
                                "baseline_audio_path": baseline_path or "",
                                "participant_audio_path": participant_path or "",
                                "transcript_text": result.get("meta", {}).get("transcript_text", ""),
                                "mode": result.get("meta", {}).get("mode", "benchmark"),
                            },
                            title=f"Delivery ({result.get('meta', {}).get('mode', 'benchmark')})",
                        )
                    except Exception:
                        pass

                    st.toast("Analysis completed successfully!")
                    st.rerun()
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
                Speech performance analytics and temporal flaw grounding benchmarked against strong speakers or reference deliveries.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Main Content Tabs ---
    result = st.session_state.get("analysis_result")
    mode = (
        result.get("meta", {}).get("mode", "benchmark")
        if result
        else ("reference" if baseline_path else "benchmark")
    )
    ctx = {
        "baseline_audio": st.session_state.get("baseline_audio"),
        "participant_audio": st.session_state.get("participant_audio"),
        "pair_id": st.session_state.get("active_pair_id"),
        "mode": mode,
        "is_mock": is_mock_mode,
    }

    if result is None:
        st.markdown(
            """
            <div class="flawscope-card" style="text-align: center; padding: 3rem 1rem;">
                <h3 style="margin-bottom: 0.4rem; color: #1F2A37;">Ready for Speech Performance Analysis</h3>
                <p style="color: #6B7280; max-width: 580px; margin: 0 auto 1.5rem auto; line-height: 1.5;">
                    Record your voice or upload audio in the sidebar, then click <strong>Analyze Spoken Delivery</strong> 
                    to benchmark your delivery against strong speakers (or compare against a reference recording in Advanced).
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
