"""Streamlit web dashboard for FlawScope: Counterpointing Speech Analytics."""

import json
from pathlib import Path
import tempfile
import streamlit as st

import config
from contracts.schema import AnalysisResult
from core.audio_io import load_audio
from core.pipeline import analyze
from app.ui.adapter import flaws_to_dataframe, words_to_dataframe
from app.ui.charts import (
    create_radar_chart,
    create_spectrogram_figure,
    create_timeline_figure,
)
from app.ui.components import (
    render_flaw_cards,
    render_score_cards,
    render_transcript_strip,
)


def load_dataset_pairs() -> list[dict]:
    """Load pre-configured speech pairs from dataset/pairs.json."""
    pairs_file = Path("dataset/pairs.json")
    if pairs_file.exists():
        try:
            with open(pairs_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def main():
    st.set_page_config(
        page_title="FlawScope | Speech Delivery Analytics",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🎙️ FlawScope")
    st.caption("Counterpointing Speech Analytics & Temporal Flaw Grounding")

    # Sidebar
    st.sidebar.header("Input & Configuration")
    dataset_pairs = load_dataset_pairs()

    input_mode = st.sidebar.radio(
        "Audio Source",
        ["Dataset Demo Pair", "Upload Custom Audio"],
        index=0 if dataset_pairs else 1,
    )

    baseline_path = None
    participant_path = None
    transcript_text = None

    if input_mode == "Dataset Demo Pair" and dataset_pairs:
        pair_options = {
            f"{p['pair_id']} (Level {p['level']} {p['kind']})": p
            for p in dataset_pairs
        }
        selected_label = st.sidebar.selectbox("Select Contrastive Pair", list(pair_options.keys()))
        selected_pair = pair_options[selected_label]

        baseline_path = selected_pair["baseline"]
        participant_path = selected_pair["participant"]
        transcript_file = Path(selected_pair["transcript"])

        if transcript_file.exists():
            transcript_text = transcript_file.read_text(encoding="utf-8")

        st.sidebar.markdown("**Baseline Audio (Reference):**")
        st.sidebar.audio(baseline_path)
        st.sidebar.markdown("**Participant Audio (Test):**")
        st.sidebar.audio(participant_path)
    else:
        uploaded_baseline = st.sidebar.file_uploader("Upload Baseline (WAV/MP3)", type=["wav", "mp3", "flac"])
        uploaded_participant = st.sidebar.file_uploader("Upload Participant (WAV/MP3)", type=["wav", "mp3", "flac"])
        transcript_input = st.sidebar.text_area("Reference Transcript", height=120)

        if uploaded_baseline and uploaded_participant and transcript_input:
            t_base = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t_base.write(uploaded_baseline.read())
            baseline_path = t_base.name

            t_part = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            t_part.write(uploaded_participant.read())
            participant_path = t_part.name

            transcript_text = transcript_input

    # Sensitivity Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Sensitivity")
    z_thresh = st.sidebar.slider(
        "Z-Score Threshold (Z)",
        min_value=1.0,
        max_value=2.5,
        value=float(config.Z_THRESH),
        step=0.1,
        help="Lower values increase sensitivity (detecting milder deviations); higher values flag only egregious flaws.",
    )

    analyze_btn = st.sidebar.button("🔍 Run Full Analysis", type="primary", use_container_width=True)

    # State preservation for results
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
        st.session_state["last_pair"] = None

    if analyze_btn and baseline_path and participant_path and transcript_text:
        with st.spinner("Analyzing audio, aligning phonemes, and grounding flaw intervals..."):
            try:
                res_dict = analyze(
                    baseline_audio=str(baseline_path),
                    participant_audio=str(participant_path),
                    transcript=transcript_text,
                    config_overrides={"Z_THRESH": z_thresh},
                )
                st.session_state["analysis_result"] = res_dict
                st.session_state["p_audio"] = str(participant_path)
                st.session_state["b_audio"] = str(baseline_path)
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

    result_dict = st.session_state.get("analysis_result")

    if not result_dict:
        # Default prompt when no analysis has run yet
        st.info("👈 Select a demo pair or upload audio files in the sidebar, then click **Run Full Analysis** to evaluate speech delivery.")
        return

    # Parse result into Pydantic model
    result = AnalysisResult.model_validate(result_dict)

    # Main Tabs
    tab_overview, tab_timeline, tab_flaws, tab_spectrogram, tab_method = st.tabs(
        ["📊 Overview", "📈 Timeline Analysis", "🚨 Flaws & Recommendations", "🌈 Spectrogram", "📖 Methodology"]
    )

    with tab_overview:
        render_score_cards(result.scores)
        st.markdown("---")

        c_radar, c_summary = st.columns([1.2, 1])
        with c_radar:
            st.markdown("##### 5-Dimension Performance Radar")
            radar_fig = create_radar_chart(result.scores)
            st.plotly_chart(radar_fig, use_container_width=True)

        with c_summary:
            st.markdown("##### Performance Breakdown")
            st.markdown(
                f"- **Baseline Duration:** `{result.meta.duration_b:.2f} s`\n"
                f"- **Participant Duration:** `{result.meta.duration_p:.2f} s`\n"
                f"- **Baseline Alignment Match:** `{result.meta.matched_ratio_b:.1%}`\n"
                f"- **Participant Alignment Match:** `{result.meta.matched_ratio_p:.1%}`\n"
                f"- **Total Detected Flaws:** `{len(result.flaws)}`\n"
                f"- **Processing Time:** `{result.meta.timings_s.get('total', 0):.2f} s`"
            )

            # Flaw types count
            if result.flaws:
                from collections import Counter
                counts = Counter(f.type.value for f in result.flaws)
                count_str = ", ".join(f"`{k}: {v}`" for k, v in counts.items())
                st.markdown(f"**Flaws by Category:** {count_str}")

        render_transcript_strip(result.words, result.flaws)

    with tab_timeline:
        st.markdown("##### Synchronized Participant vs Baseline Acoustics")
        st.caption(
            "Solid ink line = Participant | Dashed grey line = Warped Baseline | Shaded regions = Delivery flaws"
        )
        timeline_fig = create_timeline_figure(result.curves, result.flaws, result.words)
        st.plotly_chart(timeline_fig, use_container_width=True)

    with tab_flaws:
        render_flaw_cards(result.flaws)

        st.markdown("---")
        st.markdown("##### Flaw Records Table")
        df_flaws = flaws_to_dataframe(result.flaws)
        if not df_flaws.empty:
            st.dataframe(df_flaws, use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Download Flaws CSV",
                    df_flaws.to_csv(index=False),
                    "flawscope_flaws.csv",
                    "text/csv",
                    use_container_width=True,
                )
            with col_dl2:
                st.download_button(
                    "📥 Download Full Result JSON",
                    json.dumps(result_dict, indent=2),
                    "flawscope_analysis.json",
                    "application/json",
                    use_container_width=True,
                )

    with tab_spectrogram:
        st.markdown("##### Log-Mel Spectrogram")
        p_audio_path = st.session_state.get("p_audio")
        if p_audio_path and Path(p_audio_path).exists():
            y_p, sr_p = load_audio(p_audio_path)
            spec_fig = create_spectrogram_figure(y_p, sr_p, result.flaws)
            st.plotly_chart(spec_fig, use_container_width=True)
        else:
            st.info("Spectrogram available when audio file path is accessible.")

    with tab_method:
        st.markdown("### FlawScope Rubric & Mathematical Rationale")
        st.markdown(
            """
            FlawScope evaluates spoken delivery against a time-warped reference of the exact same transcript.
            
            #### 1. Flaw Detection Rules
            - **MONOTONE**: Pitch variation ($\sigma$ of $F_0$ in semitones relative to median) over a 5-word window. Flagged when deviation $d = (m_p - m_b) / s_b \le -Z$.
            - **TOO_FAST / TOO_SLOW**: Syllables per second speech rate over word-matched windows. Flagged when $d \ge +Z$ (fast) or $d \le -Z$ (slow).
            - **LOW_VOLUME**: Mean loudness relative to median speech level. Flagged when $d \le -Z$ and absolute drop $\ge 4.0\\text{ dB}$.
            - **LONG_PAUSE**: Unintended hesitation between adjacent words where $gap_p \ge 0.9\\text{s}$ and $(gap_p - gap_b) \ge 0.5\\text{s}$.
            - **UNCLEAR_ARTICULATION**: Spectral divergence via Euclidean DTW of MFCCs. Flagged when robust $Z_c = (\\text{cost} - \\text{median}) / (1.4826 \\times \\text{MAD}) \ge Z + 0.5$.
            
            #### 2. Rubric Scoring Formulation
            For each dimension $d$:
            $$\\text{penalty}_d = \\frac{\\sum_{i \\in d} (\\text{severity}_i \\times \\text{duration}_i)}{T_{\\text{speech}}}$$
            $$\\text{score}_d = 100 \\times \\exp(-3.0 \\times \\text{penalty}_d)$$
            Overall score is a weighted linear combination:
            $$\\text{Overall} = 0.25 \\times \\text{pace} + 0.25 \\times \\text{intonation} + 0.15 \\times \\text{volume} + 0.15 \\times \\text{pausing} + 0.20 \\times \\text{clarity}$$
            """
        )
        st.markdown("#### Live Configuration Snapshot")
        st.json(result.meta.config)


if __name__ == "__main__":
    main()
