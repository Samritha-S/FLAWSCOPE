"""Methodology tab displaying mathematical formulation, DTW alignment, and active pipeline configuration."""

from typing import Any, Dict
import streamlit as st

import config


def render_pipeline_flowchart() -> None:
    """Render the FlawScope architectural pipeline diagram using Graphviz."""
    dot_code = """
    digraph FlawScopePipeline {
        rankdir=LR;
        node [shape=box, style="rounded,filled", fontname="sans-serif", fontsize=10, penwidth=1.5];
        edge [fontname="sans-serif", fontsize=9, color="#8A94A6", penwidth=1.2];

        subgraph cluster_input {
            label = "1. Audio Ingestion & Preprocessing";
            fontname="sans-serif"; fontsize=10; color="#E5E7EB"; style="dashed,rounded";
            B_Audio [label="Baseline Audio\\n(Reference 16kHz mono)", fillcolor="#F4FAF9", color="#0F9D8A"];
            P_Audio [label="Participant Audio\\n(Target 16kHz mono)", fillcolor="#F4FAF9", color="#1F2A37"];
            Transcript [label="Target Transcript\\n(Plain Text)", fillcolor="#FFFFFF", color="#8A94A6"];
        }

        subgraph cluster_alignment {
            label = "2. Temporal Grounding";
            fontname="sans-serif"; fontsize=10; color="#E5E7EB"; style="dashed,rounded";
            Aligner [label="Forced Word Alignment\\n(faster-whisper base.en)", fillcolor="#FFFFFF", color="#0F9D8A"];
            DTW_Warp [label="Sub-word Acoustic DTW\\n(Frame Warping)", fillcolor="#FFFFFF", color="#0F9D8A"];
        }

        subgraph cluster_features {
            label = "3. Acoustic Feature Extraction";
            fontname="sans-serif"; fontsize=10; color="#E5E7EB"; style="dashed,rounded";
            Pitch_F0 [label="Pitch Tracking\\n(Parselmouth / Praat f0)", fillcolor="#FFFFFF", color="#2D9CDB"];
            Loudness [label="Relative Loudness\\n(RMS Energy vs Median)", fillcolor="#FFFFFF", color="#8D6E63"];
            Rate_Calc [label="Syllabic Speech Rate\\n(Syllables / Sec)", fillcolor="#FFFFFF", color="#F5A623"];
            MFCC_Spec [label="Spectral Representation\\n(64-mel Log & MFCC c1-c12)", fillcolor="#FFFFFF", color="#3BB273"];
        }

        subgraph cluster_detection {
            label = "4. Windowed Contrastive Detection";
            fontname="sans-serif"; fontsize=10; color="#E5E7EB"; style="dashed,rounded";
            Window_Engine [label="Sliding Word Windows\\n(Radius = 2 words)", fillcolor="#FFFFFF", color="#0F9D8A"];
            Detectors [label="Time-Bounded Detectors\\n(Z-score, Pause, Clarity)", fillcolor="#FFFFFF", color="#FF6B5B"];
        }

        subgraph cluster_scoring {
            label = "5. Explainability & Scoring";
            fontname="sans-serif"; fontsize=10; color="#E5E7EB"; style="dashed,rounded";
            Flaw_List [label="Flaws with Acoustic Deltas\\n(Intervals & Explanations)", fillcolor="#FFF4F2", color="#FF6B5B"];
            Rubric_Scores [label="5-Dimension Rubric Scores\\n(Exponential Decay Penalty)", fillcolor="#F4FAF9", color="#0F9D8A"];
        }

        B_Audio -> Aligner;
        P_Audio -> Aligner;
        Transcript -> Aligner;

        Aligner -> DTW_Warp;
        Aligner -> Pitch_F0;
        Aligner -> Loudness;
        Aligner -> Rate_Calc;
        Aligner -> MFCC_Spec;

        DTW_Warp -> Window_Engine;
        Pitch_F0 -> Window_Engine;
        Loudness -> Window_Engine;
        Rate_Calc -> Window_Engine;
        MFCC_Spec -> Window_Engine;

        Window_Engine -> Detectors;
        Detectors -> Flaw_List;
        Flaw_List -> Rubric_Scores;
    }
    """
    st.graphviz_chart(dot_code, use_container_width=True)


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Method tab interface with live mathematical formulations from config."""
    try:
        st.subheader("Methodology & Mathematical Rationale")
        st.markdown(
            """
            FlawScope evaluates spoken performance by contrasting acoustic signals against a reference baseline 
            of the **exact same transcript**. The pipeline eliminates subjective bias through deterministic 
            statistical bounds, word-level alignment, and acoustic feature distance metrics.
            """
        )

        st.markdown("### Pipeline Architecture")
        render_pipeline_flowchart()

        st.markdown("---")
        st.markdown("### Mathematical Formulations & Live Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="flawscope-card">
                    <h4 style="margin-top: 0; color: #1F2A37;">1. Pitch Inflection (Semitones)</h4>
                    <p class="caption-sm">Physical pitch in Hertz is converted to semitones relative to speaker median 
                    pitch across voiced frames (range: {config.F0_MIN} - {config.F0_MAX} Hz):</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"f_{0,\text{st}}(t) = 12 \cdot \log_2\left(\frac{f_0(t)}{\text{median}(f_0)}\right)")

            st.markdown(
                f"""
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">2. Windowed Contrastive Deviation (Z-Score)</h4>
                    <p class="caption-sm">For each word window (radius = {config.WIN_RADIUS} words, spanning {config.WIN_RADIUS * 2 + 1} words), 
                    mean participant value $m_p$ is compared against baseline mean $m_b$ and baseline spread $s_b$:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"d_i = \frac{m_{p,i} - m_{b,i}}{\max(s_{b,i}, \sigma_{\text{floor}})}")
            st.markdown(
                """
                <div class="caption-sm">
                    <strong>Active Threshold</strong>: Deviation is flagged as a delivery flaw if |d<sub>i</sub>| &ge; Z<sub>thresh</sub> = """
                + str(config.Z_THRESH)
                + """.<br/>
                    Spread floor &sigma;<sub>floor</sub> prevents division by near-zero variance 
                    (Intonation: """
                + str(config.SPREAD_FLOOR["MONOTONE"])
                + """, Pace: """
                + str(config.SPREAD_FLOOR["PACE"])
                + """, Volume: """
                + str(config.SPREAD_FLOOR["LOW_VOLUME"])
                + """ dB).
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">3. Speech Tempo & Rate Estimation</h4>
                    <p class="caption-sm">Rate is calculated as syllables delivered per second of active speech time:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"\text{Rate} = \frac{N_{\text{syllables}}}{\Delta t_{\text{speech}}}")

            st.markdown(
                """
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">4. Relative Vocal Loudness</h4>
                    <p class="caption-sm">RMS vocal energy in decibels relative to the participant's median voiced speech level:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"L_{\text{rel}}(t) = 20 \cdot \log_{10}\left(\frac{\text{RMS}(t) + \epsilon}{\text{median}(\text{RMS}) + \epsilon}\right)")

        with col2:
            st.markdown(
                """
                <div class="flawscope-card">
                    <h4 style="margin-top: 0; color: #1F2A37;">5. Articulation & DTW Phonetic Distance</h4>
                    <p class="caption-sm">Acoustic distance between aligned participant and baseline frames is computed over 
                    MFCC coefficients (c_1 - c_12) via Dynamic Time Warping. Deviations are evaluated using robust statistics:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"Z_{\text{dtw}}(t) = \frac{c(t) - \text{median}(c)}{1.4826 \cdot \text{MAD}(c)}")
            st.markdown(
                """
                <div class="caption-sm">
                    Flagged as unclear articulation when Z<sub>dtw</sub> &ge; Z<sub>thresh</sub> + """
                + f"{config.CLARITY_Z_EXTRA} = {config.Z_THRESH + config.CLARITY_Z_EXTRA:.1f}."
                + """
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">6. Pause Deviation Rule</h4>
                    <p class="caption-sm">Long pauses are identified by joint thresholding on duration and delta vs baseline:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(
                r"\text{Pause} > %s\text{ s} \quad \text{and} \quad \Delta_{\text{pause}} > %s\text{ s}"
                % (config.LONG_PAUSE_S, config.LONG_PAUSE_DELTA_S)
            )

            st.markdown(
                f"""
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">7. Flaw Severity Bounding</h4>
                    <p class="caption-sm">Raw deviations are normalized into a bounded continuous severity metric in [0, 1]:</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(
                r"\text{Severity} = \text{clip}\left(\frac{|d|}{2 \cdot %s}, 0, 1\right)"
                % config.Z_THRESH
            )

            st.markdown(
                f"""
                <div class="flawscope-card" style="margin-top: 1rem;">
                    <h4 style="margin-top: 0; color: #1F2A37;">8. Rubric Scoring Formulation</h4>
                    <p class="caption-sm">Each rubric dimension is penalized exponentially based on flaw severity and duration (K = {config.PENALTY_K}):</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(
                r"S_D = 100 \cdot \exp\left(-%s \cdot \frac{\sum_{f \in \mathcal{F}_D} \text{Sev}(f) \cdot \Delta t_f}{T_{\text{total}}}\right)"
                % config.PENALTY_K
            )

        st.markdown("---")
        st.markdown("### Dimension Weights & Rubric Aggregation")
        weights_cols = st.columns(5)
        dims = [
            ("Pace", "pace", config.WEIGHTS["pace"]),
            ("Intonation", "intonation", config.WEIGHTS["intonation"]),
            ("Volume", "volume", config.WEIGHTS["volume"]),
            ("Pausing", "pausing", config.WEIGHTS["pausing"]),
            ("Clarity", "clarity", config.WEIGHTS["clarity"]),
        ]
        for col, (name, key, wt) in zip(weights_cols, dims):
            with col:
                st.metric(label=f"{name} Weight", value=f"{int(wt * 100)}%")

        st.latex(r"\text{Overall Score} = \sum_{D} w_D \cdot S_D")

        st.markdown("---")
        st.markdown("### Architectural Limitations")
        st.markdown(
            """
            1. **Reference Baseline Assumption**: FlawScope relies on a high-quality baseline delivery of the exact same transcript. 
               Cross-transcript or freeform conversational evaluations are outside the current contrastive paradigm.
            2. **Acoustic Register & Gender Matching**: Highly discordant pitch registers between participant and baseline 
               are normalized via relative semitone shifts, but extreme vocal fry or whisper registers can impact pitch extractor stability.
            3. **Language Scope**: Current forced alignment and syllabic segmentation models are tuned for English phonetics (`base.en`).
            4. **Synthetic Proxy Approximations**: Generated evaluation benchmarks use DSP transforms to isolate flaws with exact ground-truth bounds. 
               Natural human deliveries often feature subtle, compounded delivery variations.
            """
        )

    except Exception as exc:
        st.error(f"Error rendering Method tab: {exc}")
