"""Validation tab module displaying benchmark metrics, confusion breakdown, severity ladder, and timeline alignment."""

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.ui.theme import FLAW_COLORS, PALETTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def render_metric_cards(summary: Dict[str, Any]) -> None:
    """Render overall benchmark metric cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    prec = summary.get("overall_precision", 0.0)
    rec = summary.get("overall_recall", 0.0)
    f1 = summary.get("overall_f1", 0.0)
    pairs_count = summary.get("total_pairs", 0)
    rho = summary.get("spearman_rho", 0.0)

    with col1:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center;">
                <div class="caption-sm">Overall Precision</div>
                <div class="score-large" style="color: {PALETTE['teal']}; font-size: 2.2rem;">{prec:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center;">
                <div class="caption-sm">Overall Recall</div>
                <div class="score-large" style="color: {PALETTE['sky']}; font-size: 2.2rem;">{rec:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center;">
                <div class="caption-sm">F1 Harmonic Mean</div>
                <div class="score-large" style="color: {PALETTE['green']}; font-size: 2.2rem;">{f1:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center;">
                <div class="caption-sm">Evaluated Pairs</div>
                <div class="score-large" style="color: {PALETTE['ink']}; font-size: 2.2rem;">{pairs_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="flawscope-card" style="text-align: center;">
                <div class="caption-sm">Severity Rank (rho)</div>
                <div class="score-large" style="color: {PALETTE['coral'] if rho < 0 else PALETTE['amber']}; font-size: 2.2rem;">{rho:+.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_per_type_grouped_bar(per_type: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Render grouped bar chart of Precision, Recall, and F1 across flaw categories."""
    categories = [k.replace("_", " ").title() for k in per_type.keys()]
    precisions = [v["precision"] for v in per_type.values()]
    recalls = [v["recall"] for v in per_type.values()]
    f1s = [v["f1"] for v in per_type.values()]

    fig = go.Figure(
        data=[
            go.Bar(name="Precision", x=categories, y=precisions, marker_color=PALETTE["teal"]),
            go.Bar(name="Recall", x=categories, y=recalls, marker_color=PALETTE["sky"]),
            go.Bar(name="F1 Score", x=categories, y=f1s, marker_color=PALETTE["green"]),
        ]
    )

    fig.update_layout(
        barmode="group",
        height=300,
        margin=dict(l=30, r=20, t=30, b=30),
        yaxis=dict(title="Score (0 - 1)", range=[0, 1.05], gridcolor="#E5E7EB"),
        xaxis=dict(gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def render_severity_ladder_chart(ladder: Dict[str, Dict[str, Any]], rho: float) -> go.Figure:
    """Render line chart of mean score progression across severity ladder."""
    levels = list(ladder.keys())
    mean_scores = [ladder[k]["mean_score"] for k in levels]
    sample_counts = [ladder[k]["count"] for k in levels]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=levels,
            y=mean_scores,
            mode="lines+markers+text",
            text=[f"{s:.1f} pts" for s in mean_scores],
            textposition="top center",
            line=dict(color=PALETTE["coral"], width=3),
            marker=dict(size=10, color=PALETTE["teal"]),
            name="Mean Rubric Score",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Severity Ladder Monotonicity Progression (Spearman rho = {rho:+.3f})",
            font=dict(size=13, color="#1F2A37"),
        ),
        xaxis=dict(title="Severity Tier (L0 Clean Baseline to L4 Egregious)", gridcolor="#E5E7EB"),
        yaxis=dict(title="Overall Score (0 - 100)", range=[50, 105], gridcolor="#E5E7EB"),
        height=300,
        margin=dict(l=40, r=30, t=40, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def render_pair_timeline_comparison(pair_data: Dict[str, Any]) -> go.Figure:
    """Render dual-row horizontal timeline comparison contrasting Ground Truth vs Detected."""
    truth_flaws = pair_data.get("truth", [])
    detected_flaws = pair_data.get("detected", [])

    fig = go.Figure()

    # Row 1: Ground Truth (y=1)
    for g in truth_flaws:
        ft = g.get("type", "UNKNOWN")
        color = FLAW_COLORS.get(ft, PALETTE["coral"])
        s = float(g.get("start", 0.0))
        e = float(g.get("end", 0.0))
        fig.add_trace(
            go.Bar(
                x=[e - s],
                y=["Ground Truth"],
                base=[s],
                orientation="h",
                marker=dict(color=color, line=dict(color="#1F2A37", width=1)),
                name=ft,
                showlegend=False,
                hovertemplate=f"Truth: {ft}<br>Interval: {s:.2f}s - {e:.2f}s<extra></extra>",
            )
        )

    # Row 2: Detected (y=0)
    for d in detected_flaws:
        ft = d.get("type", "UNKNOWN")
        color = FLAW_COLORS.get(ft, PALETTE["coral"])
        s = float(d.get("start", 0.0))
        e = float(d.get("end", 0.0))
        fig.add_trace(
            go.Bar(
                x=[max(0.1, e - s)],
                y=["Detected"],
                base=[s],
                orientation="h",
                marker=dict(color=color, line=dict(color="#1F2A37", width=1.5)),
                name=ft,
                showlegend=False,
                hovertemplate=f"Detected: {ft}<br>Interval: {s:.2f}s - {e:.2f}s<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="overlay",
        height=200,
        margin=dict(l=100, r=30, t=30, b=30),
        xaxis=dict(title="Timeline (seconds)", gridcolor="#E5E7EB"),
        yaxis=dict(gridcolor="#E5E7EB"),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def render(result: Dict[str, Any], ctx: Dict[str, Any]) -> None:
    """Render Validation tab interface."""
    try:
        results_file = PROJECT_ROOT / "eval" / "results.json"

        st.subheader("Benchmark Validation & Detection Accuracy")
        st.markdown(
            "Empirical verification of flaw detection precision, recall, and scoring monotonicity against ground-truth labels."
        )

        if not results_file.exists():
            st.warning("Benchmark results file (`eval/results.json`) not found.")
            st.markdown(
                """
                <div class="flawscope-card">
                    <h4>Run Evaluation Benchmark</h4>
                    <p>To generate the full empirical benchmark metrics, run the evaluation CLI in your terminal:</p>
                    <pre><code>python eval/run_eval.py --quick</code></pre>
                    <p class="caption-sm">Omitting <code>--quick</code> runs full verification across all 18 contrastive audio pairs.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        with open(results_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        summary = eval_data.get("summary", {})
        per_type = eval_data.get("per_type", {})
        ladder = eval_data.get("severity_ladder", {})
        pairs = eval_data.get("pairs", [])

        # 1. Metric Cards
        render_metric_cards(summary)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        # 2. Charts Row: Grouped Bar & Severity Ladder
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown(
                """
                <div class="flawscope-card">
                    <div style="font-weight: 700; color: #1F2A37; margin-bottom: 0.4rem;">
                        Accuracy Metrics by Flaw Category
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig_bar = render_per_type_grouped_bar(per_type)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_c2:
            st.markdown(
                """
                <div class="flawscope-card">
                    <div style="font-weight: 700; color: #1F2A37; margin-bottom: 0.4rem;">
                        Severity Ladder Monotonicity Progression
                    </div>
                """,
                unsafe_allow_html=True,
            )
            fig_ladder = render_severity_ladder_chart(ladder, summary.get("spearman_rho", 0.0))
            st.plotly_chart(fig_ladder, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Per-type Table
        st.markdown("### Per-Category Confusion Matrix")
        table_rows = []
        for ft, vals in per_type.items():
            table_rows.append({
                "Flaw Category": ft.replace("_", " ").title(),
                "True Positives (TP)": vals["TP"],
                "False Positives (FP)": vals["FP"],
                "False Negatives (FN)": vals["FN"],
                "Precision": f"{vals['precision']:.3f}",
                "Recall": f"{vals['recall']:.3f}",
                "F1 Score": f"{vals['f1']:.3f}",
            })
        df_per_type = pd.DataFrame(table_rows)
        st.dataframe(df_per_type, use_container_width=True, hide_index=True)

        # 4. Pair-Level Ground Truth vs Detected Visualizer
        st.markdown("### Temporal Alignment: Ground Truth vs Detected")
        st.markdown("Inspect how closely detected intervals match reference ground-truth annotations.")

        pair_ids = [p["pair_id"] for p in pairs]
        selected_pair_id = st.selectbox("Select Pair to Inspect", options=pair_ids, index=0)

        selected_pair = next((p for p in pairs if p["pair_id"] == selected_pair_id), None)
        if selected_pair:
            fig_pair = render_pair_timeline_comparison(selected_pair)
            st.plotly_chart(fig_pair, use_container_width=True)

            score = selected_pair.get("overall_score", 0.0)
            det_cnt = selected_pair.get("detected_count", 0)
            gt_cnt = selected_pair.get("truth_count", 0)
            st.caption(
                f"**Pair**: `{selected_pair_id}` | **Rubric Score**: `{score:.1f}` pts | "
                f"**Ground Truth Flaws**: `{gt_cnt}` | **Detected Flaws**: `{det_cnt}`"
            )

    except Exception as exc:
        st.error(f"Error rendering Validation tab: {exc}")
