"""UI theme colors, visual constants, and CSS injection helpers for FlawScope."""

import streamlit as st

# UI Color Palette
PALETTE = {
    "teal": "#0F9D8A",
    "amber": "#F5A623",
    "coral": "#FF6B5B",
    "sky": "#2D9CDB",
    "green": "#3BB273",
    "brown": "#8D6E63",
    "ink": "#1F2A37",
    "soft_bg": "#F4FAF9",
    "white": "#FFFFFF",
    "grey_border": "#E5E7EB",
    "baseline_curve": "#8A94A6",
    "participant_curve": "#1F2A37",
}

# Flaw Category Colors
FLAW_COLORS = {
    "MONOTONE": "#FF6B5B",
    "TOO_FAST": "#F5A623",
    "TOO_SLOW": "#2D9CDB",
    "LONG_PAUSE": "#0F9D8A",
    "LOW_VOLUME": "#8D6E63",
    "UNCLEAR_ARTICULATION": "#3BB273",
}

BASELINE_COLOR = PALETTE["baseline_curve"]
PARTICIPANT_COLOR = PALETTE["participant_curve"]


def get_score_color(score: float) -> str:
    """Return rubric score color: >= 80 green, 60-79 amber, < 60 coral."""
    if score >= 80.0:
        return PALETTE["green"]
    elif score >= 60.0:
        return PALETTE["amber"]
    return PALETTE["coral"]


def inject_css() -> None:
    """Inject polished custom CSS for cards, pills, transcript strip, and responsiveness."""
    css = f"""
    <style>
    /* Global font & background accents */
    .stApp {{
        background-color: {PALETTE["soft_bg"]};
        color: {PALETTE["ink"]};
    }}

    /* Card container with soft shadow and teal top border */
    .flawscope-card {{
        background-color: #FFFFFF;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(15, 157, 138, 0.08);
        border: 1px solid #E5E7EB;
        border-top: 3px solid {PALETTE["teal"]};
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .flawscope-card:hover {{
        box-shadow: 0 6px 18px rgba(15, 157, 138, 0.14);
    }}

    /* Flaw type rounded pill */
    .flaw-pill {{
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: #FFFFFF;
        margin-right: 0.4rem;
        vertical-align: middle;
    }}

    /* Specific flaw pill backgrounds */
    .pill-MONOTONE {{ background-color: {FLAW_COLORS["MONOTONE"]}; }}
    .pill-TOO_FAST {{ background-color: {FLAW_COLORS["TOO_FAST"]}; }}
    .pill-TOO_SLOW {{ background-color: {FLAW_COLORS["TOO_SLOW"]}; }}
    .pill-LONG_PAUSE {{ background-color: {FLAW_COLORS["LONG_PAUSE"]}; }}
    .pill-LOW_VOLUME {{ background-color: {FLAW_COLORS["LOW_VOLUME"]}; }}
    .pill-UNCLEAR_ARTICULATION {{ background-color: {FLAW_COLORS["UNCLEAR_ARTICULATION"]}; }}

    /* Small caption style */
    .caption-sm {{
        font-size: 0.8rem;
        color: #6B7280;
        margin-top: 0.2rem;
        line-height: 1.3;
    }}

    /* Mock Data Chip */
    .mock-chip {{
        display: inline-flex;
        align-items: center;
        background-color: #E6F4F1;
        color: {PALETTE["teal"]};
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(15, 157, 138, 0.3);
        margin-left: 0.5rem;
        letter-spacing: 0.05em;
    }}

    /* Large overall score display */
    .score-large {{
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 0.2rem;
    }}
    .score-sub {{
        font-size: 1.3rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .score-bar-bg {{
        background-color: #E5E7EB;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 0.4rem;
        overflow: hidden;
    }}
    .score-bar-fill {{
        height: 100%;
        border-radius: 4px;
    }}

    /* Aligned transcript strip */
    .transcript-container {{
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1.2rem;
        line-height: 2.2;
        font-size: 1.05rem;
        max-height: 280px;
        overflow-y: auto;
    }}
    .transcript-word {{
        display: inline;
        padding: 0.2rem 0.35rem;
        margin: 0.1rem 0.15rem;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.15s ease;
    }}
    .transcript-word:hover {{
        outline: 2px solid {PALETTE["teal"]};
    }}
    .word-flawed {{
        font-weight: 600;
    }}

    /* Legend strip */
    .legend-item {{
        display: inline-flex;
        align-items: center;
        margin-right: 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.82rem;
        font-weight: 500;
    }}
    .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.35rem;
    }}

    /* Coral st.warning banner */
    .coral-alert div[data-testid="stAlert"] {{
        background-color: rgba(255, 107, 91, 0.12) !important;
        border: 1px solid rgba(255, 107, 91, 0.3) !important;
        border-left: 5px solid #FF6B5B !important;
        color: #1F2A37 !important;
    }}
    .coral-alert div[data-testid="stAlert"] svg {{
        color: #FF6B5B !important;
        fill: #FF6B5B !important;
    }}

    /* Mode Chip Styling */
    .mode-chip {{
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }}
    .mode-benchmark {{
        background-color: #E8F8F5;
        color: #0F9D8A;
        border: 1px solid rgba(15, 157, 138, 0.35);
    }}
    .mode-reference {{
        background-color: #EEF2F6;
        color: #1F2A37;
        border: 1px solid #D1D5DB;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .score-large {{
            font-size: 2.2rem;
        }}
        .flawscope-card {{
            padding: 0.9rem;
        }}
        .transcript-container {{
            font-size: 0.95rem;
            line-height: 2.0;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
