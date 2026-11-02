"""Shared visual theme: custom CSS and small HTML helpers for every page."""

from __future__ import annotations

import streamlit as st


PRIMARY = "#0E7C86"
ACCENT = "#14B8A6"
DARK = "#0B3C49"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.block-container {{ padding-top: 2rem; max-width: 1250px; }}

h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; color: {DARK}; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {DARK} 0%, #0F5C66 100%);
}}
[data-testid="stSidebar"] * {{ color: #E8F6F7 !important; }}

[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid #D9ECEE;
    border-left: 5px solid {PRIMARY};
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(11, 60, 73, 0.07);
}}
[data-testid="stMetric"] label {{ color: #4B6B72 !important; }}
[data-testid="stMetricValue"] {{ color: {DARK}; font-weight: 700; }}

.stButton > button, .stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {{
    border-radius: 10px;
    font-weight: 600;
    transition: transform .12s ease, box-shadow .12s ease;
}}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(14, 124, 134, 0.3);
}}
button[kind="primary"] {{
    background: linear-gradient(135deg, {PRIMARY}, {ACCENT}) !important;
    border: none !important;
    color: #fff !important;
}}

[data-testid="stExpander"] {{
    border: 1px solid #D9ECEE;
    border-radius: 12px;
    background: #fbfeff;
}}

[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {PRIMARY};
    font-weight: 600;
}}

.ms-hero {{
    background: linear-gradient(120deg, {DARK} 0%, {PRIMARY} 60%, {ACCENT} 100%);
    color: #fff;
    border-radius: 18px;
    padding: 34px 38px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(11, 60, 73, 0.25);
}}
.ms-hero h1 {{ color: #fff; margin: 0 0 6px 0; font-size: 2.3rem; }}
.ms-hero p {{ margin: 0; font-size: 1.05rem; opacity: .92; }}

.ms-card {{
    background: #fff;
    border: 1px solid #D9ECEE;
    border-radius: 14px;
    padding: 18px 20px;
    height: 100%;
    box-shadow: 0 2px 10px rgba(11, 60, 73, 0.06);
}}
.ms-card h4 {{ margin: 0 0 6px 0; color: {DARK}; }}
.ms-card p {{ margin: 0; color: #4B6B72; font-size: .93rem; }}
.ms-step {{
    display: inline-block; background: {PRIMARY}; color: #fff;
    border-radius: 50%; width: 28px; height: 28px; line-height: 28px;
    text-align: center; font-weight: 700; margin-bottom: 8px;
}}
.ms-badge {{
    display: inline-block; padding: 3px 12px; border-radius: 999px;
    background: #E0F5F4; color: {PRIMARY}; font-weight: 600; font-size: .8rem;
    margin-right: 6px;
}}
.ms-note {{
    background: #F0FAFA; border-left: 4px solid {ACCENT};
    border-radius: 8px; padding: 10px 14px; color: #34545B; font-size: .92rem;
}}
</style>
"""


def apply_theme() -> None:
    """Inject the shared stylesheet. Call right after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    """Render a gradient banner at the top of a page."""
    st.markdown(
        f'<div class="ms-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, step: str | None = None) -> None:
    """Render a small info card."""
    badge = f'<div class="ms-step">{step}</div>' if step else ""
    st.markdown(
        f'<div class="ms-card">{badge}<h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def badge(text: str) -> str:
    """Return inline HTML for a pill badge."""
    return f'<span class="ms-badge">{text}</span>'
