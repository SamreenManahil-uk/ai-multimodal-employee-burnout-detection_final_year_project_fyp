from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib
import sys

import pandas as pd
import streamlit as st
from joblib import load

# ==========================================================
# BLOCK LOCAL OPENCV/PYTHON WEBCAM
# ==========================================================
import traceback
import cv2


def _blocked_video_capture(*args: Any, **kwargs: Any):
    print("\n" + "=" * 80)
    print("BLOCKED LOCAL OPENCV CAMERA ATTEMPT")
    print("=" * 80)
    traceback.print_stack()
    print("=" * 80 + "\n")

    raise RuntimeError(
        "Local OpenCV webcam access is disabled in the Streamlit application. "
        "Use st.camera_input() and pass the captured image to cv_predict.py."
    )


cv2.VideoCapture = _blocked_video_capture  # type: ignore[assignment]

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Multimodal Employee Burnout Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# THEME SYSTEM — Light & Dark, both navy/indigo brand family
# ==========================================================

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

LIGHT = {
    "bg": "linear-gradient(120deg, #FBD5DC 0%, #FDE7C8 25%, #E7F5EA 50%, #DCE6F6 75%, #E3DAF5 100%)",
    "bg_secondary": "#E7F5EA",
    "sidebar_bg": "#FBD5DC",
    "sidebar_bg2": "#E3DAF5",
    "sidebar_text": "#141414",
    "card_bg": "#FFFFFF",
    "card_border": "#CBD8EE",
    "card_hover_border": "#9A7FE0",
    "text_primary": "#141414",
    "text_secondary": "#4A4A4A",
    "navy": "#5B7FD9",
    "heading_text": "#0D0D0D",
    "header_grad_start": "#F49CB4",
    "header_grad_end": "#B39DE8",
    "accent": "#6C63D6",
    "iceblue": "#FFFFFF",
    "shadow": "rgba(220, 150, 170, 0.16)",
    "shadow_hover": "rgba(140, 110, 220, 0.30)",
    "input_bg": "#FFFFFF",
    "hover_dark_bg": "#CBB8F0",
    "hover_dark_text": "#141414",
    # Used only to scope the JSON-panel lemon-pastel fix (see build_css)
    # to the Light theme, as requested — Dark theme is untouched.
    "is_light": True,
}

DARK = {
    "bg": "linear-gradient(160deg, #0F1330 0%, #0C0F26 100%)",
    "bg_secondary": "#171C47",
    "sidebar_bg": "#0B0E24",
    "sidebar_bg2": "#080A1A",
    "sidebar_text": "#CADCFC",
    "card_bg": "#1B2153",
    "card_border": "#2C356F",
    "card_hover_border": "#8B9DEF",
    "text_primary": "#EDEFFC",
    "text_secondary": "#A8AFDB",
    "navy": "#2A3480",
    "heading_text": "#B9C6FF",
    "header_grad_start": "#2A3480",
    "header_grad_end": "#080A1A",
    "accent": "#8B9DEF",
    "iceblue": "#CADCFC",
    "shadow": "rgba(0, 0, 0, 0.35)",
    "shadow_hover": "rgba(107, 130, 222, 0.35)",
    "input_bg": "#171C47",
    "hover_dark_bg": "#080A1A",
    "hover_dark_text": "#FFFFFF",
    "is_light": False,
}

# Risk banner colours — now theme-aware (previously a single shared set
# that accidentally pastel-ified the Dark theme too when Light was fixed).
# LIGHT: soft pastel background + black text (as requested).
# DARK: original deep green/amber/red + white text, restored exactly as
# it was before the pastel change.
GREEN_LIGHT, GREEN_LIGHT_DARK = "#B8F2C5", "#8FE0A3"
AMBER_LIGHT, AMBER_LIGHT_DARK = "#FFE1A8", "#FFD27F"
RED_LIGHT, RED_LIGHT_DARK = "#FFC2BC", "#FFA39B"

GREEN_DARKTHEME, GREEN_DARKTHEME_DARK = "#1F7A3D", "#0F4A22"
AMBER_DARKTHEME, AMBER_DARKTHEME_DARK = "#C97A2B", "#7A3F0E"
RED_DARKTHEME, RED_DARKTHEME_DARK = "#C23B33", "#6E1A15"


def build_css(t: dict) -> str:
    # Pick risk-banner colours + banner text colour based on theme:
    # Light -> pastel backgrounds + black text. Dark -> original deep
    # colours + white text, exactly as it was before the pastel change.
    if t.get("is_light"):
        risk_green, risk_green_dark = GREEN_LIGHT, GREEN_LIGHT_DARK
        risk_amber, risk_amber_dark = AMBER_LIGHT, AMBER_LIGHT_DARK
        risk_red, risk_red_dark = RED_LIGHT, RED_LIGHT_DARK
        risk_banner_text = t['text_primary']
    else:
        risk_green, risk_green_dark = GREEN_DARKTHEME, GREEN_DARKTHEME_DARK
        risk_amber, risk_amber_dark = AMBER_DARKTHEME, AMBER_DARKTHEME_DARK
        risk_red, risk_red_dark = RED_DARKTHEME, RED_DARKTHEME_DARK
        risk_banner_text = "#FFFFFF"

    # ---- Added: JSON technical-details panels (ML/NLP/CV/Fusion details)
    # currently render with Streamlit's default dark code-block background
    # and blue-highlighted text, which is hard to read against the rest of
    # the light pastel UI. This block gives them a lemon/pastel-yellow
    # background with plain black text instead — built as a SEPARATE
    # Python string, only included when t['is_light'] is True, so the
    # Dark theme's JSON panels are left completely as they were before.
    json_panel_css = ""
    if t.get("is_light"):
        json_panel_css = """
    /* ---- Added: JSON technical-details panels, Light theme only ---- */
    [data-testid="stJson"] {
        background-color: #FFF9D6 !important;
        border: 1px solid #F0E6A8 !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    [data-testid="stJson"] * {
        color: #141414 !important;
        background-color: transparent !important;
    }
"""

    # ---- Added: dropdown polish (now applies to BOTH themes) ----------
    # 1) Selects were getting a border on BOTH their outer wrapper AND
    #    their inner value-container div, so they looked more "boxed"
    #    than the number inputs (Age, Years of experience) which only
    #    get a border on their single inner div. Removing the OUTER
    #    select border leaves just the one border, matching the number
    #    inputs exactly.
    # 2) The open dropdown list's hover/selected state was reading as a
    #    dark/black box in some browsers; this pins it to a pastel tint
    #    (hover_dark_bg) with the normal text colour, so it's always
    #    readable and always pastel, never black.
    # Every colour used below comes from the "t" dict, which already
    # holds the correct Light or Dark palette — so this single block
    # renders correctly for both themes without any extra branching.
    dropdown_polish_css = f"""
    /* The whole widget wrapper (label + dropdown box together) — remove
       any border/box-shadow/background Streamlit puts on the ENTIRE
       wrapper, not just the inner select box, since that outer wrapper
       is what was making label + box look like one solid white/bordered
       block. */
    [data-testid="stAppViewContainer"] .stSelectbox,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stElementContainer"]:has([data-baseweb="select"]) {{
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    /* The actual dropdown box itself — remove border AND box-shadow (a
       box-shadow can look identical to a border but "border: none"
       alone won't remove it). Keep only the single border already
       coming from the inner value-container div (matches Age / Years
       of experience number inputs exactly). */
    [data-testid="stAppViewContainer"] [data-baseweb="select"],
    section[data-testid="stSidebar"] [data-baseweb="select"] {{
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover,
    [data-testid="stAppViewContainer"] [role="option"]:hover,
    [data-testid="stAppViewContainer"] [role="option"][aria-selected="true"] {{
        background-color: {t['hover_dark_bg']} !important;
        color: {t['text_primary']} !important;
    }}
    /* Reinforced widget-label fix — duplicated here (guaranteed to be
       the LAST rule in the whole stylesheet) as a safety net, in case
       any other rule elsewhere was still re-whitening the label. */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        background-color: transparent !important;
        background: none !important;
        box-shadow: none !important;
    }}
"""

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Apply the custom font only to real text elements — NOT icon-font
       elements — so built-in Streamlit icons (sidebar collapse arrow,
       expander chevrons, etc.) keep rendering correctly. */
    body, p, span, div, label, h1, h2, h3, h4, h5, h6,
    button, input, textarea, li, a,
    .stMarkdown, .stText, .stButton, .stSelectbox, .stSlider, .stTextInput,
    .stTextArea, .stRadio, .stCheckbox, .stNumberInput {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}
    /* Explicitly protect Streamlit's icon font elements from the override */
    [data-testid="stIconMaterial"],
    [class*="material-icons"],
    span[data-testid*="Icon"],
    [data-testid="collapsedControl"] * {{
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }}

    /* ---------- Global page ---------- */
    .stApp {{
        background: {t['bg']};
        transition: background 0.35s ease;
    }}
    [data-testid="stAppViewContainer"] {{
        background: {t['bg']} !important;
    }}
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    header[data-testid="stHeader"],
    .stApp > header {{
        background: {t['bg']} !important;
        background-color: transparent !important;
    }}
    [data-testid="stDecoration"] {{
        display: none;
    }}
    h1, h2, h3, h4, p, span, label, .stMarkdown {{
        color: {t['text_primary']};
    }}

    /* ---------- Sidebar (theme-aware: light bg+dark text in Light theme,
       dark bg+light text in Dark theme) ---------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {t['sidebar_bg']} 0%, {t['sidebar_bg2']} 100%);
    }}
    section[data-testid="stSidebar"] * {{
        color: {t['sidebar_text']} !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {t['heading_text']} !important;
    }}
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label {{
        color: {t['sidebar_text']} !important;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: {t['accent']};
        opacity: 0.35;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] {{
        background-color: {t['card_bg']} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {{
        background-color: {t['card_bg']} !important;
        border-color: {t['card_border']} !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, background-color 0.25s ease;
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: {t['text_primary']} !important;
    }}
    /* Hover on sidebar dropdown -> turns dark, as requested */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {{
        background-color: {t['hover_dark_bg']} !important;
        border-color: {t['hover_dark_bg']} !important;
        box-shadow: 0 0 0 3px {t['shadow_hover']};
    }}
    section[data-testid="stSidebar"] [data-baseweb="select"]:hover * {{
        color: {t['hover_dark_text']} !important;
    }}

    /* ---------- Header banner ---------- */
    .app-header {{
        background: linear-gradient(120deg, {t['header_grad_start']} 0%, {t['header_grad_end']} 100%);
        border-radius: 16px;
        padding: 30px 36px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 30px {t['shadow']};
        animation: fadeInDown 0.6s cubic-bezier(.25,.8,.25,1);
    }}
    .app-header h1 {{
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
        line-height: 1.3;
    }}
    .app-header p {{
        color: {t['iceblue']} !important;
        font-size: 14.5px;
        margin: 0;
        max-width: 640px;
        line-height: 1.5;
    }}
    .about-card {{
        background-color: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        height: 100%;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }}
    .about-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px {t['shadow']};
        border-color: {t['hover_dark_bg']};
        background-color: {t['hover_dark_bg']};
    }}
    .about-card:hover .about-title,
    .about-card:hover .about-text {{
        color: {t['hover_dark_text']} !important;
    }}
    .about-card .about-icon {{ font-size: 26px; margin-bottom: 6px; }}
    .about-card .about-title {{
        font-weight: 700; font-size: 13.5px; color: {t['heading_text']}; margin-bottom: 4px;
    }}
    .about-card .about-text {{
        font-size: 12px; color: {t['text_secondary']}; line-height: 1.4;
    }}

    /* ---------- Section cards ---------- */
    .section-card {{
        background-color: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 14px;
        padding: 16px 22px;
        margin-bottom: 16px;
        transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
        animation: fadeInUp 0.5s cubic-bezier(.25,.8,.25,1);
    }}
    .section-card:hover {{
        border-color: {t['hover_dark_bg']};
        background-color: {t['hover_dark_bg']};
        box-shadow: 0 6px 18px {t['shadow']};
        transform: translateY(-2px);
    }}
    .section-card:hover h3 {{
        color: {t['hover_dark_text']} !important;
    }}
    .section-card h3 {{
        color: {t['heading_text']} !important;
        font-size: 16px;
        font-weight: 700;
        margin: 0;
    }}

    /* ---------- NUCLEAR FIX: force all dropdown/select elements in the
       MAIN content area (not sidebar) to white background + dark text,
       regardless of Streamlit's internal nested structure. ---------- */
    [data-testid="stAppViewContainer"] [data-baseweb="select"],
    [data-testid="stAppViewContainer"] [data-baseweb="select"] *,
    [data-testid="stAppViewContainer"] [data-baseweb="popover"],
    [data-testid="stAppViewContainer"] [data-baseweb="popover"] * {{
        background-color: {t['input_bg']} !important;
        color: {t['text_primary']} !important;
    }}
    [data-testid="stAppViewContainer"] [data-baseweb="select"] {{
        border: 1.5px solid {t['card_border']} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stAppViewContainer"] [role="option"]:hover,
    [data-testid="stAppViewContainer"] [role="option"][aria-selected="true"] {{
        background-color: {t['card_hover_border']}1A !important;
    }}

    /* ---------- Inputs (comprehensive, covers all nested BaseWeb parts) ---------- */
    .stTextInput input,
    .stNumberInput input,
    .stTextArea textarea,
    .stTextInput > div > div,
    .stNumberInput > div > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="base-input"] {{
        background-color: {t['input_bg']} !important;
        color: {t['text_primary']} !important;
        border-color: {t['card_border']} !important;
    }}
    .stTextInput > div, .stNumberInput > div, .stTextArea > div,
    div[data-baseweb="select"] > div:first-child {{
        background-color: {t['input_bg']} !important;
        border-radius: 8px !important;
        border: 1.5px solid {t['card_border']} !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {t['accent']} !important;
        box-shadow: 0 0 0 3px {t['shadow_hover']} !important;
    }}
    /* Number input +/- step buttons */
    .stNumberInput button {{
        background-color: {t['card_bg']} !important;
        color: {t['heading_text']} !important;
        border-color: {t['card_border']} !important;
    }}
    .stNumberInput button:hover {{
        background-color: {t['accent']} !important;
        color: #FFFFFF !important;
    }}
    /* Dropdown menu popover (the list that opens) */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {{
        background-color: {t['input_bg']} !important;
    }}
    [data-baseweb="menu"] li, [role="option"] {{
        color: {t['text_primary']} !important;
        background-color: {t['input_bg']} !important;
    }}
    [role="option"]:hover {{
        background-color: {t['card_hover_border']}22 !important;
    }}

    /* ---------- Buttons (including Streamlit's primary-button variant) ---------- */
    .stButton > button,
    .stButton > button[kind="primary"],
    .stButton > button[kind="secondary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="baseButton-secondary"] {{
        background: linear-gradient(120deg, {t['header_grad_start']} 0%, {t['header_grad_end']} 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.75em 1em !important;
        box-shadow: 0 4px 14px {t['shadow']};
        transition: transform 0.2s cubic-bezier(.34,1.56,.64,1), box-shadow 0.25s ease, background 0.3s ease;
    }}
    .stButton > button *,
    button[data-testid="baseButton-primary"] *,
    button[data-testid="baseButton-secondary"] * {{
        color: #FFFFFF !important;
    }}
    .stButton > button:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="baseButton-secondary"]:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 26px {t['shadow_hover']};
        background: linear-gradient(120deg, {t['accent']} 0%, {t['navy']} 100%) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0px) scale(0.99);
    }}

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: transparent;
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: {t['card_bg']};
        border-radius: 10px 10px 0 0;
        padding: 12px 22px;
        font-weight: 600;
        font-size: 14.5px;
        color: {t['text_secondary']} !important;
        transition: all 0.25s ease;
        white-space: nowrap;
        min-width: fit-content;
    }}
    .stTabs [data-baseweb="tab"] p {{
        color: inherit !important;
        margin: 0 !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {t['card_hover_border']}22;
        color: {t['heading_text']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(120deg, {t['header_grad_start']} 0%, {t['header_grad_end']} 100%) !important;
        color: #FFFFFF !important;
    }}
    .stTabs [aria-selected="true"] p {{
        color: #FFFFFF !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ---------- Metrics / progress ---------- */
    div[data-testid="stMetric"] {{
        background-color: {t['card_bg']};
        border: 1px solid {t['card_border']};
        border-radius: 12px;
        padding: 12px 14px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px {t['shadow']};
    }}
    div[data-testid="stMetricValue"] {{ color: {t['heading_text']} !important; font-weight: 800 !important; }}

    /* Sidebar metrics (Model Performance) — theme-aware, sidebar can be
       light or dark depending on the active theme */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {{
        background-color: {t['card_bg']} !important;
        border: 1px solid {t['card_border']} !important;
        border-radius: 12px;
        padding: 10px 12px;
        transition: background-color 0.25s ease, border-color 0.25s ease;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetric"]:hover {{
        background-color: {t['hover_dark_bg']} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetric"]:hover * {{
        color: {t['hover_dark_text']} !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {{
        color: {t['heading_text']} !important;
        font-weight: 800 !important;
        font-size: 22px !important;
    }}
    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"] {{
        color: {t['text_secondary']} !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stProgress"] > div > div {{
        background: linear-gradient(90deg, {t['accent']}, {t['navy']}) !important;
        transition: width 0.6s cubic-bezier(.25,.8,.25,1);
    }}

    /* ---------- Expanders ---------- */
    .streamlit-expanderHeader {{
        background-color: {t['card_bg']} !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: {t['heading_text']} !important;
        transition: background-color 0.25s ease;
    }}
    .streamlit-expanderHeader:hover {{
        background-color: {t['card_hover_border']}22 !important;
    }}
    {json_panel_css}

    /* ---------- Dropdown "white box, black text" class (ADDED) --------
       Fixes: Gender / Job role / Company size / Work mode / Job
       satisfaction / Manager support / Work-life balance / Social
       support / Stress level / Anxiety score / Depression score / Seeks
       professional help / Currently receiving therapy / ML pipeline —
       all still showing a solid black box in some browsers because
       Streamlit's own internal theme variables (which some of its
       widgets read directly, bypassing page CSS) were still dark. This
       pins those variables to match our palette, so no widget can fall
       back to Streamlit's default dark chrome. Combined with the
       existing nuclear-fix rules above, every dropdown box now stays
       {t['input_bg']} (white in Light theme) with {t['text_primary']} text. */
    :root, .stApp {{
        --primary-color: {t['accent']};
        --background-color: {t['card_bg']};
        --secondary-background-color: {t['card_bg']};
        --text-color: {t['text_primary']};
    }}
    [data-testid="stAppViewContainer"] .stSelectbox,
    [data-testid="stAppViewContainer"] .stSelectbox *,
    [data-testid="stAppViewContainer"] [role="combobox"],
    section[data-testid="stSidebar"] .stSelectbox,
    section[data-testid="stSidebar"] .stSelectbox *,
    section[data-testid="stSidebar"] [role="combobox"] {{
        background-color: {t['input_bg']} !important;
        color: {t['text_primary']} !important;
    }}
    /* Keep the dropdown-arrow icon's own hit-area transparent so it
       doesn't render as a solid coloured square. */
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg {{
        background: transparent !important;
    }}
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg rect,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg rect {{
        fill: transparent !important;
    }}
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg path,
    [data-testid="stAppViewContainer"] [data-baseweb="select"] svg polygon,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg path,
    section[data-testid="stSidebar"] [data-baseweb="select"] svg polygon {{
        fill: {t['text_secondary']} !important;
    }}
    /* Widget LABELS (e.g. "Gender", "Stress level", "ML pipeline") must
       stay plain text with no background box behind them — only the
       field box itself is white. IMPORTANT: this must be written with
       the SAME selector specificity as the ".stSelectbox *" white-
       background rule above (both use two chained selectors), otherwise
       CSS resolves the conflict by specificity, not by which rule comes
       later in the file — a plain "[data-testid=stWidgetLabel]" alone
       was actually LOSING to ".stSelectbox *" even though it appeared
       after it, which is exactly why the heading still showed a white
       box in the screenshot. */
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{
        background-color: transparent !important;
    }}

    /* ---------- File uploader "black box" fix (ADDED) -----------------
       Fixes: the "Upload a face image" dropzone/button (shows "Browse
       files", file-size limit text, and any uploaded filename), which
       was rendering with Streamlit's default dark box + white text. */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {{
        background-color: {t['input_bg']} !important;
        color: {t['text_primary']} !important;
        border-color: {t['card_border']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {t['card_bg']} !important;
        color: {t['text_primary']} !important;
        border: 1.5px solid {t['card_border']} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover {{
        background-color: {t['hover_dark_bg']} !important;
        color: {t['hover_dark_text']} !important;
    }}

    /* ---------- Reward badge (professional "task complete" moment) —
       ADDED for the reward transition request. A small checkmark-in-a-
       circle that traces itself, then a label fades in. Not confetti or
       balloons — a quiet, professional completion cue. ---------- */
    .reward-badge {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 4px 0 14px 0;
    }}
    .reward-check {{
        width: 30px;
        height: 30px;
        flex-shrink: 0;
    }}
    .reward-check-circle {{
        stroke: {t['accent']};
        stroke-width: 3;
        stroke-dasharray: 145;
        stroke-dashoffset: 145;
        animation: rewardCircleDraw 0.55s ease-out forwards;
    }}
    .reward-check-mark {{
        stroke: {t['accent']};
        stroke-width: 4;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-dasharray: 36;
        stroke-dashoffset: 36;
        animation: rewardCheckDraw 0.3s ease-out 0.5s forwards;
    }}
    .reward-badge-text {{
        font-size: 15px;
        font-weight: 700;
        color: {t['heading_text']};
        opacity: 0;
        transform: translateX(-6px);
        animation: rewardTextFade 0.35s ease-out 0.75s forwards;
    }}

    /* ---------- Risk result banner ---------- */
    .risk-banner {{
        position: relative;
        border-radius: 16px;
        padding: 28px 32px;
        margin: 10px 0 22px 0;
        color: {risk_banner_text};
        box-shadow: 0 12px 32px rgba(0,0,0,0.10);
        animation: rewardPop 0.65s cubic-bezier(.34,1.56,.64,1) 0.9s backwards,
                   glowPulse 1.4s ease-out 1.55s;
    }}
    /* One-time diagonal light sweep across the banner — part of the
       reward transition, added alongside the bouncier entrance above.
       Uses a soft dark tint (not white) now that the banners are pastel
       and light, so the sweep stays visible either way. */
    .risk-banner::after {{
        content: "";
        position: absolute;
        top: 0;
        left: -60%;
        width: 40%;
        height: 100%;
        background: linear-gradient(
            120deg,
            rgba(0,0,0,0) 0%,
            rgba(255,255,255,0.65) 50%,
            rgba(0,0,0,0) 100%
        );
        transform: skewX(-20deg);
        animation: shimmerSweep 1.1s ease-out 1.55s;
    }}
    .risk-banner h2 {{ margin: 0 0 4px 0; color: {risk_banner_text} !important; font-size: 25px; font-weight: 800; }}
    .risk-banner p {{ margin: 0; font-size: 14.5px; opacity: 0.94; }}
    .risk-low {{ background: linear-gradient(135deg, {risk_green} 0%, {risk_green_dark} 100%); }}
    .risk-moderate {{ background: linear-gradient(135deg, {risk_amber} 0%, {risk_amber_dark} 100%); }}
    .risk-high {{ background: linear-gradient(135deg, {risk_red} 0%, {risk_red_dark} 100%); }}

    /* ---------- Component result cards ---------- */
    .component-card {{
        border-radius: 14px;
        padding: 18px 18px;
        text-align: center;
        border: 1px solid {t['card_border']};
        background-color: {t['card_bg']};
        height: 100%;
        transition: transform 0.25s cubic-bezier(.34,1.56,.64,1), box-shadow 0.25s ease, border-color 0.25s ease;
        animation: fadeInUp 0.55s cubic-bezier(.25,.8,.25,1);
    }}
    .component-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 26px {t['shadow']};
        border-color: {t['hover_dark_bg']};
        background-color: {t['hover_dark_bg']};
    }}
    .component-card:hover .comp-title,
    .component-card:hover .comp-score {{
        color: {t['hover_dark_text']} !important;
    }}
    .component-card .comp-title {{
        font-weight: 700; color: {t['heading_text']}; font-size: 12.5px;
        text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px;
    }}
    .component-card .comp-score {{ font-size: 32px; font-weight: 800; color: {t['text_primary']}; }}
    .component-card.unavailable {{ opacity: 0.55; }}

    /* ---------- Animations ---------- */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes popIn {{
        0% {{ opacity: 0; transform: scale(0.94); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    /* ---- Added for the reward transition ---- */
    @keyframes rewardPop {{
        0%   {{ opacity: 0; transform: scale(0.80) translateY(10px); }}
        60%  {{ opacity: 1; transform: scale(1.03) translateY(0); }}
        100% {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    @keyframes shimmerSweep {{
        0%   {{ left: -60%; }}
        100% {{ left: 130%; }}
    }}
    @keyframes glowPulse {{
        0%   {{ box-shadow: 0 12px 32px rgba(0,0,0,0.18), 0 0 0px rgba(255,255,255,0); }}
        40%  {{ box-shadow: 0 12px 32px rgba(0,0,0,0.18), 0 0 40px rgba(255,255,255,0.55); }}
        100% {{ box-shadow: 0 12px 32px rgba(0,0,0,0.18), 0 0 0px rgba(255,255,255,0); }}
    }}
    @keyframes rewardCircleDraw {{
        to {{ stroke-dashoffset: 0; }}
    }}
    @keyframes rewardCheckDraw {{
        to {{ stroke-dashoffset: 0; }}
    }}
    @keyframes rewardTextFade {{
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    /* Placed LAST on purpose: this must come after the NUCLEAR FIX
       border rule and the Inputs-block border rule (both re-apply a
       border to [data-baseweb="select"] with the exact same CSS
       specificity), so whichever rule appears LAST in the file wins.
       Putting dropdown_polish_css here — after everything else —
       guarantees it always wins that tie, instead of losing silently
       like it was before. */
    {dropdown_polish_css}
</style>
"""


theme_dict = DARK if st.session_state.theme_mode == "Dark" else LIGHT
st.markdown(build_css(theme_dict), unsafe_allow_html=True)

# ==========================================================
# PROJECT PATHS
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = [
    CURRENT_DIR,
    CURRENT_DIR.parent,
    CURRENT_DIR.parent.parent,
]

PROJECT_ROOT = next(
    (folder for folder in ROOT_CANDIDATES if (folder / "models").exists()),
    CURRENT_DIR,
)

MODELS_DIR = PROJECT_ROOT / "models"

for folder in [
    CURRENT_DIR,
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "machine_learning",
    PROJECT_ROOT / "nlp",
    PROJECT_ROOT / "computer_vision",
]:
    if folder.exists() and str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from nlp_predict import analyse_employee_feedback
from cv_predict import analyse_face_image
from fusion import fuse_burnout_scores
from save_assessment import save_complete_assessment

# ==========================================================
# ML IMPORT
# ==========================================================

try:
    ml_predict_module = importlib.import_module("ml_predict")
    predict_employee_burnout = ml_predict_module.predict_employee_burnout

    if hasattr(ml_predict_module, "PROJECT_ROOT"):
        ml_predict_module.PROJECT_ROOT = PROJECT_ROOT

    if hasattr(ml_predict_module, "MODELS_DIR"):
        ml_predict_module.MODELS_DIR = MODELS_DIR

except Exception as exc:
    st.error("Could not import ml_predict.py.")
    st.exception(exc)
    st.stop()

# ==========================================================
# SHAP IMPORT (ADDED — live local explanation)
# Non-fatal on purpose: SHAP is a bonus explainability feature, not a
# core requirement, so if the 'shap' package isn't installed or the
# module fails to import for any reason, the rest of the app must keep
# working exactly as before. explain_employee_prediction is left as
# None in that case, and every call site checks for that before using it.
# ==========================================================

try:
    # IMPORTANT: force a fresh reload every script run instead of using
    # whatever "shap_explain" module object Python already has cached in
    # sys.modules. Streamlit's "rerun on file save" reruns app.py itself,
    # but it does NOT automatically re-import already-imported submodules
    # — so editing shap_explain.py and letting Streamlit auto-rerun was
    # silently still running the OLD in-memory version (missing new
    # parameters like "theme"), even though the file on disk was correct.
    # Deleting it from sys.modules first guarantees import_module() below
    # actually re-reads the current file from disk every time.
    if "shap_explain" in sys.modules:
        del sys.modules["shap_explain"]
    shap_explain_module = importlib.import_module("shap_explain")
    explain_employee_prediction = shap_explain_module.explain_employee_prediction
except Exception:
    explain_employee_prediction = None

# ==========================================================
# SETTINGS (unchanged from original — logic untouched)
# ==========================================================

PIPELINE_LABELS = {
    "feature_engineered": "Feature-engineered pipeline",
    "baseline": "Baseline pipeline",
}

ENGINEERED_FEATURES = {
    "mental_health_index",
    "workload_index",
    "work_life_support_index",
    "lifestyle_index",
    "screen_workload_interaction",
}

FIELD_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "job_role": "Job role",
    "experience_years": "Years of experience",
    "company_size": "Company size",
    "work_mode": "Work mode",
    "work_hours_per_week": "Work hours per week",
    "overtime_hours": "Overtime hours per week",
    "meetings_per_day": "Meetings per day",
    "deadlines_missed": "Deadlines missed",
    "job_satisfaction": "Job satisfaction",
    "manager_support": "Manager support",
    "work_life_balance": "Work-life balance",
    "sleep_hours": "Sleep hours per night",
    "physical_activity_days": "Physical activity days per week",
    "screen_time_hours": "Screen time per day",
    "caffeine_intake": "Daily caffeine intake",
    "social_support_score": "Social support score",
    "has_therapy": "Currently receiving therapy",
    "stress_level": "Stress level",
    "anxiety_score": "Anxiety score",
    "depression_score": "Depression score",
    "seeks_professional_help": "Seeks professional help",
}

HELP_TEXT = {
    "work_hours_per_week": "Enter the employee's usual weekly working hours.",
    "overtime_hours": "Enter additional hours worked beyond normal contracted hours.",
    "meetings_per_day": "Average number of meetings on a working day.",
    "deadlines_missed": "Use the same definition used in the training dataset.",
    "job_satisfaction": "Choose the level that best describes current job satisfaction.",
    "manager_support": "Choose how supported the employee feels by their manager.",
    "work_life_balance": "Choose the level that best describes work-life balance.",
    "stress_level": "Choose the employee's current level of work-related stress.",
    "anxiety_score": "Choose the level that best describes current anxiety symptoms.",
    "depression_score": "Choose the level that best describes current low-mood symptoms.",
}

# ==========================================================
# METADATA
# ==========================================================

@st.cache_resource
def load_metadata(pipeline_name: str) -> dict[str, Any]:
    metadata_path = MODELS_DIR / pipeline_name / "ml_metadata.joblib"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = load(metadata_path)

    if not isinstance(metadata, dict):
        raise TypeError("The saved ML metadata must be a dictionary.")

    return metadata


def available_pipelines() -> list[str]:
    available: list[str] = []

    for pipeline_name in PIPELINE_LABELS:
        pipeline_dir = MODELS_DIR / pipeline_name

        required_files = [
            pipeline_dir / "best_model.joblib",
            pipeline_dir / "onehot_encoder.joblib",
            pipeline_dir / "ml_metadata.joblib",
        ]

        if all(path.exists() for path in required_files):
            available.append(pipeline_name)

    return available

# ==========================================================
# FORM HELPERS (logic unchanged)
# ==========================================================

def readable_label(column_name: str) -> str:
    return FIELD_LABELS.get(
        column_name,
        column_name.replace("_", " ").title(),
    )


def field_group(column_name: str) -> str:
    personal = {
        "age", "gender", "job_role", "experience_years",
        "company_size", "work_mode",
    }

    workload = {
        "work_hours_per_week", "overtime_hours", "meetings_per_day",
        "deadlines_missed", "screen_time_hours", "caffeine_intake",
    }

    wellbeing = {
        "job_satisfaction", "manager_support", "work_life_balance",
        "sleep_hours", "physical_activity_days", "social_support_score",
        "has_therapy", "stress_level", "anxiety_score",
        "depression_score", "seeks_professional_help",
    }

    if column_name in personal:
        return "Personal and employment information"

    if column_name in workload:
        return "Workload and working pattern"

    if column_name in wellbeing:
        return "Wellbeing and support"

    return "Additional information"


RATING_FIELDS = {
    "job_satisfaction", "manager_support", "work_life_balance",
    "social_support_score", "stress_level", "anxiety_score",
    "depression_score",
}

RATING_LEVEL_VALUES = {
    "Low": 2.0,
    "Moderate": 5.0,
    "High": 8.0,
}

BOOLEAN_FIELDS = {"has_therapy", "seeks_professional_help"}

GROUP_ICONS = {
    "Personal and employment information": "👤",
    "Workload and working pattern": "📊",
    "Wellbeing and support": "💚",
    "Additional information": "📋",
}


def render_rating_widget(column_name: str) -> float:
    selected_label = st.selectbox(
        readable_label(column_name),
        options=list(RATING_LEVEL_VALUES.keys()),
        index=1,
        help=HELP_TEXT.get(column_name),
        key=f"field_{column_name}",
    )
    return float(RATING_LEVEL_VALUES[selected_label])


def render_boolean_widget(column_name: str) -> int:
    selected = st.selectbox(
        readable_label(column_name),
        options=["No", "Yes"],
        help=HELP_TEXT.get(column_name),
        key=f"field_{column_name}",
    )
    return 1 if selected == "Yes" else 0


def numeric_widget_settings(column_name: str) -> dict[str, Any]:
    settings: dict[str, dict[str, Any]] = {
        "age": {"min_value": 18, "max_value": 80, "value": 30, "step": 1},
        "experience_years": {
            "min_value": 0.0, "max_value": 60.0, "value": 5.0, "step": 0.5
        },
        "work_hours_per_week": {
            "min_value": 0.0, "max_value": 100.0, "value": 40.0, "step": 1.0
        },
        "overtime_hours": {
            "min_value": 0.0, "max_value": 60.0, "value": 5.0, "step": 1.0
        },
        "meetings_per_day": {
            "min_value": 0.0, "max_value": 20.0, "value": 3.0, "step": 1.0
        },
        "deadlines_missed": {
            "min_value": 0.0, "max_value": 50.0, "value": 0.0, "step": 1.0
        },
        "sleep_hours": {
            "min_value": 0.0, "max_value": 14.0, "value": 7.0, "step": 0.5
        },
        "physical_activity_days": {
            "min_value": 0.0, "max_value": 7.0, "value": 3.0, "step": 1.0
        },
        "screen_time_hours": {
            "min_value": 0.0, "max_value": 24.0, "value": 8.0, "step": 0.5
        },
        "caffeine_intake": {
            "min_value": 0.0, "max_value": 20.0, "value": 2.0, "step": 1.0
        },
    }

    if column_name in RATING_FIELDS:
        return {
            "min_value": 0.0,
            "max_value": 10.0,
            "value": 5.0,
            "step": 1.0,
        }

    return settings.get(
        column_name,
        {
            "min_value": 0.0,
            "max_value": 1000.0,
            "value": 0.0,
            "step": 1.0,
        },
    )


def render_categorical_widget(column_name: str, options: list[str]) -> str:
    clean_options = [str(option) for option in options]

    if not clean_options:
        return st.text_input(
            readable_label(column_name),
            help=HELP_TEXT.get(column_name),
            key=f"field_{column_name}",
        )

    return st.selectbox(
        readable_label(column_name),
        options=clean_options,
        help=HELP_TEXT.get(column_name),
        key=f"field_{column_name}",
    )


def render_numerical_widget(column_name: str) -> float | int:
    if column_name in RATING_FIELDS:
        return render_rating_widget(column_name)

    if column_name in BOOLEAN_FIELDS:
        return render_boolean_widget(column_name)

    settings = numeric_widget_settings(column_name)

    return st.number_input(
        readable_label(column_name),
        min_value=settings["min_value"],
        max_value=settings["max_value"],
        value=settings["value"],
        step=settings["step"],
        help=HELP_TEXT.get(column_name),
        key=f"field_{column_name}",
    )


def render_employee_form(metadata: dict[str, Any]) -> dict[str, Any]:
    original_columns = list(metadata.get("original_input_columns", []))
    categorical_columns = set(metadata.get("categorical_columns", []))
    categorical_values = metadata.get(
        "categorical_values_seen_during_training",
        {},
    )

    input_columns = [
        column
        for column in original_columns
        if column not in ENGINEERED_FEATURES
    ]

    groups = [
        "Personal and employment information",
        "Workload and working pattern",
        "Wellbeing and support",
        "Additional information",
    ]

    employee_data: dict[str, Any] = {}

    for group in groups:
        group_columns = [
            column for column in input_columns
            if field_group(column) == group
        ]

        if not group_columns:
            continue

        icon = GROUP_ICONS.get(group, "•")
        st.markdown(
            f'<div class="section-card"><h3>{icon}&nbsp; {group}</h3></div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns(2)

        for index, column_name in enumerate(group_columns):
            target_column = left if index % 2 == 0 else right

            with target_column:
                if column_name in categorical_columns:
                    employee_data[column_name] = render_categorical_widget(
                        column_name=column_name,
                        options=categorical_values.get(column_name, []),
                    )
                else:
                    employee_data[column_name] = render_numerical_widget(
                        column_name
                    )

        st.write("")

    return employee_data

# ==========================================================
# RESULT HELPERS (redesigned visuals, same underlying logic)
# ==========================================================

def risk_band(prediction: int) -> tuple[str, str, str]:
    """Returns (band, explanation, css_class) — BINARY, matching the
    model's actual Low/Elevated classification (fusion_result["prediction"]
    is 0 or 1), not an arbitrary 3-way score split. This keeps the banner
    title and the "Decision:" line always in agreement, since both now
    come from the same underlying 0/1 value instead of two different
    calculations that could disagree (a continuous score cut into three
    bands vs. a binary prediction flag)."""
    if prediction == 1:
        return (
            "Elevated",
            "The combined model indicates elevated burnout risk.",
            "risk-high",
        )

    return (
        "Low",
        "The combined model indicates comparatively low burnout risk.",
        "risk-low",
    )


def show_component_result(
    title: str,
    result: dict[str, Any],
    score_key: str,
) -> None:
    available = bool(result.get("available", result.get("ml_available", False)))

    if not available:
        st.markdown(
            f'<div class="component-card unavailable">'
            f'<div class="comp-title">{title}</div>'
            f'<div class="comp-score">N/A</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        error_msg = result.get("error", f"{title} was unavailable.")
        st.caption(f"⚠️ {error_msg}")
        return

    score = float(result[score_key])
    st.markdown(
        f'<div class="component-card">'
        f'<div class="comp-title">{title}</div>'
        f'<div class="comp-score">{score * 100:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    details = {
        key: value
        for key, value in result.items()
        if key not in {score_key, "available", "ml_available"}
    }

    if details:
        with st.expander(f"{title} technical details"):
            st.json(details)


def display_final_result(
    ml_result: dict[str, Any],
    nlp_result: dict[str, Any],
    cv_result: dict[str, Any],
    fusion_result: dict[str, Any],
) -> None:
    st.markdown("## 📋 Final Multimodal Assessment")

    score = float(fusion_result["fusion_score"])
    prediction = int(fusion_result["prediction"])
    band, explanation, css_class = risk_band(prediction)
    decision = "Elevated risk" if prediction == 1 else "Low risk"

    st.markdown(
        f'<div class="risk-banner {css_class}">'
        f'<h2>{band} Risk &nbsp;·&nbsp; {score * 100:.1f}%</h2>'
        f'<p>{explanation} &nbsp;|&nbsp; Decision: <strong>{decision}</strong></p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.progress(score)
    st.write("")

    component_1, component_2, component_3 = st.columns(3)

    with component_1:
        show_component_result("ML", ml_result, "ml_score")

    with component_2:
        show_component_result("NLP", nlp_result, "nlp_score")

    with component_3:
        show_component_result("CV", cv_result, "cv_score")

    st.write("")
    with st.expander("🔍 Fusion details (weights, contributions)"):
        st.json(fusion_result)

    st.warning(
        "This is an academic decision-support prototype, not a medical "
        "diagnosis. Do not use it as the only basis for health or employment "
        "decisions."
    )


def display_local_shap_explanation(shap_result: dict[str, Any]) -> None:
    """
    ADDED: shows the live LOCAL SHAP explanation for this specific
    prediction — i.e. which features pushed THIS employee's score up or
    down, and by how much. Purely additive: only called from main() if
    explain_employee_prediction was successfully imported AND returned
    shap_available=True, so it can never break the rest of the results.
    """
    if not shap_result.get("shap_available"):
        st.caption(
            "⚠️ Local SHAP explanation unavailable: "
            f"{shap_result.get('error', 'unknown error')}"
        )
        return

    st.markdown(
        '<div class="section-card"><h3>🔬&nbsp; Why This Prediction? '
        '(Local SHAP Explanation)</h3></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This waterfall shows exactly which factors pushed THIS employee's "
        "ML score up (red) or down (blue), starting from the model's "
        "average output and ending at this specific prediction — as "
        "opposed to global feature importance, which describes the model "
        "as a whole rather than this one case."
    )

    st.pyplot(shap_result["waterfall_figure"])

    saved_path = shap_result.get("saved_plot_path")
    if saved_path:
        st.caption(f"💾 Plot saved to: `{saved_path}`")

    top_features = shap_result.get("top_features") or []
    if top_features:
        with st.expander("📋 Top contributing features (exact SHAP values)"):
            top_features_table = pd.DataFrame(top_features[:10])
            top_features_table = top_features_table.rename(
                columns={
                    "feature": "Feature",
                    "feature_value": "Value",
                    "shap_value": "SHAP contribution",
                }
            )
            # Colourful in-cell bars on the SHAP contribution column:
            # bars grow left (blue, lowers risk) or right (pink, raises
            # risk) from the zero-point, so the table reads at a glance
            # instead of needing to parse raw numbers. Falls back to a
            # plain table if Styler rendering isn't available for any
            # reason (e.g. jinja2 missing) — never breaks the results page.
            try:
                styled_table = (
                    top_features_table.style
                    .format({"Value": "{:.2f}", "SHAP contribution": "{:+.4f}"})
                    .bar(
                        subset=["SHAP contribution"],
                        align="mid",
                        color=["#6FA8E8", "#F49CB4"],
                    )
                    .set_properties(**{"font-weight": "600"}, subset=["Feature"])
                )
                st.dataframe(styled_table, use_container_width=True, hide_index=True)
            except Exception:
                st.dataframe(top_features_table, use_container_width=True, hide_index=True)

# ==========================================================
# APPLICATION
# ==========================================================

def main() -> None:
    st.markdown(
        '<div class="app-header">'
        '<h1>🧠 Multimodal Employee Burnout Assessment</h1>'
        '<p>Combining a structured questionnaire (ML), written feedback (NLP), '
        'and facial expression (CV) into one explainable risk assessment.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    about_col1, about_col2, about_col3, about_col4 = st.columns(4)
    about_cards = [
        ("📊", "Structured Data", "Workload, satisfaction, stress and support indicators, scored by XGBoost."),
        ("💬", "Written Feedback", "Free-text comments analysed by a zero-shot NLP model for exhaustion signals."),
        ("📷", "Facial Expression", "An optional face image analysed for visible signs of negative affect."),
        ("🔍", "Explainable Result", "Every score is combined transparently and explained using SHAP."),
    ]
    for col, (icon, title, text) in zip(
        [about_col1, about_col2, about_col3, about_col4], about_cards
    ):
        with col:
            st.markdown(
                f'<div class="about-card">'
                f'<div class="about-icon">{icon}</div>'
                f'<div class="about-title">{title}</div>'
                f'<div class="about-text">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.write("")

    pipelines = available_pipelines()

    if not pipelines:
        st.error("No complete archived ML pipeline was found.")
        st.code(
            "models/feature_engineered/best_model.joblib\n"
            "models/feature_engineered/onehot_encoder.joblib\n"
            "models/feature_engineered/ml_metadata.joblib"
        )
        st.stop()

    with st.sidebar:
        st.markdown("### 🎨 Appearance")
        theme_choice = st.radio(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme_mode == "Light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        if theme_choice != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_choice
            st.rerun()

        st.divider()
        st.markdown("### 📈 Model Performance")
        stat_col1, stat_col2 = st.columns(2)
        stat_col1.metric("Accuracy", "92.8%")
        stat_col2.metric("ROC-AUC", "0.958")
        st.caption("XGBoost, validated on 10,000 held-out test records.")

        st.divider()
        st.markdown("### ⚙️ Assessment Settings")

        selected_pipeline = st.selectbox(
            "ML pipeline",
            options=pipelines,
            format_func=lambda value: PIPELINE_LABELS[value],
        )

        DEFAULT_WEIGHTS = {"ml": 0.50, "nlp": 0.30, "cv": 0.20}

        if "ml_weight" not in st.session_state:
            st.session_state.ml_weight = DEFAULT_WEIGHTS["ml"]
            st.session_state.nlp_weight = DEFAULT_WEIGHTS["nlp"]
            st.session_state.cv_weight = DEFAULT_WEIGHTS["cv"]

        st.markdown("**Fusion weights**")
        ml_weight = st.slider("ML weight", 0.0, 1.0, key="ml_weight", step=0.05)
        nlp_weight = st.slider("NLP weight", 0.0, 1.0, key="nlp_weight", step=0.05)
        cv_weight = st.slider("CV weight", 0.0, 1.0, key="cv_weight", step=0.05)

        weights_changed = (
            round(ml_weight, 2) != DEFAULT_WEIGHTS["ml"]
            or round(nlp_weight, 2) != DEFAULT_WEIGHTS["nlp"]
            or round(cv_weight, 2) != DEFAULT_WEIGHTS["cv"]
        )

        if weights_changed:
            st.warning(
                "⚠️ Custom weights in use — this is **not** the validated "
                "configuration (0.50 / 0.30 / 0.20). Results below will not "
                "match the reported research findings."
            )
            if st.button("↺ Reset to Validated Weights", use_container_width=True):
                st.session_state.ml_weight = DEFAULT_WEIGHTS["ml"]
                st.session_state.nlp_weight = DEFAULT_WEIGHTS["nlp"]
                st.session_state.cv_weight = DEFAULT_WEIGHTS["cv"]
                st.rerun()
        else:
            st.caption("✅ Using the validated weight configuration.")

        st.caption(
            "Weights are automatically normalised. These sliders are provided "
            "for demonstration and exploration — the validated configuration "
            "above was determined through evidence-based justification, not "
            "exhaustive optimisation (see report, Section V)."
        )

        threshold = st.slider(
            "Final decision threshold",
            0.0,
            1.0,
            0.50,
            0.01,
        )

        st.divider()
        st.markdown(
            "🔒 **Privacy**\n\nDo not enter names, email addresses, employee "
            "IDs, or other directly identifying information."
        )

    try:
        metadata = load_metadata(selected_pipeline)
    except Exception as exc:
        st.error("The selected ML metadata could not be loaded.")
        st.exception(exc)
        st.stop()

    tab_ml, tab_nlp, tab_cv = st.tabs(
        ["📊  1. ML Questionnaire", "💬  2. NLP Feedback", "📷  3. CV Face Capture"]
    )

    with tab_ml:
        employee_data = render_employee_form(metadata)

    with tab_nlp:
        st.markdown(
            '<div class="section-card"><h3>💬&nbsp; Employee Feedback</h3></div>',
            unsafe_allow_html=True,
        )
        feedback_text = st.text_area(
            "Employee feedback",
            height=180,
            placeholder=(
                "Example: I feel exhausted after work and find it difficult "
                "to concentrate or recover before the next day."
            ),
            label_visibility="collapsed",
        )

    with tab_cv:
        st.markdown(
            '<div class="section-card"><h3>📷&nbsp; Face Capture</h3></div>',
            unsafe_allow_html=True,
        )
        image_source = st.radio(
            "Image source",
            ["Use camera", "Upload image"],
            horizontal=True,
        )

        if image_source == "Use camera":
            face_image = st.camera_input("Capture a clear front-facing image")
        else:
            face_image = st.file_uploader(
                "Upload a face image",
                type=["jpg", "jpeg", "png"],
            )

        st.caption(
            "Use one clear, front-facing face. Only the browser camera shown "
            "inside this Streamlit page should be used. The CV score is an "
            "experimental visual signal and must not be treated as a clinical result."
        )

    st.write("")
    consent = st.checkbox(
        "I confirm this assessment is authorised and the participant has "
        "consented to the processing of questionnaire, text, and image data."
    )

    submitted = st.button(
        "🚀  Run Complete Multimodal Assessment",
        type="primary",
        use_container_width=True,
    )

    if not submitted:
        return

    if not consent:
        st.warning("Please confirm consent before running the assessment.")
        return

    if not feedback_text.strip():
        st.warning("Please enter employee feedback for NLP analysis.")
        return

    if face_image is None:
        st.warning("Please capture or upload a face image for CV analysis.")
        return

    with st.spinner("Running ML, NLP, CV and fusion pipelines..."):
        ml_result = predict_employee_burnout(
            employee_data=employee_data,
            pipeline_name=selected_pipeline,
        )

        ml_result["available"] = bool(ml_result.get("ml_available", False))

        nlp_result = analyse_employee_feedback(
            feedback_text=feedback_text,
            project_root=PROJECT_ROOT,
        )

        cv_result = analyse_face_image(
            uploaded_image=face_image,
            project_root=PROJECT_ROOT,
        )

        fusion_result = fuse_burnout_scores(
            ml_result=ml_result,
            nlp_result=nlp_result,
            cv_result=cv_result,
            weights={
                "ml": ml_weight,
                "nlp": nlp_weight,
                "cv": cv_weight,
            },
            threshold=threshold,
        )

        saved_path = save_complete_assessment(
            employee_data=employee_data,
            feedback_text=feedback_text,
            ml_result=ml_result,
            nlp_result=nlp_result,
            cv_result=cv_result,
            fusion_result=fusion_result,
            project_root=PROJECT_ROOT,
        )

        # ---- Added: live LOCAL SHAP explanation for this one prediction.
        # Only attempted if the shap_explain module imported successfully
        # AND the ML prediction itself actually succeeded — a SHAP
        # failure here must never break the rest of the results.
        # theme is passed through so the plot's colours match whichever
        # theme the person currently has selected (Light or Dark). Wrapped
        # defensively: if an older cached version of explain_employee_prediction
        # doesn't accept "theme" yet, fall back to calling without it
        # rather than crashing the whole results page.
        shap_result = None
        if explain_employee_prediction is not None and ml_result.get("ml_available"):
            try:
                shap_result = explain_employee_prediction(
                    employee_data=employee_data,
                    pipeline_name=selected_pipeline,
                    theme=st.session_state.theme_mode.lower(),
                )
            except TypeError:
                shap_result = explain_employee_prediction(
                    employee_data=employee_data,
                    pipeline_name=selected_pipeline,
                )

    # ---- Added for the reward transition ----
    st.markdown(
        '<div class="reward-badge">'
        '<svg class="reward-check" viewBox="0 0 52 52">'
        '<circle class="reward-check-circle" cx="26" cy="26" r="23" fill="none"/>'
        '<path class="reward-check-mark" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>'
        '</svg>'
        '<span class="reward-badge-text">Assessment Complete</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    display_final_result(
        ml_result=ml_result,
        nlp_result=nlp_result,
        cv_result=cv_result,
        fusion_result=fusion_result,
    )

    # ---- Added: show the local SHAP explanation right after the main
    # result, only if it was successfully computed above.
    if shap_result is not None:
        display_local_shap_explanation(shap_result)

    st.success(f"✅ Assessment record saved to: {saved_path}")


if __name__ == "__main__":
    main()