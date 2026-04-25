import streamlit as st
import time
import json
import os
import html
from datetime import datetime
from pipeline import RAGPipeline

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity University RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS (red + white theme) ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=Outfit:wght@300;500;600&display=swap');

:root {
  --rw-red: #B91C1C;
  --rw-red-dark: #7F1D1D;
  --rw-red-light: #FECACA;
  --rw-white: #FFFFFF;
  --rw-cream: #FFFBFB;
  --rw-text: #1C1917;
  --rw-muted: #57534E;
  --rw-border: rgba(185, 28, 28, 0.2);
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--rw-cream) !important;
    color: var(--rw-text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

section.main > div {
    background: var(--rw-cream) !important;
}

/* Hide Streamlit app menu & footer; keep <header> usable so sidebar toggle is not removed */
#MainMenu { visibility: hidden !important; height: 0 !important; max-height: 0 !important; overflow: hidden !important; }
footer { visibility: hidden !important; height: 0 !important; max-height: 0 !important; overflow: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Streamlit top chrome (Deploy, toolbar): light bar + dark text — fixes black bar / invisible labels */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
  background-color: #FAFAF9 !important;
  background-image: none !important;
  color: #1C1917 !important;
  border-bottom: 1px solid rgba(185, 28, 28, 0.12) !important;
}
[data-testid="stHeader"] a,
[data-testid="stHeader"] button,
[data-testid="stHeader"] span,
[data-testid="stHeader"] p,
[data-testid="stHeader"] label {
  color: #1C1917 !important;
}
[data-testid="stHeader"] a:hover,
[data-testid="stHeader"] button:hover {
  color: #B91C1C !important;
}
[data-testid="stHeader"] svg,
[data-testid="stHeader"] path {
  fill: #1C1917 !important;
  stroke: #1C1917 !important;
  color: #1C1917 !important;
}
[data-testid="stToolbar"],
[data-testid="stToolbarActions"] {
  background-color: transparent !important;
  color: #1C1917 !important;
}
[data-testid="stDeployButton"] a,
[data-testid="stDeployButton"] button,
[data-testid="stDeployButton"] span {
  color: #1C1917 !important;
}
/* Fallback when Streamlit wraps chrome in a plain <header> */
.stApp > header {
  background-color: #FAFAF9 !important;
  color: #1C1917 !important;
}

/* Sidebar expand/collapse — obvious control (Streamlit test ids vary by version) */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  z-index: 999992 !important;
}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] [data-baseweb="button"],
[data-testid="collapsedControl"] button {
  background: linear-gradient(135deg, #DC2626, #B91C1C) !important;
  border: 2px solid rgba(255, 255, 255, 0.45) !important;
  border-radius: 10px !important;
  box-shadow: 0 2px 14px rgba(0, 0, 0, 0.22) !important;
  min-width: 2.75rem !important;
  min-height: 2.75rem !important;
  color: #fff !important;
}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapseButton"] [data-baseweb="button"]:hover,
[data-testid="collapsedControl"] button:hover {
  filter: brightness(1.08) !important;
}

/* ── Sidebar: layered “control deck” (red + white + gold accent) ── */
[data-testid="stSidebar"] {
    position: relative !important;
    overflow-x: hidden !important;
    background:
        radial-gradient(ellipse 120% 80% at 100% 0%,
            rgba(255, 255, 255, 0.12) 0%,
            transparent 55%),
        radial-gradient(ellipse 100% 60% at 0% 100%,
            rgba(0, 0, 0, 0.35) 0%,
            transparent 50%),
        repeating-linear-gradient(
            -28deg,
            rgba(0, 0, 0, 0) 0px,
            rgba(0, 0, 0, 0) 11px,
            rgba(0, 0, 0, 0.04) 11px,
            rgba(0, 0, 0, 0.04) 12px
        ),
        linear-gradient(170deg, #9F1919 0%, #5C1010 48%, #3F0B0B 100%) !important;
    border-right: none !important;
    box-shadow:
        inset 0 1px 0 rgba(255, 255, 255, 0.12),
        4px 0 32px rgba(0, 0, 0, 0.25) !important;
    padding: 0 !important;
}
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    z-index: 0;
    background: linear-gradient(
        180deg,
        #FEF3C7 0%,
        #FDE68A 18%,
        rgba(255, 255, 255, 0.5) 50%,
        #FCA5A5 78%,
        #B91C1C 100%
    );
    box-shadow: 2px 0 12px rgba(0, 0, 0, 0.2);
    pointer-events: none;
}
[data-testid="stSidebar"]::after {
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 1px;
    z-index: 0;
    background: linear-gradient(
        180deg,
        rgba(255, 255, 255, 0.4) 0%,
        rgba(255, 255, 255, 0.05) 50%,
        rgba(0, 0, 0, 0.15) 100%
    );
    pointer-events: none;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] small {
    color: rgba(255, 255, 255, 0.95) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="input"] {
    color: var(--rw-text) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* Sidebar: glass panels around controls */
[data-testid="stSidebar"] [data-baseweb="slider"] {
    padding: 4px 2px 8px;
}
[data-testid="stSidebar"] div[data-testid="column"] {
    min-width: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-family: "Outfit", "DM Sans", sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
}
/* Slider / select as frosted “cards” */
[data-testid="stSidebar"] .stSlider {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    padding: 8px 12px 14px !important;
    margin-bottom: 4px !important;
}
[data-testid="stSidebar"] .stSelectbox {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 16px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    padding: 8px 12px 12px !important;
    margin-bottom: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid rgba(185, 28, 28, 0.2) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12) !important;
}
[data-testid="stSidebar"] [data-baseweb="switch"] {
    background: rgba(0, 0, 0, 0.28) !important;
}
[data-testid="stSidebar"] .stCheckbox,
[data-testid="stSidebar"] [data-testid="stWidget"]:has([data-baseweb="switch"]) {
    background: rgba(0, 0, 0, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 6px 10px 10px !important;
    margin-bottom: 4px !important;
}
/* Metrics as floating chips in sidebar */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.1), rgba(0, 0, 0, 0.12)) !important;
    border: 1px solid rgba(255, 255, 255, 0.18) !important;
    border-radius: 14px !important;
    padding: 12px 10px !important;
    text-align: center !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: "Syne", sans-serif !important;
    font-size: 1.6rem !important;
}
/* Sidebar scroll bar */
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 6px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #FEF3C7, #B91C1C);
    border-radius: 3px;
}
@keyframes acity-status-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.15); }
    50% { box-shadow: 0 0 0 8px rgba(255, 255, 255, 0); }
}
.acity-sb-pulse { animation: acity-status-pulse 2.2s ease-in-out infinite; }

.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F5F5F4; }
::-webkit-scrollbar-thumb { background: #DC2626; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #B91C1C; }

/* ── Main chat log: scrolls inside viewport (key matches st.container in main) ── */
div.st-key-acity_chat_log {
  min-height: 260px !important;
  max-height: min(calc(100vh - 200px), 92vh) !important;
  height: min(calc(100vh - 200px), 92vh) !important;
  box-sizing: border-box !important;
  border-bottom: 1px solid var(--rw-border) !important;
  background: var(--rw-cream) !important;
}
div.st-key-acity_chat_log,
div.st-key-acity_chat_log [data-testid="stVerticalBlock"] {
  scrollbar-gutter: stable;
}
div.st-key-acity_chat_log *::-webkit-scrollbar {
  width: 9px;
  height: 9px;
}
div.st-key-acity_chat_log *::-webkit-scrollbar-track {
  background: #F5F5F4;
  border-radius: 4px;
}
div.st-key-acity_chat_log *::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, #DC2626, #B91C1C);
  border-radius: 4px;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.15);
}
div.st-key-acity_chat_log *::-webkit-scrollbar-thumb:hover {
  background: #991B1B;
}

/* ── Text input (chat bar) ── */
.stTextInput > div > div > input {
    background: #FFFFFF !important;
    border: 2px solid var(--rw-border) !important;
    border-radius: 12px !important;
    color: var(--rw-text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--rw-red) !important;
    box-shadow: 0 0 0 3px rgba(185, 28, 28, 0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #A8A29E !important; }

/* ── Primary buttons: white on red, hover darker ── */
.stButton > button {
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    padding: 10px 24px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(185, 28, 28, 0.35) !important;
}

/* Main chat form: Send (st.form_submit_button) matches primary red button */
[data-testid="stMain"] .stFormSubmitButton button,
[data-testid="stMain"] .stFormSubmitButton [data-baseweb="button"],
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] [data-baseweb="button"] {
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
    background-image: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.03em !important;
    padding: 10px 18px !important;
    min-height: 48px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    box-shadow: none !important;
}
[data-testid="stMain"] .stFormSubmitButton button:hover,
[data-testid="stMain"] .stFormSubmitButton [data-baseweb="button"]:hover,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] [data-baseweb="button"]:hover {
    background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%) !important;
    background-image: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(185, 28, 28, 0.35) !important;
}
[data-testid="stMain"] .stFormSubmitButton button:focus-visible,
[data-testid="stMain"] .stFormSubmitButton [data-baseweb="button"]:focus-visible,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button:focus-visible,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] [data-baseweb="button"]:focus-visible {
    outline: 2px solid rgba(185, 28, 28, 0.45) !important;
    outline-offset: 2px !important;
}

/* Sidebar metrics / widgets (Streamlit) */
[data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: rgba(255, 255, 255, 0.7) !important; }

/* ── Expander (main) ── */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border: 1px solid var(--rw-border) !important;
    border-radius: 10px !important;
    color: var(--rw-text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    padding: 10px 16px !important;
}
.streamlit-expanderContent {
    background: #FFF7F7 !important;
    border: 1px solid var(--rw-border) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
}

/* ── Metric cards (main) ── */
[data-testid="stMain"] [data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid var(--rw-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMain"] [data-testid="stMetricLabel"] { color: var(--rw-muted) !important; font-size: 11px !important; }
[data-testid="stMain"] [data-testid="stMetricValue"] { color: var(--rw-red) !important; font-family: 'Syne', sans-serif !important; font-size: 22px !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1px solid var(--rw-border) !important;
    border-radius: 10px !important;
    color: var(--rw-text) !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: var(--rw-red) !important; }
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] { background: #FFFFFF !important; }

/* ── Toggle in sidebar: light track ── */
[data-testid="stSidebar"] [data-baseweb="checkbox"] { border-color: rgba(255,255,255,0.5) !important; }
[data-testid="stSidebar"] [data-baseweb="switch"] { background: rgba(0,0,0,0.2) !important; }

/* ── Radio ── */
.stRadio > div { gap: 8px !important; }
.stRadio > div > label {
    background: #FFFFFF !important;
    border: 1px solid var(--rw-border) !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover { border-color: var(--rw-red) !important; }

/* Sidebar actions: glass/stroke buttons (wins over global .stButton) */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.2), rgba(0, 0, 0, 0.15)) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.15) inset, 0 8px 28px rgba(0, 0, 0, 0.25) !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.32), rgba(185, 28, 28, 0.45)) !important;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.25) inset, 0 10px 32px rgba(0, 0, 0, 0.35) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "feedback_log" not in st.session_state:
    st.session_state.feedback_log = []
if "show_details" not in st.session_state:
    st.session_state.show_details = {}

# ── Helper: render sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / header (distinctive “control deck” chrome)
        st.markdown("""
        <div style="position:relative; padding: 28px 20px 22px 22px;
                    background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 100%);
                    border-bottom: 1px solid rgba(255,255,255,0.12);">
            <div style="position:absolute; top:0; left:22px; right:22px; height:2px;
                        background: linear-gradient(90deg, rgba(254,243,199,0.9) 0%, rgba(255,255,255,0.2) 35%,
                        transparent 100%); border-radius: 2px;"></div>
            <div style="font-family:'Outfit',sans-serif; font-weight:300; font-size:9px; letter-spacing:0.28em;
                        color: rgba(255,255,255,0.55); text-transform:uppercase; margin-bottom:10px;
                        padding-left:2px;">Control deck</div>
            <div style="display:flex; align-items:center; gap:14px;">
                <div style="position:relative; width:52px; height:52px; flex-shrink:0;">
                    <div style="position:absolute; inset:0; border-radius:16px; opacity:0.9;
                        background: conic-gradient(from 200deg, #FEF3C7, #FCA5A5, #B91C1C, #7F1D1D, #FEF3C7);"></div>
                    <div style="position:absolute; inset:3px; border-radius:13px; background:linear-gradient(160deg,#FFFFFF,#F5F5F4);
                        display:flex; align-items:center; justify-content:center; font-size:24px;
                        box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);">🎓</div>
                </div>
                <div>
                    <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:20px; color:#FFFFFF;
                        letter-spacing:-0.03em; line-height:1.1;">ACity RAG</div>
                    <div style="font-family:'DM Mono',monospace; font-size:9.5px; color: rgba(255,255,255,0.75);
                        letter-spacing:0.14em; text-transform:uppercase; margin-top:4px;">Academic City University · Ghana</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Status indicator
        status_color = "#A7F3D0" if st.session_state.pipeline_ready else "#FDE68A"
        status_sub = "Index ready · ask in main chat" if st.session_state.pipeline_ready else "Load corpus & FAISS to begin"
        status_text  = "System online" if st.session_state.pipeline_ready else "Not initialized"
        st.markdown(f"""
        <div style="margin: 0 18px 20px; padding: 0; border-radius: 16px; overflow: hidden;
            background: linear-gradient(135deg, rgba(255,255,255,0.11) 0%, rgba(0,0,0,0.12) 100%);
            border: 1px solid rgba(255,255,255,0.22);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2), inset 0 1px 0 rgba(255,255,255,0.12);">
            <div style="height:2px; background: linear-gradient(90deg, {status_color}, transparent);"></div>
            <div style="display:flex; align-items:center; gap: 12px; padding: 14px 16px 14px 14px;">
                <div class="acity-sb-pulse" style="width:12px; height:12px; border-radius:50%;
                    background: radial-gradient(circle at 30% 30%, #FFFFFF 0%, {status_color} 45%, {status_color} 100%);
                    box-shadow: 0 0 0 1px rgba(0,0,0,0.2);"></div>
                <div style="flex:1;">
                    <div style="font-family:'Outfit',sans-serif; font-weight:600; font-size:12px; color: {status_color};
                        letter-spacing: 0.04em; text-transform: uppercase;">{status_text}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px; color: rgba(255,255,255,0.55);
                        margin-top:2px; letter-spacing: 0.02em;">{status_sub}</div>
                </div>
            </div>
            <div style="padding: 0 16px 12px; font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.1em;
                color: rgba(255,255,255,0.35); text-transform:uppercase;">RAG engine · v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        # Init button
        if not st.session_state.pipeline_ready:
            st.markdown("<div style='padding: 0 18px;'>", unsafe_allow_html=True)
            if st.button("⚡ Initialize pipeline", key="init_btn"):
                with st.spinner("Loading documents & building index…"):
                    try:
                        pipeline = RAGPipeline()
                        pipeline.initialize()
                        st.session_state.pipeline = pipeline
                        st.session_state.pipeline_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Settings
        st.markdown("""
        <div style="padding: 0 18px 6px; display:flex; align-items:center; gap:10px; margin-top:4px;">
            <div style="flex:0 0 18px; height:1px; background: linear-gradient(90deg, rgba(254,243,199,0.5), transparent);"></div>
            <div style="font-family:'Outfit',sans-serif; font-weight:600; font-size:10px; letter-spacing:0.2em;
                color: rgba(255,255,255,0.88); text-transform: uppercase;">Retrieval</div>
            <div style="flex:1; height:1px; background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);"></div>
        </div>
        """, unsafe_allow_html=True)

        top_k = st.slider("Top-K Chunks", 1, 10, 5, key="top_k",
                          help="Number of document chunks to retrieve")
        prompt_style = st.selectbox(
            "Prompt Template",
            ["Hallucination-Controlled", "Chain-of-Thought", "Basic"],
            key="prompt_style"
        )
        show_chunks   = st.toggle("Show retrieved chunks",   value=True,  key="show_chunks")
        show_scores   = st.toggle("Show similarity scores",  value=True,  key="show_scores")
        show_prompt   = st.toggle("Show full prompt",        value=False, key="show_prompt")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Data sources
        st.markdown("""
        <div style="padding: 10px 18px 6px; display:flex; align-items:center; gap:10px;">
            <div style="flex:0 0 18px; height:1px; background: linear-gradient(90deg, rgba(254,243,199,0.5), transparent);"></div>
            <div style="font-family:'Outfit',sans-serif; font-weight:600; font-size:10px; letter-spacing:0.2em;
                color: rgba(255,255,255,0.88); text-transform: uppercase;">Corpus</div>
            <div style="flex:1; height:1px; background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);"></div>
        </div>
        """, unsafe_allow_html=True)

        sources = [
            ("📊", "Ghana Elections CSV", "Election results dataset"),
            ("📄", "2025 Budget PDF",     "Budget Statement & Economic Policy"),
        ]
        for icon, name, desc in sources:
            st.markdown(f"""
            <div style="margin: 0 18px 10px; padding:0; position:relative;">
                <div style="position:absolute; left:0; top:8px; bottom:8px; width:3px; border-radius:2px;
                    background: linear-gradient(180deg, #FEF3C7, rgba(255,255,255,0.2)); box-shadow: 0 0 12px rgba(254,243,199,0.35);"></div>
                <div style="margin-left:8px; padding:12px 14px 12px 16px; border-radius: 0 14px 14px 0;
                    background: linear-gradient(105deg, rgba(255,255,255,0.09) 0%, rgba(0,0,0,0.1) 100%);
                    border: 1px solid rgba(255,255,255,0.12); border-left: none;
                    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 4px 16px rgba(0,0,0,0.12);">
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="font-size:18px; line-height:1; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">{icon}</span>
                    <div>
                        <div style="font-family:'Outfit',sans-serif; font-weight:600; font-size:12.5px; color:#FFFFFF;
                            letter-spacing:0.01em;">{name}</div>
                        <div style="font-family:'DM Mono',monospace; font-size:9.5px; color: rgba(255,255,255,0.55);
                            margin-top:2px; letter-spacing:0.02em;">{desc}</div>
                    </div>
                </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Stats
        st.markdown("""
        <div style="padding: 10px 18px 6px; display:flex; align-items:center; gap:10px;">
            <div style="flex:0 0 18px; height:1px; background: linear-gradient(90deg, rgba(254,243,199,0.5), transparent);"></div>
            <div style="font-family:'Outfit',sans-serif; font-weight:600; font-size:10px; letter-spacing:0.2em;
                color: rgba(255,255,255,0.88); text-transform: uppercase;">Session</div>
            <div style="flex:1; height:1px; background: linear-gradient(90deg, rgba(255,255,255,0.12), transparent);"></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            fb = st.session_state.feedback_log
            pos = sum(1 for f in fb if f.get("rating") == "👍") if fb else 0
            st.metric("👍 Helpful", pos)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Clear chat
        st.markdown("<div style='padding: 0 18px;'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", key="clear_btn"):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # Footer (in document flow so it never stacks under the Clear button)
        st.markdown("""
        <div style="margin-top: 20px; padding: 16px 20px 32px; text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.12);
            clear: both;">
            <div style="font-family:'DM Mono',monospace; font-size:9px; letter-spacing:0.12em; text-transform:uppercase;
                        color: rgba(255,255,255,0.35); line-height:1.6;">
                Academic City University — RAG capstone
                <span style="display:block; margin-top:4px; color: rgba(254,243,199,0.4);">Vanilla stack · FAISS · Groq</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Helper: render chat message ─────────────────────────────────────────────────
def render_message(msg, idx):
    role    = msg["role"]
    content = msg["content"]
    ts      = msg.get("timestamp", "")

    if role == "user":
        safe = html.escape(str(content))
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; margin:16px 0 8px;">
            <div style="max-width:70%; background:linear-gradient(135deg,#DC2626,#B91C1C);
                        border-radius:18px 18px 4px 18px; padding:14px 18px;
                        box-shadow: 0 4px 16px rgba(185,28,28,0.28);">
                <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                            color:#fff; line-height:1.6;">{safe}</div>
                <div style="font-family:'DM Mono',monospace; font-size:10px;
                            color:rgba(255,255,255,0.4); margin-top:6px;
                            text-align:right;">{ts}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:  # assistant
        answer = content.get("answer", "")
        chunks = content.get("chunks", [])
        scores = content.get("scores", [])
        prompt = content.get("prompt", "")
        source_tags = content.get("sources", [])
        latency = content.get("latency", "")

        # Source badges
        badge_html = ""
        seen = set()
        for s in source_tags:
            if s not in seen:
                seen.add(s)
                color = "#B91C1C" if "Budget" in s else "#0D9488"
                badge_html += f"""<span style="background:{color}22; border:1px solid {color}55;
                    color:{color}; font-family:'DM Mono',monospace; font-size:10px;
                    border-radius:6px; padding:2px 8px; margin-right:6px;">{s}</span>"""

        st.markdown(f"""
        <div style="margin:8px 0 4px;">
            <div style="display:flex; align-items:flex-start; gap:12px;">
                <div style="width:36px; height:36px; background:linear-gradient(135deg,#FFFFFF,#F5F5F4);
                            border:1px solid rgba(185,28,28,0.25); border-radius:10px;
                            display:flex; align-items:center; justify-content:center;
                            font-size:18px; flex-shrink:0;">🎓</div>
                <div style="flex:1; background:#FFFFFF;
                            border:1px solid rgba(185,28,28,0.2); border-radius:4px 18px 18px 18px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.04); padding:16px 20px;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                        <span style="font-family:'Syne',sans-serif; font-weight:700; font-size:12px;
                                     color:#B91C1C; letter-spacing:0.06em;">ACity Assistant</span>
                        {f'<span style="font-family:DM Mono,monospace; font-size:10px; color:#A8A29E;">⚡ {latency}</span>' if latency else ''}
                    </div>
                    {f'<div style="margin-bottom:10px;">{badge_html}</div>' if badge_html else ''}
                    <div style="font-family:'DM Sans',sans-serif; font-size:14px;
                                color:#1C1917; line-height:1.75; white-space:pre-wrap;">{html.escape(str(answer))}</div>
                    <div style="font-family:'DM Mono',monospace; font-size:10px;
                                color:#A8A29E; margin-top:10px;">{ts}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail panels
        if st.session_state.get("show_chunks") and chunks:
            with st.expander(f"📄 Retrieved Chunks ({len(chunks)})", expanded=False):
                for i, (chunk, score) in enumerate(zip(chunks, scores)):
                    relevance = "🟢 High" if score > 0.75 else ("🟡 Medium" if score > 0.5 else "🔴 Low")
                    bar_w = int(score * 100)
                    bar_color = "#10B981" if score > 0.75 else ("#F59E0B" if score > 0.5 else "#EF4444")
                    st.markdown(f"""
                    <div style="margin-bottom:12px; padding:14px;
                                background:#FFFFFF;
                                border:1px solid rgba(185,28,28,0.12); border-radius:10px;">
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                            <span style="font-family:'DM Mono',monospace; font-size:11px;
                                         color:#B91C1C;">Chunk {i+1}</span>
                            <span style="font-family:'DM Mono',monospace; font-size:11px;
                                         color:#57534E;">{relevance} · {score:.4f}</span>
                        </div>
                        <div style="height:3px; background:#F5F5F4;
                                    border-radius:2px; margin-bottom:10px;">
                            <div style="height:100%; width:{bar_w}%;
                                        background:{bar_color}; border-radius:2px;
                                        transition:width 0.5s ease;"></div>
                        </div>
                        <div style="font-family:'DM Sans',sans-serif; font-size:12px;
                                    color:#44403C; line-height:1.6;">{html.escape((chunk[:400] + ('…' if len(chunk) > 400 else '')))}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if st.session_state.get("show_prompt") and prompt:
            with st.expander("📨 Full Prompt Sent to LLM", expanded=False):
                st.code(prompt, language="markdown")

        # Feedback
        fb_key = f"fb_{idx}"
        if fb_key not in st.session_state:
            st.session_state[fb_key] = None

        col1, col2, col3 = st.columns([1, 1, 10])
        with col1:
            if st.button("👍", key=f"up_{idx}", help="Helpful"):
                st.session_state[fb_key] = "👍"
                st.session_state.feedback_log.append({
                    "idx": idx, "rating": "👍",
                    "answer_preview": answer[:80],
                    "timestamp": datetime.now().isoformat()
                })
                _save_feedback()
        with col2:
            if st.button("👎", key=f"dn_{idx}", help="Not helpful"):
                st.session_state[fb_key] = "👎"
                st.session_state.feedback_log.append({
                    "idx": idx, "rating": "👎",
                    "answer_preview": answer[:80],
                    "timestamp": datetime.now().isoformat()
                })
                _save_feedback()

        if st.session_state[fb_key]:
            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:11px;
                        color:#78716C; margin-top:4px; padding-left:4px;">
                Feedback recorded {st.session_state[fb_key]}
            </div>
            """, unsafe_allow_html=True)


def _save_feedback():
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/feedback.json", "w") as f:
            json.dump(st.session_state.feedback_log, f, indent=2)
    except Exception:
        pass


# ── Helper: render hero / welcome screen ───────────────────────────────────────
def render_welcome():
    # Hero section
    st.markdown("""
    <div style="display:flex; flex-direction:column; align-items:center;
                justify-content:center; padding:60px 40px 32px; text-align:center;">
        <div style="width:80px; height:80px;
                    background:linear-gradient(135deg,#DC2626 0%,#B91C1C 100%);
                    border-radius:20px; display:flex; align-items:center;
                    justify-content:center; font-size:40px; margin-bottom:28px;
                    color:#fff; box-shadow:0 20px 50px rgba(185,28,28,0.28);">🎓</div>
        <h1 style="font-family:'Syne',sans-serif; font-weight:800; font-size:36px;
                   color:#1C1917; letter-spacing:-0.03em; margin:0 0 8px;">
            ACity RAG Assistant
        </h1>
        <p style="font-family:'DM Mono',monospace; font-size:12px;
                  color:#B91C1C; letter-spacing:0.12em; margin:0 0 24px;">
            RETRIEVAL-AUGMENTED GENERATION · ACADEMIC CITY UNIVERSITY GHANA
        </p>
        <p style="font-family:'DM Sans',sans-serif; font-size:15px;
                  color:#57534E; max-width:480px; line-height:1.7; margin:0 0 32px;">
            Ask anything about <strong style="color:#B91C1C;">Ghana's 2025 Budget</strong>
            or <strong style="color:#0D9488;">Ghana Election Results</strong>.
            Every answer is grounded in the source documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion label
    st.markdown("""
    <p style="text-align:center; font-family:'Syne',sans-serif; font-weight:700;
              font-size:11px; color:#A8A29E; letter-spacing:0.12em;
              text-transform:uppercase; margin-bottom:16px;">
        Try asking…
    </p>
    """, unsafe_allow_html=True)

    # Suggestion buttons using native Streamlit columns
    suggestions = [
        ("💰", "What is Ghana's inflation target for 2025?"),
        ("🗳️", "Who won the presidential election in Ashanti Region?"),
        ("📈", "What is the GDP growth projection in the 2025 budget?"),
        ("🏆", "Which party won the most parliamentary seats?"),
        ("💵", "What are the key revenue measures in the 2025 budget?"),
        ("📊", "What was the voter turnout in the Greater Accra region?"),
    ]

    col1, col2, col3 = st.columns(3)
    cols = [col1, col2, col3]
    for i, (icon, q) in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(f"{icon} {q}", key=f"sug_{i}"):
                st.session_state["pending_query"] = q

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)


# ── Main layout ─────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Top bar ──
    st.markdown("""
    <div style="background:#FFFFFF; backdrop-filter:blur(20px);
                border-bottom:1px solid rgba(185,28,28,0.12);
                padding:16px 32px; display:flex; align-items:center;
                justify-content:space-between; position:sticky; top:0; z-index:100;">
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:16px;
                    color:#1C1917; letter-spacing:-0.01em;">
            💬 Chat Interface
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:11px;
                    color:#B91C1C;">
            Ghana Elections · 2025 Budget · RAG-Powered
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Message area: scrollable log once there is history (fixed height → scrollbar) ──
    if not st.session_state.messages:
        render_welcome()
    else:
        with st.container(
            height=600,
            key="acity_chat_log",
            border=False,
            autoscroll=True,
        ):
            st.markdown("<div style='padding:24px 32px;'>", unsafe_allow_html=True)
            for i, msg in enumerate(st.session_state.messages):
                render_message(msg, i)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Input bar ──
    st.markdown("""
    <div style="position:sticky; bottom:0; background:#FFFBFB;
                backdrop-filter:blur(20px); border-top:1px solid rgba(185,28,28,0.12);
                padding:20px 32px;">
    """, unsafe_allow_html=True)

    # Clear the text field on the run *after* a successful submit (must happen
    # before query_input is instantiated — cannot set session_state[key] after).
    if st.session_state.pop("_clear_query_input", False):
        st.session_state["query_input"] = ""

    # Suggestion click sets pending_query; must assign before text_input (value= is only initial).
    if "pending_query" in st.session_state and st.session_state["pending_query"]:
        st.session_state["query_input"] = st.session_state.pop("pending_query")

    if not st.session_state.pipeline_ready:
        st.markdown(
            """
            <div style="margin:0 0 14px 0; padding:12px 16px; border-radius:12px;
                background:linear-gradient(90deg,#FEF2F2,#FFFBFB);
                border:1px solid rgba(185,28,28,0.35);
                box-shadow:0 1px 3px rgba(0,0,0,0.06);
                font-family:'DM Sans',sans-serif; font-size:14px; color:#44403C; line-height:1.5;">
                <span style="font-family:'Syne',sans-serif; font-weight:700; color:#B91C1C;">⚡ Initialize the pipeline first</span>
                — open the <strong>Control deck</strong> on the left and tap
                <strong style="color:#1C1917;">Initialize pipeline</strong>, then send your message.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Form: Enter in the text field submits the form (Send). Suggestions only pre-fill; user can edit then send.
    with st.form("chat_query_form", clear_on_submit=False, border=False):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            query = st.text_input(
                "",
                placeholder="Ask about Ghana Elections or the 2025 Budget…",
                key="query_input",
                label_visibility="collapsed",
            )
        with col_btn:
            send = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    do_submit = bool(send) and bool(query and query.strip())

    # ── Handle query ──
    if do_submit:
        if not st.session_state.pipeline_ready:
            st.warning("⚡ Please initialize the pipeline first using the sidebar button.")
            return

        user_msg = {
            "role": "user",
            "content": query.strip(),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_msg)
        st.session_state.query_count += 1

        with st.spinner("🔍 Retrieving relevant context…"):
            try:
                t0 = time.time()
                result = st.session_state.pipeline.query(
                    query=query.strip(),
                    top_k=st.session_state.get("top_k", 5),
                    prompt_style=st.session_state.get("prompt_style", "Hallucination-Controlled")
                )
                latency = f"{time.time() - t0:.2f}s"

                assistant_msg = {
                    "role": "assistant",
                    "content": {
                        "answer":  result["answer"],
                        "chunks":  result["chunks"],
                        "scores":  result["scores"],
                        "prompt":  result["prompt"],
                        "sources": result["sources"],
                        "latency": latency,
                        "logs":    result.get("logs", [])
                    },
                    "timestamp": datetime.now().strftime("%H:%M")
                }
                st.session_state.messages.append(assistant_msg)

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "answer": f"⚠️ Error processing your query: {str(e)}",
                        "chunks": [], "scores": [], "prompt": "", "sources": []
                    },
                    "timestamp": datetime.now().strftime("%H:%M")
                })

        st.session_state["_clear_query_input"] = True
        st.rerun()


if __name__ == "__main__":
    main()
