"""
app.py
──────
EcoPlanner – Sustainable Building Design Dashboard
Author : Senior AI Automation Engineer

Run with:
    streamlit run app.py

Design language: Dark-mode "living blueprint" — deep forest greens and
charcoal backgrounds evoking technical drawings with organic warmth.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Internal modules ───────────────────────────────────────────────────────────
from core.energy_sim import EnergyAnalyzer
from core.materials_rag import get_ai_recommendations, load_materials_db

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EcoPlanner · Sustainable Building AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS  — "Living Blueprint" dark theme
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
/* ── Google Fonts — Modern Theme ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&display=swap');

/* ── Root variables ── */
:root {
    --eco-bg:          #0d1410;
    --eco-surface:     #141c17;
    --eco-surface-2:   #1b2820;
    --eco-border:      #2a3d31;
    --eco-green-dim:   #2d6a4f;
    --eco-green:       #52b788;
    --eco-green-light: #95d5b2;
    --eco-amber:       #e9c46a;
    --eco-red:         #e76f51;
    --eco-blue:        #48cae4;
    --eco-text:        #d8f3dc;
    --eco-text-muted:  #74c69d;
    --eco-text-dim:    #52796f;
    --font-display:    'Playfair Display', Georgia, serif;
    --font-body:       'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono:       'JetBrains Mono', 'Courier New', monospace;
}

/* ── Modern Animations ── */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes glowPulse {
    0%, 100% {
        box-shadow: 0 0 10px rgba(82, 183, 136, 0.3);
    }
    50% {
        box-shadow: 0 0 20px rgba(82, 183, 136, 0.6);
    }
}

@keyframes shimmer {
    0% {
        background-position: -1000px 0;
    }
    100% {
        background-position: 1000px 0;
    }
}

/* ── Global overrides ── */
html, body, [class*="css"] {
    background-color: var(--eco-bg) !important;
    color: var(--eco-text) !important;
    font-family: var(--font-body) !important;
}

/* Streamlit main container */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--eco-surface) !important;
    border-right: 1px solid var(--eco-border) !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

/* ── Header / Title area ── */
.eco-header {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
    padding: 0 0 1.5rem 0;
    border-bottom: 1px solid var(--eco-border);
    margin-bottom: 2rem;
    animation: fadeInUp 0.8s ease-out;
}
.eco-header-icon {
    font-size: 2.8rem;
    line-height: 1;
    animation: fadeInUp 0.8s ease-out 0.1s both;
}
.eco-header-text h1 {
    font-family: var(--font-display) !important;
    font-size: 2.4rem !important;
    font-weight: 600 !important;
    color: var(--eco-green-light) !important;
    margin: 0 0 0.1rem 0 !important;
    letter-spacing: -0.02em;
    animation: slideInRight 0.8s ease-out 0.2s both;
}
.eco-header-text p {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--eco-text-dim) !important;
    margin: 0 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    animation: fadeInUp 0.8s ease-out 0.3s both;
}

/* ── Section headers ── */
.eco-section-title {
    font-family: var(--font-display) !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    font-style: italic;
    color: var(--eco-green) !important;
    letter-spacing: 0.01em;
    margin: 0 0 1rem 0 !important;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--eco-border);
    animation: fadeInUp 0.6s ease-out;
}
.eco-subsection {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    color: var(--eco-text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.75rem !important;
    animation: fadeInUp 0.6s ease-out;
}

/* ── Metric cards ── */
.eco-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.eco-metric-card {
    background: var(--eco-surface-2);
    border: 1px solid var(--eco-border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out;
}
.eco-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--eco-green-dim), var(--eco-green));
}
.eco-metric-card:hover {
    border-color: var(--eco-green-dim);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(82, 183, 136, 0.15);
}
.eco-metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--eco-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.eco-metric-value {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--eco-green-light);
    line-height: 1;
}
.eco-metric-unit {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--eco-text-muted);
    margin-top: 0.25rem;
}
.eco-metric-delta-pos {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--eco-green);
}
.eco-metric-delta-neg {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--eco-red);
}

/* ── Energy breakdown bar ── */
.eco-bar-container {
    background: var(--eco-surface-2);
    border: 1px solid var(--eco-border);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}
.eco-bar-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.eco-bar-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--eco-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    min-width: 80px;
}
.eco-bar-track {
    flex: 1;
    background: var(--eco-border);
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.eco-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s ease;
}
.eco-bar-val {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--eco-text-muted);
    min-width: 100px;
    text-align: right;
}

/* ── Material recommendation cards ── */
.eco-mat-card {
    background: var(--eco-surface-2);
    border: 1px solid var(--eco-border);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    transition: all 0.3s ease;
    animation: fadeInUp 0.6s ease-out;
}
.eco-mat-card:hover {
    border-color: var(--eco-green-dim);
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(82, 183, 136, 0.2);
}
.eco-mat-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.6rem;
}
.eco-mat-name {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--eco-green-light);
}
.eco-mat-category {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--eco-text-dim);
    background: var(--eco-border);
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.eco-mat-stats {
    display: flex;
    gap: 1.5rem;
    margin: 0.6rem 0;
}
.eco-mat-stat {
    display: flex;
    flex-direction: column;
}
.eco-mat-stat-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--eco-text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.eco-mat-stat-value {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    color: var(--eco-text);
}
.eco-mat-offset {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(82, 183, 136, 0.12);
    border: 1px solid rgba(82, 183, 136, 0.3);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--eco-green);
    margin-top: 0.4rem;
}
.eco-mat-rationale {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--eco-text-muted);
    line-height: 1.5;
    margin-top: 0.7rem;
    padding-top: 0.7rem;
    border-top: 1px solid var(--eco-border);
}

/* ── Strategy box ── */
.eco-strategy-box {
    background: linear-gradient(135deg, rgba(44, 106, 79, 0.15), rgba(13, 20, 16, 0.8));
    border: 1px solid var(--eco-green-dim);
    border-radius: 10px;
    padding: 1.4rem;
    margin-top: 1.5rem;
    animation: fadeInUp 0.8s ease-out;
    transition: all 0.3s ease;
}
.eco-strategy-box:hover {
    border-color: var(--eco-green);
    box-shadow: 0 8px 24px rgba(82, 183, 136, 0.1);
}
.eco-strategy-title {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--eco-green);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.6rem;
}
.eco-strategy-text {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--eco-text);
    line-height: 1.6;
}

/* ── Reduction badge ── */
.eco-reduction-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(82, 183, 136, 0.15);
    border: 1px solid var(--eco-green);
    border-radius: 6px;
    padding: 0.5rem 1rem;
    font-family: var(--font-display);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--eco-green-light);
    margin-top: 0.8rem;
}

/* ── Sidebar inputs ── */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--eco-text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] h3 {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 300 !important;
    font-style: italic !important;
    color: var(--eco-green) !important;
}

/* Streamlit metric widget overrides */
[data-testid="metric-container"] {
    background: var(--eco-surface-2) !important;
    border: 1px solid var(--eco-border) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    color: var(--eco-text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.6rem !important;
    color: var(--eco-green-light) !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
}

/* Buttons */
.stButton > button {
    background: var(--eco-surface-2) !important;
    border: 1px solid var(--eco-green-dim) !important;
    color: var(--eco-green-light) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: rgba(82, 183, 136, 0.12) !important;
    border-color: var(--eco-green) !important;
    color: var(--eco-green-light) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 16px rgba(82, 183, 136, 0.15) !important;
}
.stDownloadButton > button {
    background: rgba(82, 183, 136, 0.12) !important;
    border: 1px solid var(--eco-green) !important;
    color: var(--eco-green-light) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    font-weight: 500 !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 16px rgba(82, 183, 136, 0.2) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--eco-text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Divider */
hr {
    border-color: var(--eco-border) !important;
}

/* Info / warning boxes */
.stAlert {
    background: var(--eco-surface-2) !important;
    border-color: var(--eco-border) !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--eco-bg); }
::-webkit-scrollbar-thumb { background: var(--eco-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--eco-green-dim); }
</style>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    defaults = {
        "energy_result": None,
        "ai_result": None,
        "last_inputs": {},
        "api_key_validated": False,
        "app_loaded": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP ANIMATION
# ═══════════════════════════════════════════════════════════════════════════════

if not st.session_state["app_loaded"]:
    st.markdown(
        """
        <style>
        @keyframes startupGlow {
            0% {
                opacity: 0;
                transform: scale(0.95);
            }
            50% {
                opacity: 1;
                transform: scale(1.02);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        .startup-banner {
            text-align: center;
            padding: 2rem 1rem;
            background: linear-gradient(135deg, rgba(82, 183, 136, 0.08), rgba(72, 202, 228, 0.08));
            border: 1px solid rgba(82, 183, 136, 0.2);
            border-radius: 10px;
            margin-bottom: 1.5rem;
            animation: startupGlow 0.8s ease-out;
        }
        .startup-banner h2 {
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 600;
            background: linear-gradient(90deg, #52b788, #95d5b2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.5rem 0;
        }
        .startup-banner p {
            font-family: 'Inter', sans-serif;
            color: #74c69d;
            font-size: 0.95rem;
            margin: 0;
        }
        </style>
        <div class="startup-banner">
            <h2>✨ Welcome to EcoPlanner</h2>
            <p>Sustainable Building Design · AI-Powered Analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["app_loaded"] = True


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _eui_benchmark(eui: float) -> tuple[str, str]:
    """Return (label, css_colour) for the EUI benchmark."""
    if eui < 50:
        return "Excellent — Passive-ready", "#52b788"
    elif eui < 100:
        return "Good — Above regulation", "#95d5b2"
    elif eui < 150:
        return "Average — Code compliant", "#e9c46a"
    elif eui < 200:
        return "Below average", "#f4a261"
    else:
        return "Poor — Needs improvement", "#e76f51"


def _build_report(inputs: dict, energy: dict, ai: dict) -> bytes:
    """Serialise all results to a UTF-8 JSON report."""
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "version": "1.0.0",
        "application": "EcoPlanner",
        "building_inputs": inputs,
        "energy_analysis": energy,
        "material_recommendations": ai,
    }
    return json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")


def _pct_bar_html(label: str, value: float, total: float, colour: str) -> str:
    pct = min(100, round(value / total * 100, 1)) if total > 0 else 0
    return f"""
<div class="eco-bar-row">
  <span class="eco-bar-label">{label}</span>
  <div class="eco-bar-track">
    <div class="eco-bar-fill" style="width:{pct}%;background:{colour};"></div>
  </div>
  <span class="eco-bar-val">{value:,.0f} kWh &nbsp;({pct}%)</span>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Branding ──────────────────────────────────────────────────────────────
    st.markdown(
        """
<div style="text-align:center;padding:0.5rem 0 1.5rem 0;">
  <div style="font-size:2.2rem;margin-bottom:0.2rem;">🌿</div>
  <div style="font-family:'Fraunces',serif;font-size:1.4rem;color:#95d5b2;
              font-weight:600;letter-spacing:-0.01em;">EcoPlanner</div>
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#52796f;
              text-transform:uppercase;letter-spacing:0.1em;">
    Sustainable Building AI · v1.0
  </div>
</div>
<hr style="border-color:#2a3d31;margin:0 0 1.2rem 0;">
""",
        unsafe_allow_html=True,
    )

    # ── Eco Quotes ────────────────────────────────────────────────────────────
    import random
    eco_quotes = [
        "\"The greatest threat to our planet is the belief that someone else will save it.\" - Robert Swan",
        "\"Design with nature, not against it.\" - Frank Lloyd Wright",
        "\"Sustainable building is not a luxury—it's a necessity.\" - Green Builder",
        "\"Every building is a vote for the kind of world we want to live in.\" - Unknown",
        "\"Energy efficiency: the first fuel of sustainability.\" - IEA",
        "\"Smart design today, living well tomorrow.\" - EcoPlanner",
    ]
    
    st.markdown(
        f"""
        <style>
        .eco-quote-box {{
            background: linear-gradient(135deg, rgba(82, 183, 136, 0.12), rgba(72, 202, 228, 0.08));
            border-left: 3px solid #52b788;
            border-radius: 6px;
            padding: 1rem;
            margin: 1.2rem 0;
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #95d5b2;
            font-style: italic;
            line-height: 1.6;
            animation: fadeInUp 0.8s ease-out 0.2s both;
        }}
        .eco-quote-box::before {{
    
            display: block;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        </style>
        <div class="eco-quote-box">
            {random.choice(eco_quotes)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── API Config ────────────────────────────────────────────────────────────
    # Get API key from environment
    api_key = os.getenv("OXLO_API_KEY", "")
    oxlo_model = "deepseek-r1-8b"

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Building Parameters ────────────────────────────────────────────────────
    st.markdown("### 🏗 Building Parameters")

    floor_area = st.number_input(
        "Floor Area (m²)",
        min_value=20.0,
        max_value=50_000.0,
        value=500.0,
        step=10.0,
        help="Total conditioned floor area",
    )
    wwr = st.slider(
        "Window-to-Wall Ratio",
        min_value=0.05,
        max_value=0.80,
        value=0.30,
        step=0.05,
        format="%.2f",
        help="Proportion of external wall that is glazed",
    )
    avg_temp = st.slider(
        "Avg. Regional Temperature (°C)",
        min_value=-20.0,
        max_value=40.0,
        value=12.0,
        step=0.5,
        help="Annual average outdoor dry-bulb temperature for your climate zone",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Advanced parameters (collapsible) ─────────────────────────────────────
    with st.expander("⚙ Advanced Parameters"):
        u_wall = st.number_input(
            "Wall U-value (W/m²·K)",
            min_value=0.05, max_value=2.0, value=0.30, step=0.05,
        )
        u_roof = st.number_input(
            "Roof U-value (W/m²·K)",
            min_value=0.05, max_value=2.0, value=0.20, step=0.05,
        )
        u_glazing = st.number_input(
            "Glazing U-value (W/m²·K)",
            min_value=0.5, max_value=6.0, value=2.0, step=0.1,
        )
        hvac_cop = st.number_input(
            "HVAC CoP",
            min_value=1.0, max_value=8.0, value=3.0, step=0.1,
            help="Combined COP for heating & cooling plant",
        )
        num_storeys = st.number_input(
            "Number of storeys",
            min_value=1, max_value=50, value=3, step=1,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Run button ─────────────────────────────────────────────────────────────
    run_analysis = st.button(
        "▶  Run EcoPlanner Analysis",
        use_container_width=True,
        type="primary",
    )

    st.markdown(
        "<div style='font-family:DM Mono,monospace;font-size:0.6rem;"
        "color:#52796f;text-align:center;padding-top:0.8rem;'>"
        "Powered by Oxlo.ai · ISO 13790 Energy Model</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<div class="eco-header">
  <div class="eco-header-icon">🌿</div>
  <div class="eco-header-text">
    <h1>EcoPlanner</h1>
    <p>Sustainable Building Design · AI-Powered Energy & Materials Analysis</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if run_analysis:
    current_inputs = {
        "floor_area": floor_area,
        "wwr": wwr,
        "avg_temp": avg_temp,
        "u_wall": u_wall,
        "u_roof": u_roof,
        "u_glazing": u_glazing,
        "hvac_cop": hvac_cop,
        "num_storeys": num_storeys,
    }

    # ── 1. Energy simulation ───────────────────────────────────────────────────
    with st.spinner("🔋 Running energy simulation…"):
        try:
            analyzer = EnergyAnalyzer()
            result = analyzer.analyze(
                floor_area=floor_area,
                window_to_wall_ratio=wwr,
                avg_outdoor_temp_c=avg_temp,
                u_wall=u_wall,
                u_roof=u_roof,
                u_glazing=u_glazing,
                hvac_cop=hvac_cop,
                num_storeys=int(num_storeys),
            )
            st.session_state["energy_result"] = result
            st.session_state["last_inputs"] = current_inputs
        except ValueError as e:
            st.error(f"⚠️ Input validation error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"⚠️ Energy simulation failed: {e}")
            st.stop()

    # ── 2. AI material recommendations ────────────────────────────────────────
    if not api_key:
        st.warning(
            "🔑 No Oxlo.ai API key provided. Energy analysis is complete below. "
            "Add your API key in the sidebar to unlock AI material recommendations."
        )
        st.session_state["ai_result"] = None
    else:
        with st.spinner("🤖 Consulting AI for eco-material recommendations…"):
            try:
                ai_result = get_ai_recommendations(
                    energy_summary=result.summary,
                    api_key=api_key,
                    model=oxlo_model,
                )
                st.session_state["ai_result"] = ai_result
            except Exception as e:
                st.error(f"⚠️ AI recommendation failed: {e}")
                st.session_state["ai_result"] = {"error": str(e), "recommendations": []}

    st.success("✅ Analysis complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

energy_result = st.session_state.get("energy_result")
ai_result = st.session_state.get("ai_result")

if energy_result is None:
    # ── Welcome / empty state ─────────────────────────────────────────────────
    st.markdown(
        """
<div style="text-align:center;padding:4rem 2rem;opacity:0.6;">
  <div style="font-size:3rem;margin-bottom:1rem;">🏗️</div>
  <div style="font-family:'Fraunces',serif;font-size:1.4rem;color:#74c69d;
              font-weight:300;font-style:italic;margin-bottom:0.5rem;">
    Configure your building in the sidebar
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#52796f;
              max-width:500px;margin:0 auto;line-height:1.8;">
    Set floor area, window-to-wall ratio, and regional temperature, then click 
    <strong style="color:#74c69d;">Run EcoPlanner Analysis</strong> to generate 
    energy metrics and AI-powered material recommendations.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.stop()


# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")


# ═══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — ENERGY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

with col_left:
    st.markdown(
        '<p class="eco-section-title">⚡ Energy Performance Metrics</p>',
        unsafe_allow_html=True,
    )

    eui = energy_result.eui_kwh_per_m2
    eui_label, eui_colour = _eui_benchmark(eui)

    # ── Primary KPI metrics ────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Total Energy",
            value=f"{energy_result.total_kwh / 1000:.1f} MWh",
            delta=f"EUI: {eui} kWh/m²",
            delta_color="off",
        )
    with m2:
        st.metric(
            label="Operational CO₂",
            value=f"{energy_result.operational_co2_tonnes:.1f} t",
            delta="per year",
            delta_color="off",
        )
    with m3:
        st.metric(
            label="EUI Benchmark",
            value=eui_label.split("—")[0].strip(),
            delta=eui_label.split("—")[-1].strip() if "—" in eui_label else "",
            delta_color="off",
        )

    # ── Energy breakdown bar chart ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="eco-subsection">Annual Energy Breakdown</p>',
        unsafe_allow_html=True,
    )

    total_kwh = energy_result.total_kwh or 1
    bars_html = (
        '<div class="eco-bar-container">'
        + _pct_bar_html("Heating", energy_result.heating_kwh, total_kwh, "#48cae4")
        + _pct_bar_html("Cooling", energy_result.cooling_kwh, total_kwh, "#e9c46a")
        + _pct_bar_html("Lighting", energy_result.lighting_kwh, total_kwh, "#52b788")
        + "</div>"
    )
    st.markdown(bars_html, unsafe_allow_html=True)

    # ── Detailed metrics grid ──────────────────────────────────────────────────
    st.markdown(
        '<p class="eco-subsection">Climate & Thermal Indicators</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Heating Load", f"{energy_result.heating_kwh:,.0f}", "kWh/yr")
    with c2:
        st.metric("Cooling Load", f"{energy_result.cooling_kwh:,.0f}", "kWh/yr")
    with c3:
        st.metric("HDD", f"{energy_result.hdd:,.0f}", "°C·days")
    with c4:
        st.metric("CDD", f"{energy_result.cdd:,.0f}", "°C·days")

    # ── CO₂ intensity card ─────────────────────────────────────────────────────
    co2_per_m2 = (
        energy_result.operational_co2_kg / energy_result.summary.get("floor_area_m2", 1)
    )
    st.markdown(
        f"""
<div class="eco-bar-container" style="margin-top:0.8rem;">
  <div class="eco-metric-label">Carbon Intensity</div>
  <div style="display:flex;align-items:baseline;gap:0.5rem;margin:0.4rem 0;">
    <span style="font-family:'Fraunces',serif;font-size:2rem;
                 color:#95d5b2;font-weight:600;">{co2_per_m2:.1f}</span>
    <span style="font-family:'DM Mono',monospace;font-size:0.75rem;
                 color:#74c69d;">kg CO₂/m²/yr</span>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#52796f;">
    Grid intensity: 0.233 kg CO₂/kWh (IEA 2023 global average)
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Materials DB preview ───────────────────────────────────────────────────
    with st.expander("📋 View Materials Database"):
        try:
            df = load_materials_db()
            display_df = df[["Material_Name", "Category", "Embodied_Carbon",
                             "Thermal_Conductivity", "Durability_Years"]].copy()
            display_df.columns = ["Material", "Category", "Emb. Carbon\n(kg CO₂/m²)",
                                   "λ (W/m·K)", "Durability (yr)"]
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
        except Exception as e:
            st.error(f"Could not load materials database: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — AI RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with col_right:
    st.markdown(
        '<p class="eco-section-title">🤖 AI Material Recommendations</p>',
        unsafe_allow_html=True,
    )

    if ai_result is None:
        st.markdown(
            """
<div style="background:var(--eco-surface-2);border:1px solid var(--eco-border);
            border-radius:8px;padding:1.5rem;text-align:center;opacity:0.7;">
  <div style="font-size:1.8rem;margin-bottom:0.6rem;">🔑</div>
  <div style="font-family:'Fraunces',serif;font-size:1rem;color:#74c69d;
              font-style:italic;">
    Add your Oxlo.ai API key to unlock AI recommendations
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#52796f;
              margin-top:0.4rem;">
    Energy analysis above is fully functional without an API key
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    elif ai_result.get("error"):
        st.error(f"**AI Error:** {ai_result['error']}")

    else:
        # ── Overall CO₂ reduction ──────────────────────────────────────────────
        reduction_pct = ai_result.get("estimated_lifecycle_co2_reduction_pct", 0)
        if reduction_pct:
            st.markdown(
                f"""
<div style="margin-bottom:1.2rem;">
  <div class="eco-metric-label">Estimated Lifecycle CO₂ Reduction</div>
  <div class="eco-reduction-badge">
    ↓ {reduction_pct:.0f}% &nbsp;
    <span style="font-family:'DM Mono',monospace;font-size:0.72rem;
                 font-weight:300;color:#74c69d;">vs. conventional build</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        # ── Individual material cards ──────────────────────────────────────────
        st.markdown(
            '<p class="eco-subsection">Recommended Materials</p>',
            unsafe_allow_html=True,
        )

        recommendations = ai_result.get("recommendations", [])
        if not recommendations:
            st.warning("The AI returned no recommendations. Try a different model or check your API key.")
        else:
            for rec in recommendations:
                name = rec.get("material_name", "Unknown")
                category = rec.get("category", "—")
                emb_carbon = rec.get("embodied_carbon", 0)
                thermal_k = rec.get("thermal_conductivity", 0)
                durability = rec.get("durability_years", 0)
                offset = rec.get("carbon_offset_kg_co2e_m2", 0)
                baseline = rec.get("baseline_material", "standard")
                rationale = rec.get("why_recommended", "")
                description = rec.get("description", "")

                offset_str = (
                    f"↓ {offset:.0f} kg CO₂e/m² saved vs {baseline}"
                    if offset > 0
                    else f"Reference material — {abs(offset):.0f} kg CO₂e/m² above baseline"
                )

                st.markdown(
                    f"""
<div class="eco-mat-card">
  <div class="eco-mat-card-header">
    <span class="eco-mat-name">{name}</span>
    <span class="eco-mat-category">{category}</span>
  </div>
  <div class="eco-mat-stats">
    <div class="eco-mat-stat">
      <span class="eco-mat-stat-label">Emb. Carbon</span>
      <span class="eco-mat-stat-value">{emb_carbon} kg CO₂e/m²</span>
    </div>
    <div class="eco-mat-stat">
      <span class="eco-mat-stat-label">λ Value</span>
      <span class="eco-mat-stat-value">{thermal_k} W/m·K</span>
    </div>
    <div class="eco-mat-stat">
      <span class="eco-mat-stat-label">Durability</span>
      <span class="eco-mat-stat-value">{durability} yrs</span>
    </div>
  </div>
  <div class="eco-mat-offset">🌿 {offset_str}</div>
  {f'<div class="eco-mat-rationale">{rationale}</div>' if rationale else ''}
  {f'<div class="eco-mat-rationale" style="opacity:0.6;font-size:0.76rem;">{description}</div>' if description and not rationale else ''}
</div>
""",
                    unsafe_allow_html=True,
                )

        # ── Overall strategy ───────────────────────────────────────────────────
        strategy = ai_result.get("overall_strategy", "")
        if strategy:
            st.markdown(
                f"""
<div class="eco-strategy-box">
  <div class="eco-strategy-title">🌍 Holistic Sustainability Strategy</div>
  <div class="eco-strategy-text">{strategy}</div>
</div>
""",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD REPORT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p class="eco-section-title">📥 Export Report</p>',
    unsafe_allow_html=True,
)

report_bytes = _build_report(
    inputs=st.session_state.get("last_inputs", {}),
    energy=energy_result.summary if energy_result else {},
    ai=ai_result if ai_result else {},
)
st.download_button(
    label="⬇  Download JSON Report",
    data=report_bytes,
    file_name=f"ecoplanner_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json",
    mime="application/json",
    use_container_width=True,
)

st.markdown(
    """
<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#52796f;
            line-height:1.7;padding:0.3rem 0;">
  <strong style="color:#74c69d;">JSON Report</strong> includes building inputs, 
  full energy breakdown, AI material selections, and carbon offset calculations.
</div>
""",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:2rem 0 0.5rem 0;">
  <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#2a3d31;
              letter-spacing:0.1em;text-transform:uppercase;">
    EcoPlanner · ISO 13790 Energy Model · Oxlo.ai RAG Materials · 
    Built with Streamlit · v1.0.0
  </div>
</div>
""",
    unsafe_allow_html=True,
)
