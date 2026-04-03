"""
core/materials_rag.py
──────────────────────
EcoPlanner – Materials RAG (Retrieval-Augmented Generation) Module

Responsibilities
----------------
1. Load and validate the local materials CSV database.
2. Filter candidates relevant to the building's energy profile.
3. Call Oxlo.ai (OpenAI-compatible) with a structured prompt that FORCES
   the model to select ONLY from the provided CSV materials — no hallucination.
4. Parse and return structured recommendations including carbon offset maths.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

load_dotenv()
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).parent
_DATA_DIR = _MODULE_DIR.parent / "data"
MATERIALS_CSV_PATH = _DATA_DIR / "materials.csv"

# ── Oxlo.ai defaults ──────────────────────────────────────────────────────────
OXLO_BASE_URL = os.getenv("OXLO_BASE_URL", "https://api.oxlo.ai/v1")
OXLO_API_KEY = os.getenv("OXLO_API_KEY", "")
OXLO_MODEL = os.getenv("OXLO_MODEL", "llama-3.3-70b")

# Baseline reference materials (must match CSV Material_Name exactly)
BASELINE_MATERIALS = {
    "structural": "Standard Concrete (baseline)",
    "insulation": "Standard Glass Wool",
    "masonry": "Ordinary Fired Brick",
    "timber": "Conventional Timber Frame",
}


# ── CSV Loader ─────────────────────────────────────────────────────────────────

def load_materials_db(csv_path: Path = MATERIALS_CSV_PATH) -> pd.DataFrame:
    """
    Load and validate the materials CSV database.

    Returns
    -------
    pd.DataFrame with columns validated and numeric types enforced.

    Raises
    ------
    FileNotFoundError  : CSV not found at given path.
    ValueError         : Required columns missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Materials database not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {
        "Material_Name", "Category", "Embodied_Carbon",
        "Thermal_Conductivity", "Durability_Years", "Description",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Materials CSV missing columns: {missing}")

    # Coerce numeric columns
    for col in ("Embodied_Carbon", "Thermal_Conductivity", "Durability_Years"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Material_Name", "Embodied_Carbon"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Loaded %d materials from %s", len(df), csv_path)
    return df


def get_eco_candidates(
    df: pd.DataFrame,
    exclude_baselines: bool = True,
    top_n: int = 16,
) -> pd.DataFrame:
    """
    Return the most promising eco materials (lowest embodied carbon first),
    optionally excluding the baseline reference rows.
    """
    baselines = set(BASELINE_MATERIALS.values())
    if exclude_baselines:
        df = df[~df["Material_Name"].isin(baselines)]

    return df.nsmallest(top_n, "Embodied_Carbon").reset_index(drop=True)


# ── Prompt Builder ─────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return (
        "You are EcoPlanner AI — a specialist in sustainable building design and "
        "low-carbon material selection. Your task is to recommend eco-friendly "
        "building materials from an APPROVED LIST ONLY.\n\n"
        "STRICT RULES:\n"
        "  1. You MUST select materials ONLY from the 'APPROVED MATERIALS LIST' "
        "     provided in the user message.  Do NOT suggest any material not in "
        "     that list.\n"
        "  2. Recommend exactly 4 materials covering different building elements "
        "     (structure/frame, insulation, wall/cladding, floor/finish).\n"
        "  3. For EACH recommendation calculate the Carbon Offset as:\n"
        "       Carbon Offset (kg CO₂e/m²) = Baseline_Embodied_Carbon "
        "       − Recommended_Embodied_Carbon\n"
        "     The baseline values will be supplied in the prompt.\n"
        "  4. Respond ONLY with valid JSON — no markdown fences, no preamble.\n"
        "  5. Use the exact JSON schema below:\n\n"
        '{\n'
        '  "recommendations": [\n'
        '    {\n'
        '      "material_name": "<exact name from list>",\n'
        '      "category": "<category>",\n'
        '      "embodied_carbon": <number>,\n'
        '      "thermal_conductivity": <number>,\n'
        '      "durability_years": <number>,\n'
        '      "carbon_offset_kg_co2e_m2": <number>,\n'
        '      "baseline_material": "<name of baseline used>",\n'
        '      "why_recommended": "<2–3 sentence rationale tied to building context>"\n'
        '    }\n'
        '  ],\n'
        '  "overall_strategy": "<3–4 sentence holistic sustainability strategy>",\n'
        '  "estimated_lifecycle_co2_reduction_pct": <number 0–100>\n'
        '}\n'
    )


def _build_user_prompt(
    energy_summary: dict,
    candidates_df: pd.DataFrame,
    baselines_df: pd.DataFrame,
) -> str:
    """
    Construct the user-turn prompt injecting:
    - Building energy context
    - Approved materials list (CSV rows)
    - Baseline embodied carbon reference values
    """

    # ── Energy context ────────────────────────────────────────────────────────
    context_lines = [
        "## Building Energy Profile",
        f"- Floor area          : {energy_summary.get('floor_area_m2', 'N/A')} m²",
        f"- Window-to-wall ratio: {energy_summary.get('wwr', 'N/A')}",
        f"- Avg outdoor temp    : {energy_summary.get('avg_temp_c', 'N/A')} °C",
        f"- Annual heating load : {energy_summary.get('heating_kwh_yr', 'N/A')} kWh/yr",
        f"- Annual cooling load : {energy_summary.get('cooling_kwh_yr', 'N/A')} kWh/yr",
        f"- Energy Use Intensity: {energy_summary.get('eui_kwh_m2_yr', 'N/A')} kWh/m²/yr",
        f"- Operational CO₂     : {energy_summary.get('operational_co2_tonnes_yr', 'N/A')} tonnes/yr",
        f"- HDD / CDD           : {energy_summary.get('hdd', 'N/A')} / {energy_summary.get('cdd', 'N/A')}",
        "",
    ]

    # ── Baseline reference ────────────────────────────────────────────────────
    context_lines.append("## Baseline Reference Materials (for Carbon Offset Calculation)")
    for _, row in baselines_df.iterrows():
        context_lines.append(
            f"- {row['Material_Name']}: {row['Embodied_Carbon']} kg CO₂e/m²"
        )
    context_lines.append("")

    # ── Approved list ─────────────────────────────────────────────────────────
    context_lines.append("## APPROVED MATERIALS LIST (select ONLY from this list)")
    context_lines.append(
        "| # | Material_Name | Category | Embodied_Carbon (kg CO₂e/m²) | "
        "Thermal_Conductivity (W/m·K) | Durability_Years | Description |"
    )
    context_lines.append(
        "|---|---------------|----------|------------------------------|"
        "-----------------------------|------------------|-------------|"
    )
    for i, row in candidates_df.iterrows():
        context_lines.append(
            f"| {i+1} | {row['Material_Name']} | {row['Category']} | "
            f"{row['Embodied_Carbon']} | {row['Thermal_Conductivity']} | "
            f"{row['Durability_Years']} | {str(row['Description'])[:80]} |"
        )

    context_lines.append(
        "\n\nSelect 4 materials from ONLY the table above that are most "
        "appropriate for this building's climate and energy profile. "
        "Respond ONLY with the JSON schema specified in your instructions."
    )

    return "\n".join(context_lines)


# ── AI Recommendations ─────────────────────────────────────────────────────────

def get_ai_recommendations(
    energy_summary: dict,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Call Oxlo.ai with a RAG prompt and return structured material recommendations.

    Parameters
    ----------
    energy_summary : dict from EnergyAnalyzer.get_summary_dict()
    api_key        : Override OXLO_API_KEY env var
    base_url       : Override OXLO_BASE_URL env var
    model          : Override OXLO_MODEL env var

    Returns
    -------
    dict with keys:
        'recommendations' : list of material dicts
        'overall_strategy': str
        'estimated_lifecycle_co2_reduction_pct': float
        'error'           : str | None  – populated only on failure
    """
    resolved_key = api_key or OXLO_API_KEY
    resolved_url = base_url or OXLO_BASE_URL
    resolved_model = model or OXLO_MODEL

    if not resolved_key:
        return _error_response(
            "OXLO_API_KEY not set. Add it to your .env file or pass api_key= parameter."
        )

    # ── Load materials data ───────────────────────────────────────────────────
    try:
        df = load_materials_db()
    except (FileNotFoundError, ValueError) as exc:
        return _error_response(f"Materials DB error: {exc}")

    candidates = get_eco_candidates(df, exclude_baselines=True)
    baselines = df[df["Material_Name"].isin(BASELINE_MATERIALS.values())]

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(energy_summary, candidates, baselines)

    # ── Call Oxlo.ai ──────────────────────────────────────────────────────────
    client = OpenAI(api_key=resolved_key, base_url=resolved_url)

    try:
        logger.info("Sending request to Oxlo.ai model=%s", resolved_model)
        response = client.chat.completions.create(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,        # Low temperature → deterministic, factual
            max_tokens=2000,
            timeout=60,
        )
    except APIConnectionError as exc:
        return _error_response(f"Connection error reaching Oxlo.ai: {exc}")
    except APITimeoutError:
        return _error_response("Request to Oxlo.ai timed out (60 s). Try again.")
    except APIStatusError as exc:
        return _error_response(
            f"Oxlo.ai API error {exc.status_code}: {exc.message}"
        )
    except Exception as exc:
        return _error_response(f"Unexpected error: {exc}")

    # ── Parse JSON response ────────────────────────────────────────────────────
    raw_text = response.choices[0].message.content or ""
    logger.debug("Raw AI response: %s", raw_text[:500])

    try:
        # Strip potential markdown fences the model may add despite instructions
        clean = raw_text.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.lower().startswith("json"):
                clean = clean[4:]
        parsed: dict = json.loads(clean.strip())
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s\nRaw: %s", exc, raw_text[:300])
        return _error_response(
            f"AI returned non-JSON response. Raw excerpt:\n{raw_text[:300]}"
        )

    # ── Enrich with material DB data ──────────────────────────────────────────
    parsed = _enrich_with_db(parsed, df)
    parsed["error"] = None
    return parsed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich_with_db(parsed: dict, df: pd.DataFrame) -> dict:
    """
    Cross-reference AI recommendations against the CSV to fill any missing
    numeric fields and attach the full Description text.
    """
    name_to_row = {
        row["Material_Name"]: row for _, row in df.iterrows()
    }
    for rec in parsed.get("recommendations", []):
        name = rec.get("material_name", "")
        db_row = name_to_row.get(name)
        if db_row is not None:
            rec.setdefault("embodied_carbon", float(db_row["Embodied_Carbon"]))
            rec.setdefault("thermal_conductivity", float(db_row["Thermal_Conductivity"]))
            rec.setdefault("durability_years", int(db_row["Durability_Years"]))
            rec["description"] = str(db_row["Description"])
    return parsed


def _error_response(message: str) -> dict:
    """Return a standardised error dict."""
    logger.error("get_ai_recommendations error: %s", message)
    return {
        "recommendations": [],
        "overall_strategy": "",
        "estimated_lifecycle_co2_reduction_pct": 0,
        "error": message,
    }
