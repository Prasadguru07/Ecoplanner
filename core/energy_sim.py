"""
core/energy_sim.py
──────────────────
EcoPlanner – Energy Simulation Module
Purpose: Wraps pybuildingenergy (and falls back to ISO 13790-style manual
         calculations) to estimate annual energy demand and embodied CO₂ for
         a simplified building envelope.

All public methods return plain Python dicts so the Streamlit UI layer never
needs to import pybuildingenergy directly.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
HEATING_DEGREE_DAYS_BASE = 18.0          # °C base for HDD calculation
COOLING_DEGREE_DAYS_BASE = 26.0          # °C base for CDD calculation
GRID_CARBON_INTENSITY_KG_PER_KWH = 0.233 # Global average (IEA 2023) kg CO₂/kWh
STANDARD_LIGHTING_W_PER_M2 = 8.0        # W/m² – ASHRAE 90.1 office baseline
OCCUPANCY_HOURS_PER_YEAR = 2500          # Typical occupied hours/year


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class BuildingInputs:
    """
    Validated inputs for a simplified single-zone building.

    Attributes
    ----------
    floor_area          : Conditioned floor area in m²
    window_to_wall_ratio: WWR as a fraction  0.0 – 1.0
    avg_outdoor_temp_c  : Annual mean outdoor dry-bulb temperature in °C
    num_storeys         : Number of above-ground storeys (default 1)
    ceiling_height_m    : Floor-to-ceiling height in metres (default 2.7)
    u_wall              : Wall U-value  W/(m²·K)  (default 0.30)
    u_roof              : Roof U-value  W/(m²·K)  (default 0.20)
    u_glazing           : Window U-value W/(m²·K) (default 2.0 – double glazed)
    shgc                : Solar Heat Gain Coefficient for glazing (default 0.4)
    hvac_cop            : HVAC Coefficient of Performance (default 3.0)
    occupants           : Number of occupants (default derived from floor area)
    """

    floor_area: float
    window_to_wall_ratio: float
    avg_outdoor_temp_c: float
    num_storeys: int = 1
    ceiling_height_m: float = 2.7
    u_wall: float = 0.30
    u_roof: float = 0.20
    u_glazing: float = 2.0
    shgc: float = 0.4
    hvac_cop: float = 3.0
    occupants: Optional[int] = None

    def __post_init__(self) -> None:
        # ── Input validation ──────────────────────────────────────────────────
        if not (10 <= self.floor_area <= 100_000):
            raise ValueError(f"floor_area {self.floor_area} must be 10–100,000 m²")
        if not (0.05 <= self.window_to_wall_ratio <= 0.90):
            raise ValueError("window_to_wall_ratio must be 0.05–0.90")
        if not (-30 <= self.avg_outdoor_temp_c <= 50):
            raise ValueError("avg_outdoor_temp_c must be –30 to 50 °C")
        if self.occupants is None:
            # 10 m² per person is a common default
            self.occupants = max(1, int(self.floor_area / 10))


@dataclass
class EnergyResult:
    """Container for all calculated energy & carbon metrics."""

    # Primary demand (kWh/year)
    heating_kwh: float = 0.0
    cooling_kwh: float = 0.0
    lighting_kwh: float = 0.0
    hvac_total_kwh: float = 0.0
    total_kwh: float = 0.0

    # Intensities
    eui_kwh_per_m2: float = 0.0          # Energy Use Intensity

    # Carbon
    operational_co2_kg: float = 0.0
    operational_co2_tonnes: float = 0.0

    # Diagnostics
    hdd: float = 0.0
    cdd: float = 0.0
    effective_u_envelope: float = 0.0

    # Human-friendly summary dict (populated in EnergyAnalyzer.analyze)
    summary: dict = field(default_factory=dict)


# ── Core Analyser ─────────────────────────────────────────────────────────────

class EnergyAnalyzer:
    """
    Simplified quasi-steady-state energy model (ISO 13790-inspired).

    Usage
    -----
    >>> analyzer = EnergyAnalyzer()
    >>> result = analyzer.analyze(floor_area=200, window_to_wall_ratio=0.30,
    ...                           avg_outdoor_temp_c=12.0)
    >>> print(result.summary)
    """

    # Internal design temperature (thermostat set-point) °C
    _T_INDOOR_HEAT = 20.0
    _T_INDOOR_COOL = 26.0

    def __init__(self) -> None:
        self._pybuildingenergy_available = self._check_pbe()

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        floor_area: float,
        window_to_wall_ratio: float,
        avg_outdoor_temp_c: float,
        **kwargs,
    ) -> EnergyResult:
        """
        Run energy simulation and return an EnergyResult.

        Parameters
        ----------
        floor_area             : Conditioned area in m²
        window_to_wall_ratio   : Fraction 0.0–1.0
        avg_outdoor_temp_c     : Annual average outdoor temperature °C
        **kwargs               : Optional overrides forwarded to BuildingInputs
        """
        inputs = BuildingInputs(
            floor_area=float(floor_area),
            window_to_wall_ratio=float(window_to_wall_ratio),
            avg_outdoor_temp_c=float(avg_outdoor_temp_c),
            **kwargs,
        )

        if self._pybuildingenergy_available:
            try:
                result = self._run_pbe(inputs)
                logger.info("pybuildingenergy simulation completed successfully.")
                return result
            except Exception as exc:
                logger.warning(
                    "pybuildingenergy failed (%s); using ISO 13790 fallback.", exc
                )

        return self._run_iso13790(inputs)

    def get_summary_dict(self, result: EnergyResult) -> dict:
        """Return a flat dict suitable for JSON export / report generation."""
        return result.summary

    # ── pybuildingenergy path ─────────────────────────────────────────────────

    @staticmethod
    def _check_pbe() -> bool:
        try:
            import pybuildingenergy  # noqa: F401
            return True
        except ImportError:
            logger.info(
                "pybuildingenergy not installed; using ISO 13790 analytical model."
            )
            return False

    def _run_pbe(self, inputs: BuildingInputs) -> EnergyResult:
        """
        Thin wrapper around pybuildingenergy's core calculation.
        Returns an EnergyResult populated from the library output.
        """
        from pybuildingenergy.source.utils import BEP_graphs  # type: ignore
        from pybuildingenergy.data.bepdataclass import BEP  # type: ignore
        from pybuildingenergy.source.buildingenergy import Building  # type: ignore

        # Build a minimal BEP data-class with our inputs
        bep = BEP(
            floor_area=inputs.floor_area,
            wwr=inputs.window_to_wall_ratio,
        )
        building = Building(bep)
        annual = building.annual_energy_demand()  # kWh/year dict

        heating = float(annual.get("heating", 0.0))
        cooling = float(annual.get("cooling", 0.0))
        lighting = (
            STANDARD_LIGHTING_W_PER_M2
            * inputs.floor_area
            * OCCUPANCY_HOURS_PER_YEAR
            / 1000
        )
        hvac_total = (heating + cooling) / inputs.hvac_cop
        total = hvac_total + lighting

        co2 = total * GRID_CARBON_INTENSITY_KG_PER_KWH

        result = EnergyResult(
            heating_kwh=round(heating, 1),
            cooling_kwh=round(cooling, 1),
            lighting_kwh=round(lighting, 1),
            hvac_total_kwh=round(hvac_total, 1),
            total_kwh=round(total, 1),
            eui_kwh_per_m2=round(total / inputs.floor_area, 2),
            operational_co2_kg=round(co2, 1),
            operational_co2_tonnes=round(co2 / 1000, 3),
        )
        result.summary = self._build_summary(inputs, result)
        return result

    # ── ISO 13790 analytical fallback ─────────────────────────────────────────

    def _run_iso13790(self, inputs: BuildingInputs) -> EnergyResult:
        """
        Monthly quasi-steady-state method (simplified single zone).

        Energy balance per month:
            Q_H = max(0, Q_transmission_loss + Q_ventilation_loss - η_H * Q_gains)
            Q_C = max(0, η_C * Q_gains - Q_transmission_loss - Q_ventilation_loss)
        """
        fa = inputs.floor_area

        # ── Geometry derivation ───────────────────────────────────────────────
        # Assume square footprint for simplicity
        footprint = fa / inputs.num_storeys
        side = math.sqrt(footprint)
        perimeter = 4 * side
        wall_area_gross = perimeter * inputs.ceiling_height_m * inputs.num_storeys
        glazing_area = wall_area_gross * inputs.window_to_wall_ratio
        opaque_wall_area = wall_area_gross - glazing_area
        roof_area = footprint  # top storey only

        # ── Thermal conductances (UA) W/K ─────────────────────────────────────
        ua_wall = opaque_wall_area * inputs.u_wall
        ua_glazing = glazing_area * inputs.u_glazing
        ua_roof = roof_area * inputs.u_roof
        # Ground floor – assume U = 0.25 W/(m²·K)
        ua_floor = footprint * 0.25
        # Infiltration: 0.5 ACH, heat capacity of air 0.34 Wh/(m³·K)
        volume = fa * inputs.ceiling_height_m
        ua_ventilation = 0.5 * volume * 0.34

        ua_total = ua_wall + ua_glazing + ua_roof + ua_floor + ua_ventilation

        # ── Solar gains (simplified horizontal irradiance) ────────────────────
        # Use sinusoidal approximation of monthly irradiance for avg temperature
        # Solar fraction proportional to glazing area * SHGC
        solar_gain_peak_w = glazing_area * inputs.shgc * 200  # 200 W/m² avg peak

        # ── Monthly simulation ────────────────────────────────────────────────
        monthly_temp_offsets = [
            -7.5, -6.5, -3.0, 2.0, 6.5, 9.5,
            11.0, 10.5, 6.5, 2.0, -3.0, -6.0,
        ]  # typical deviation from annual mean (mid-latitude NH)
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        total_heating_kwh = 0.0
        total_cooling_kwh = 0.0
        total_hdd = 0.0
        total_cdd = 0.0

        for month_idx, (offset, days) in enumerate(
            zip(monthly_temp_offsets, days_in_month)
        ):
            t_out = inputs.avg_outdoor_temp_c + offset
            hours = days * 24

            # Degree days
            hdd_month = max(0, self._T_INDOOR_HEAT - t_out) * days
            cdd_month = max(0, t_out - self._T_INDOOR_COOL) * days
            total_hdd += hdd_month
            total_cdd += cdd_month

            # Transmission + ventilation losses [kWh]
            delta_t_heat = max(0.0, self._T_INDOOR_HEAT - t_out)
            delta_t_cool = max(0.0, t_out - self._T_INDOOR_COOL)

            q_loss_heat = ua_total * delta_t_heat * hours / 1000
            q_loss_cool = ua_total * delta_t_cool * hours / 1000

            # Internal gains: people (80 W/person) + equipment (5 W/m²)
            internal_gain_w = inputs.occupants * 80 + fa * 5
            q_internal_kwh = internal_gain_w * OCCUPANCY_HOURS_PER_YEAR / 1000 / 12

            # Solar gain varies with season (simple cosine)
            solar_fraction = 0.5 + 0.5 * math.cos(
                math.pi * (month_idx - 6) / 6
            )  # peaks in winter (month 0/11)
            # Invert for cooling: solar peaks in summer
            solar_fraction_cool = 1.0 - solar_fraction
            q_solar_heat_kwh = solar_gain_peak_w * hours * solar_fraction / 1000
            q_solar_cool_kwh = solar_gain_peak_w * hours * solar_fraction_cool / 1000

            # Utilisation factors (EN ISO 13790 simplified)
            gains_heat = q_internal_kwh + q_solar_heat_kwh
            gains_cool = q_internal_kwh + q_solar_cool_kwh

            eta_h = min(1.0, q_loss_heat / gains_heat) if gains_heat > 0 else 1.0
            eta_c = min(1.0, gains_cool / max(q_loss_cool, 0.001))

            q_heating = max(0.0, q_loss_heat - eta_h * gains_heat)
            q_cooling = max(0.0, eta_c * gains_cool - q_loss_cool)

            total_heating_kwh += q_heating
            total_cooling_kwh += q_cooling

        # ── Annual totals ─────────────────────────────────────────────────────
        lighting_kwh = STANDARD_LIGHTING_W_PER_M2 * fa * OCCUPANCY_HOURS_PER_YEAR / 1000
        hvac_kwh = (total_heating_kwh + total_cooling_kwh) / inputs.hvac_cop
        total_kwh = hvac_kwh + lighting_kwh
        eui = total_kwh / fa
        co2_kg = total_kwh * GRID_CARBON_INTENSITY_KG_PER_KWH

        result = EnergyResult(
            heating_kwh=round(total_heating_kwh, 1),
            cooling_kwh=round(total_cooling_kwh, 1),
            lighting_kwh=round(lighting_kwh, 1),
            hvac_total_kwh=round(hvac_kwh, 1),
            total_kwh=round(total_kwh, 1),
            eui_kwh_per_m2=round(eui, 2),
            operational_co2_kg=round(co2_kg, 1),
            operational_co2_tonnes=round(co2_kg / 1000, 3),
            hdd=round(total_hdd, 1),
            cdd=round(total_cdd, 1),
            effective_u_envelope=round(ua_total / (wall_area_gross + roof_area + footprint), 3),
        )
        result.summary = self._build_summary(inputs, result)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary(inputs: BuildingInputs, result: EnergyResult) -> dict:
        return {
            # Inputs
            "floor_area_m2": inputs.floor_area,
            "wwr": inputs.window_to_wall_ratio,
            "avg_temp_c": inputs.avg_outdoor_temp_c,
            "occupants": inputs.occupants,
            # Energy
            "heating_kwh_yr": result.heating_kwh,
            "cooling_kwh_yr": result.cooling_kwh,
            "lighting_kwh_yr": result.lighting_kwh,
            "total_kwh_yr": result.total_kwh,
            "eui_kwh_m2_yr": result.eui_kwh_per_m2,
            # Carbon
            "operational_co2_kg_yr": result.operational_co2_kg,
            "operational_co2_tonnes_yr": result.operational_co2_tonnes,
            # Climate indicators
            "hdd": result.hdd,
            "cdd": result.cdd,
        }
