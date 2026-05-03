from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler


SCENARIOS = {
    "S0": "Unaware baseline controller + fixed maintenance",
    "S1": "Reactive maintenance + fixed control",
    "S2": "Preventive scheduled maintenance + fixed control",
    "S3": "Predictive maintenance + weight-sensitive optimization",
}

SEVERITY_LEVELS: Dict[str, Dict[str, float]] = {
    "Mild": {"B_FOUL_mult": 0.70, "DUST_RATE_mult": 0.70, "COP_AGING_RATE_mult": 0.70, "RF_STAR_mult": 0.90, "K_CLOG_mult": 0.90, "DEG_TRIGGER_shift": +0.03},
    "Moderate": {"B_FOUL_mult": 1.00, "DUST_RATE_mult": 1.00, "COP_AGING_RATE_mult": 1.00, "RF_STAR_mult": 1.00, "K_CLOG_mult": 1.00, "DEG_TRIGGER_shift": 0.00},
    "Severe": {"B_FOUL_mult": 1.35, "DUST_RATE_mult": 1.40, "COP_AGING_RATE_mult": 1.30, "RF_STAR_mult": 1.15, "K_CLOG_mult": 1.20, "DEG_TRIGGER_shift": -0.03},
    "High": {"B_FOUL_mult": 1.60, "DUST_RATE_mult": 1.65, "COP_AGING_RATE_mult": 1.50, "RF_STAR_mult": 1.25, "K_CLOG_mult": 1.30, "DEG_TRIGGER_shift": -0.05},
}

CLIMATE_LEVELS: Dict[str, Dict[str, float]] = {
    "C0_Baseline": {"temp_shift": 0.0, "summer_pulse": 0.0, "future_drift_per_year": 0.03, "rh_shift": 0.0, "solar_mult": 1.00},
    "C1_Warm": {"temp_shift": 1.5, "summer_pulse": 0.8, "future_drift_per_year": 0.04, "rh_shift": -1.0, "solar_mult": 1.03},
    "C2_Heatwave": {"temp_shift": 1.0, "summer_pulse": 3.0, "future_drift_per_year": 0.05, "rh_shift": -4.0, "solar_mult": 1.05},
    "C3_FutureHot": {"temp_shift": 4.0, "summer_pulse": 1.5, "future_drift_per_year": 0.08, "rh_shift": -2.0, "solar_mult": 1.04},
}

HVAC_PRESETS = {
    "Chiller_AHU": {"COP_COOL_NOM": 4.5, "COP_HEAT_NOM": 3.2, "FAN_EFF": 0.70},
    "VRF": {"COP_COOL_NOM": 3.8, "COP_HEAT_NOM": 3.6, "FAN_EFF": 0.62},
    "Packaged_DX": {"COP_COOL_NOM": 3.2, "COP_HEAT_NOM": 3.0, "FAN_EFF": 0.60},
    "Heat_Pump": {"COP_COOL_NOM": 3.4, "COP_HEAT_NOM": 3.8, "FAN_EFF": 0.65},
}

ZONE_TYPE_DEFAULT_FACTORS = {
    "Lecture": {"term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
    "Office": {"term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
    "Lab": {"term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
    "Corridor": {"term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
    "Service": {"term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    "Custom": {"term_factor": 1.00, "break_factor": 1.00, "summer_factor": 1.00},
}


@dataclass
class BuildingSpec:
    building_type: str = "Educational / University building"
    location: str = "User-defined"
    conditioned_area_m2: float = 5000.0
    floors: int = 4
    n_spaces: int = 40
    occupancy_density_p_m2: float = 0.08
    lighting_w_m2: float = 10.0
    equipment_w_m2: float = 8.0
    airflow_m3h_m2: float = 4.0
    infiltration_ach: float = 0.5
    sensible_w_per_person: float = 75.0
    cooling_intensity_w_m2: float = 100.0
    heating_intensity_w_m2: float = 55.0
    wall_u: float = 0.6
    roof_u: float = 0.35
    window_u: float = 2.7
    shgc: float = 0.35
    glazing_ratio: float = 0.30


@dataclass
class HVACConfig:
    years: int = 20
    days_per_year: int = 365
    hvac_system_type: str = "Chiller_AHU"
    COP_COOL_NOM: float = 4.5
    COP_HEAT_NOM: float = 3.2
    COP_AGING_RATE: float = 0.005
    FAN_EFF: float = 0.70
    T_SET: float = 23.0
    T_SP_MIN: float = 21.0
    T_SP_MAX: float = 26.0
    AF_MIN: float = 0.55
    AF_MAX: float = 1.00
    RF_STAR: float = 2e-4
    B_FOUL: float = 0.015
    RF_THRESH: float = 1.6e-4
    RF_WARN: float = 1.2e-4
    DP_CLEAN: float = 150.0
    DP_THRESH: float = 420.0
    DP_WARN: float = 320.0
    DP_MAX: float = 450.0
    DUST_RATE: float = 1.2
    K_CLOG: float = 6.0
    DEG_TRIGGER: float = 0.55
    E_PRICE: float = 0.12
    CO2_FACTOR: float = 0.536
    COST_FILTER: float = 50.0
    COST_HX: float = 300.0
    FILTER_INTERVAL: int = 90
    HX_INTERVAL: int = 180
    W_ENERGY: float = 0.35
    W_DEGRAD: float = 0.25
    W_COMFORT: float = 0.25
    W_CARBON: float = 0.15
    DT_REF_COOL: float = 15.0
    DT_REF_HEAT: float = 18.0
    A_COOL_ENV: float = 0.45
    A_HEAT_ENV: float = 0.55
    INTERNAL_USE_FACTOR: float = 0.65
    HEAT_INTERNAL_CREDIT: float = 0.60
    SOLAR_COOL_FACTOR: float = 0.12
    INFIL_COOL_FACTOR: float = 0.08
    INFIL_HEAT_FACTOR: float = 0.10
    HUMIDITY_COOL_FACTOR: float = 0.004
    HUMIDITY_COMFORT_FACTOR: float = 0.02
    APO_POP: int = 18
    APO_ITERS: int = 10
    degradation_model: str = "physics"
    LINEAR_DEG_PER_DAY: float = 0.00012
    EXP_DEG_RATE_PER_DAY: float = 0.00018


def apply_hvac_preset(cfg: HVACConfig) -> HVACConfig:
    preset = HVAC_PRESETS.get(cfg.hvac_system_type, HVAC_PRESETS["Chiller_AHU"])
    out = HVACConfig(**asdict(cfg))
    out.COP_COOL_NOM = preset["COP_COOL_NOM"]
    out.COP_HEAT_NOM = preset["COP_HEAT_NOM"]
    out.FAN_EFF = preset["FAN_EFF"]
    return out


def aggregate_zone_occupancy(bldg: BuildingSpec, zone_df: Optional[pd.DataFrame]) -> Tuple[BuildingSpec, Dict[str, float]]:
    if zone_df is None or len(zone_df) == 0:
        return bldg, {
            "mode": "general",
            "weighted_occ_density": bldg.occupancy_density_p_m2,
            "schedule_profile": {"term_factor": 0.80, "break_factor": 0.25, "summer_factor": 0.35},
        }

    df = zone_df.copy()
    required = ["zone_name", "zone_type", "area_m2", "occ_density"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Zone occupancy table missing column: {col}")

    if "term_factor" not in df.columns:
        if "schedule_factor" in df.columns:
            df["term_factor"] = df["schedule_factor"]
            df["break_factor"] = df["schedule_factor"]
            df["summer_factor"] = df["schedule_factor"]
        else:
            df["term_factor"] = np.nan
            df["break_factor"] = np.nan
            df["summer_factor"] = np.nan

    for factor_col in ["term_factor", "break_factor", "summer_factor"]:
        for i, row in df.iterrows():
            if pd.isna(row[factor_col]):
                prof = ZONE_TYPE_DEFAULT_FACTORS.get(str(row["zone_type"]), ZONE_TYPE_DEFAULT_FACTORS["Custom"])
                df.at[i, factor_col] = prof[factor_col]

    area_sum = float(df["area_m2"].sum())
    if area_sum <= 0:
        raise ValueError("Zone area total must be > 0")

    weighted_occ_density = float((df["area_m2"] * df["occ_density"]).sum() / area_sum)
    occ_weight = df["area_m2"] * df["occ_density"]
    if float(occ_weight.sum()) <= 0:
        occ_weight = df["area_m2"]

    schedule_profile = {
        "term_factor": float((occ_weight * df["term_factor"]).sum() / occ_weight.sum()),
        "break_factor": float((occ_weight * df["break_factor"]).sum() / occ_weight.sum()),
        "summer_factor": float((occ_weight * df["summer_factor"]).sum() / occ_weight.sum()),
    }

    out = BuildingSpec(**asdict(bldg))
    out.conditioned_area_m2 = area_sum
    out.n_spaces = int(len(df))
    out.occupancy_density_p_m2 = weighted_occ_density

    zone_table = df[["zone_name", "zone_type", "area_m2", "occ_density", "term_factor", "break_factor", "summer_factor"]].copy()

    return out, {
        "mode": "zone_specific",
        "weighted_occ_density": weighted_occ_density,
        "schedule_profile": schedule_profile,
        "n_zones": int(len(df)),
        "zone_table": zone_table.to_dict(orient="records"),
    }


def derive_building_numbers(bldg: BuildingSpec) -> Dict[str, float]:
    return {
        "Q_cool_des_kw": bldg.conditioned_area_m2 * bldg.cooling_intensity_w_m2 / 1000.0,
        "Q_heat_des_kw": bldg.conditioned_area_m2 * bldg.heating_intensity_w_m2 / 1000.0,
        "Q_air_nom_m3h": bldg.conditioned_area_m2 * bldg.airflow_m3h_m2,
        "N_people_max": bldg.conditioned_area_m2 * bldg.occupancy_density_p_m2,
        "Internal_kw_max": bldg.conditioned_area_m2 * (bldg.lighting_w_m2 + bldg.equipment_w_m2) / 1000.0,
    }


def synthetic_daily_weather(random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for doy in range(1, 366):
        t_mean = 21.5 + 9.0 * math.sin(2 * math.pi * (doy - 80) / 365.0) + rng.normal(0, 0.5)
        t_max = t_mean + 5.0 + 1.5 * max(math.sin(2 * math.pi * (doy - 80) / 365.0), 0.0) + rng.normal(0, 0.4)
        rh_mean = float(np.clip(65 + 15 * math.sin(2 * math.pi * (doy - 150) / 365.0) + rng.normal(0, 2.0), 25, 95))
        ghi_mean = float(max(0.0, 350 + 250 * max(math.sin(math.pi * doy / 365.0), 0.0) + rng.normal(0, 20.0)))
        rows.append({"day_of_year": doy, "T_mean_C": t_mean, "T_max_C": t_max, "RH_mean_pct": rh_mean, "GHI_mean_Wm2": ghi_mean})
    return pd.DataFrame(rows)


def read_epw_daily(epw_path: str | Path) -> pd.DataFrame:
    epw_path = Path(epw_path)
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")
    names = [
        "Year", "Month", "Day", "Hour", "Minute", "DataSource", "DryBulb", "DewPoint", "RH", "Pressure",
        "ExtHorzRad", "ExtDirNormRad", "HorzIRSky", "GHI", "DNI", "DHI", "GHIllum", "DNIllum", "DHIllum",
        "ZenLum", "WindDir", "WindSpd", "TotSkyCvr", "OpaqSkyCvr", "Visibility", "CeilingHgt", "PresWeathObs",
        "PresWeathCodes", "PrecipWater", "AerosolOptDepth", "SnowDepth", "DaysSinceSnow", "Albedo",
        "LiquidPrecipDepth", "LiquidPrecipQty",
    ]
    df = pd.read_csv(epw_path, skiprows=8, header=None, names=names)
    use = df[["Month", "Day", "DryBulb", "RH", "GHI"]].copy()
    use = use[~((use["Month"] == 2) & (use["Day"] == 29))].copy()
    daily = use.groupby(["Month", "Day"], as_index=False).agg(
        T_mean_C=("DryBulb", "mean"),
        T_max_C=("DryBulb", "max"),
        RH_mean_pct=("RH", "mean"),
        GHI_mean_Wm2=("GHI", "mean"),
    )
    daily["date"] = pd.to_datetime({"year": 2001, "month": daily["Month"], "day": daily["Day"]})
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily = daily.sort_values("day_of_year")[["day_of_year", "T_mean_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2"]].reset_index(drop=True)
    if len(daily) != 365:
        raise ValueError(f"Expected 365 daily rows after EPW aggregation, got {len(daily)}")
    return daily


def weather_summary_dict(df: pd.DataFrame, source: str, epw_path: str | None) -> Dict[str, object]:
    return {
        "source_mode": source,
        "epw_path": epw_path,
        "n_days": int(len(df)),
        "T_mean_annual_avg_C": float(df["T_mean_C"].mean()),
        "T_max_annual_avg_C": float(df["T_max_C"].mean()),
        "RH_annual_avg_pct": float(df["RH_mean_pct"].mean()),
        "GHI_annual_avg_Wm2": float(df["GHI_mean_Wm2"].mean()),
    }


def climate_and_operation_for_day(d: int, base_weather: pd.DataFrame, climate_name: str, schedule_profile: Optional[Dict[str, float]] = None) -> Tuple[float, float, float, float, float]:
    doy = d % 365 + 1
    year_idx = d // 365
    row = base_weather.iloc[doy - 1]
    rules = CLIMATE_LEVELS[climate_name]
    pulse = 0.0
    if 150 <= doy <= 260:
        phase = (doy - 150) / (260 - 150) * math.pi
        pulse = rules["summer_pulse"] * math.sin(phase)
    T_mean = float(row["T_mean_C"] + rules["temp_shift"] + pulse + rules["future_drift_per_year"] * year_idx)
    T_max = float(row["T_max_C"] + rules["temp_shift"] + 1.15 * pulse + 1.2 * rules["future_drift_per_year"] * year_idx)
    RH_mean = float(np.clip(row["RH_mean_pct"] + rules["rh_shift"], 15, 95))
    GHI_mean = float(max(0.0, row["GHI_mean_Wm2"] * rules["solar_mult"]))
    week = ((doy - 1) // 7) % 52
    in_sem = (1 <= week <= 16) or (20 <= week <= 34)
    is_summer = 180 <= doy <= 242
    if schedule_profile is None:
        occ = 0.80 if in_sem else (0.35 if is_summer else 0.25)
    else:
        occ = schedule_profile["term_factor"] if in_sem else (schedule_profile["summer_factor"] if is_summer else schedule_profile["break_factor"])
        occ = float(np.clip(occ, 0.0, 1.0))
    return T_mean, T_max, RH_mean, GHI_mean, occ


def apply_severity(cfg: HVACConfig, severity: str) -> HVACConfig:
    rules = SEVERITY_LEVELS[severity]
    out = HVACConfig(**asdict(cfg))
    out.B_FOUL *= rules["B_FOUL_mult"]
    out.DUST_RATE *= rules["DUST_RATE_mult"]
    out.COP_AGING_RATE *= rules["COP_AGING_RATE_mult"]
    out.RF_STAR *= rules["RF_STAR_mult"]
    out.K_CLOG *= rules["K_CLOG_mult"]
    out.DEG_TRIGGER = float(np.clip(out.DEG_TRIGGER + rules["DEG_TRIGGER_shift"], 0.35, 0.75))
    return out


def degradation_index(cfg: HVACConfig, rf: float, dust: float) -> Tuple[float, float]:
    dp = min(cfg.DP_CLEAN + cfg.K_CLOG * dust, cfg.DP_MAX)
    deg = 0.5 * (rf / max(cfg.RF_STAR, 1e-12)) + 0.5 * (dp / cfg.DP_MAX)
    return dp, deg


def severity_scalar(severity: str) -> float:
    return {"Mild": 0.70, "Moderate": 1.00, "Severe": 1.35, "High": 1.60}.get(severity, 1.0)


def weather_stress_scalar(T_mean: float, RH_mean: float, GHI_mean: float) -> float:
    temp_stress = max((T_mean - 24.0) / 12.0, 0.0)
    humid_stress = max((RH_mean - 60.0) / 30.0, 0.0)
    solar_stress = min(max(GHI_mean / 700.0, 0.0), 1.5)
    return 1.0 + 0.45 * temp_stress + 0.10 * humid_stress + 0.05 * solar_stress


def ts_degradation_update(cfg: HVACConfig, severity: str, prev_delta: float, T_mean: float, RH_mean: float, GHI_mean: float, model_name: str) -> Tuple[float, float, float, float]:
    sev_mult = severity_scalar(severity)
    stress = weather_stress_scalar(T_mean, RH_mean, GHI_mean)
    if model_name == "linear_ts":
        delta_next = min(1.0, prev_delta + cfg.LINEAR_DEG_PER_DAY * sev_mult * stress)
    elif model_name == "exponential_ts":
        rate = cfg.EXP_DEG_RATE_PER_DAY * sev_mult * stress
        delta_next = 1.0 - (1.0 - prev_delta) * math.exp(-rate)
    else:
        raise ValueError(f"Unsupported time-series degradation model: {model_name}")
    rf_next = min(cfg.RF_STAR, cfg.RF_STAR * min(delta_next * 1.20, 1.0))
    dp_next = min(cfg.DP_CLEAN + delta_next * (cfg.DP_MAX - cfg.DP_CLEAN), cfg.DP_MAX)
    dust_next = max((dp_next - cfg.DP_CLEAN) / max(cfg.K_CLOG, 1e-9), 0.0)
    return rf_next, dust_next, dp_next, delta_next


def simulate_baseline_no_degradation(strategy: str, climate_name: str, bldg: BuildingSpec, base_cfg: HVACConfig, base_weather: pd.DataFrame, schedule_profile: Optional[Dict[str, float]] = None, random_state: int = 42):
    cfg = apply_hvac_preset(HVACConfig(**asdict(base_cfg)))
    derived = derive_building_numbers(bldg)
    daily_rows = []
    for d in range(cfg.years * cfg.days_per_year):
        year = d // 365 + 1
        doy = d % 365 + 1
        T_mean, T_max, RH_mean, GHI_mean, occ = climate_and_operation_for_day(d, base_weather, climate_name, schedule_profile)
        T_sp = cfg.T_SET
        af = 1.0
        loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
        mode = loads["mode"]
        current_cop = cop_cooling(cfg, T_mean, 0.0, 0.0) if mode == "cooling" else cop_heating(cfg, T_mean, 0.0, 0.0)
        P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
        P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * cfg.DP_CLEAN / max(cfg.FAN_EFF, 1e-6)) / 1000.0
        E_day = (P_hvac + P_fan) * 24.0
        co2 = E_day * cfg.CO2_FACTOR
        T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0)
        comfort_dev = abs(T_zone - cfg.T_SET)
        discomfort_flag = int((occ > 0.5) and (comfort_dev > 0.3))
        daily_rows.append({
            "strategy": strategy, "severity": "Baseline_NoDegradation", "climate": climate_name,
            "scenario_combo_3axis": f"{strategy}_Baseline_NoDegradation_{climate_name}",
            "building_type": bldg.building_type, "area_m2": bldg.conditioned_area_m2, "floors": bldg.floors, "n_spaces": bldg.n_spaces,
            "hvac_system_type": cfg.hvac_system_type, "Q_cool_des_kw": derived["Q_cool_des_kw"], "Q_heat_des_kw": derived["Q_heat_des_kw"],
            "Q_air_nom_m3h": derived["Q_air_nom_m3h"], "day": d + 1, "year": year, "day_of_year": doy, "T_amb_C": T_mean, "T_max_C": T_max,
            "RH_mean_pct": RH_mean, "GHI_mean_Wm2": GHI_mean, "occ": occ, "T_sp_C": T_sp, "alpha_flow": af, "R_f": 0.0, "dust_kg": 0.0,
            "dP_Pa": cfg.DP_CLEAN, "delta": 0.0, "COP_eff": current_cop, "mode": mode, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"],
            "Q_HVAC_kw": loads["Q_HVAC_kw"], "energy_kwh_day": E_day, "co2_kg_day": co2, "comfort_dev_C": comfort_dev,
            "occupied_discomfort_flag": discomfort_flag, "cost_usd_day": E_day * cfg.E_PRICE, "hx_cleaned": 0, "filter_replaced": 0, "baseline_flag": 1,
        })
    daily = pd.DataFrame(daily_rows)
    annual = daily.groupby(["strategy", "severity", "climate", "year"], as_index=False).agg(
        annual_energy_MWh=("energy_kwh_day", lambda s: float(s.sum() / 1000.0)),
        annual_cost_usd=("cost_usd_day", "sum"),
        annual_co2_tonne=("co2_kg_day", lambda s: float(s.sum() / 1000.0)),
        mean_COP=("COP_eff", "mean"),
        mean_delta=("delta", "mean"),
        mean_comfort_dev=("comfort_dev_C", "mean"),
        mean_Q_cool_kw=("Q_cool_kw", "mean"),
        mean_Q_heat_kw=("Q_heat_kw", "mean"),
        occupied_discomfort_days=("occupied_discomfort_flag", "sum"),
    )
    summary = {
        "strategy": strategy, "severity": "Baseline_NoDegradation", "climate": climate_name,
        "scenario_combo_3axis": f"{strategy}_Baseline_NoDegradation_{climate_name}",
        "Building Area m2": bldg.conditioned_area_m2, "No. of Spaces": bldg.n_spaces, "HVAC System": cfg.hvac_system_type,
        "Cooling Design kW": derived["Q_cool_des_kw"], "Heating Design kW": derived["Q_heat_des_kw"], "Airflow m3h": derived["Q_air_nom_m3h"],
        "Total Energy MWh": float(daily["energy_kwh_day"].sum() / 1000.0), "Total Cost USD": float(daily["cost_usd_day"].sum()),
        "Total CO2 tonne": float(daily["co2_kg_day"].sum() / 1000.0), "Mean COP": float(daily["COP_eff"].mean()), "Mean Degradation Index": 0.0,
        "Mean Comfort Deviation C": float(daily["comfort_dev_C"].mean()), "Mean Cooling Load kW": float(daily["Q_cool_kw"].mean()),
        "Mean Heating Load kW": float(daily["Q_heat_kw"].mean()), "Occupied Discomfort Days": int(daily["occupied_discomfort_flag"].sum()),
        "Filter Replacements count": 0, "HX Cleanings count": 0,
    }
    return daily, annual, summary


def cop_cooling(cfg: HVACConfig, T_a: float, year_frac: float, rf: float) -> float:
    cop_aged = cfg.COP_COOL_NOM - cfg.COP_AGING_RATE * year_frac
    cop_foul = cop_aged / (1.0 + 0.45 * (rf / max(cfg.RF_STAR, 1e-12)))
    cop_amb = 1.0 - 0.018 * max(T_a - 25.0, 0.0)
    return min(cfg.COP_COOL_NOM, max(0.8, cop_foul * cop_amb))


def cop_heating(cfg: HVACConfig, T_a: float, year_frac: float, rf: float) -> float:
    cop_aged = cfg.COP_HEAT_NOM - 0.6 * cfg.COP_AGING_RATE * year_frac
    cop_foul = cop_aged / (1.0 + 0.30 * (rf / max(cfg.RF_STAR, 1e-12)))
    cop_amb = 1.0 - 0.010 * max(18.0 - T_a, 0.0)
    return min(cfg.COP_HEAT_NOM, max(0.8, cop_foul * cop_amb))


def cooling_heating_loads(bldg: BuildingSpec, cfg: HVACConfig, derived: Dict[str, float], T_mean: float, RH_mean: float, GHI_mean: float, T_sp: float, occ: float, doy: int) -> Dict[str, float]:
    q_cool_des = derived["Q_cool_des_kw"]
    q_heat_des = derived["Q_heat_des_kw"]
    n_people = derived["N_people_max"] * occ
    internal_kw = derived["Internal_kw_max"] * max(0.20, occ * cfg.INTERNAL_USE_FACTOR + 0.20)
    dT_cool = max(T_mean - T_sp, 0.0)
    dT_heat = max(T_sp - T_mean, 0.0)
    ghi_norm = min(max(GHI_mean / 700.0, 0.0), 1.5)
    humidity_mult = 1.0 + cfg.HUMIDITY_COOL_FACTOR * max(RH_mean - 60.0, 0.0)
    q_cool_env = cfg.A_COOL_ENV * q_cool_des * (dT_cool / cfg.DT_REF_COOL)
    q_cool_solar = cfg.SOLAR_COOL_FACTOR * q_cool_des * ghi_norm * max(math.sin(math.pi * doy / 365.0), 0.0)
    q_cool_occ = n_people * bldg.sensible_w_per_person / 1000.0
    q_cool_inf = cfg.INFIL_COOL_FACTOR * q_cool_des * (dT_cool / cfg.DT_REF_COOL)
    q_cool = max(0.0, min((q_cool_env + q_cool_solar + internal_kw + q_cool_occ + q_cool_inf) * humidity_mult, 1.20 * q_cool_des))
    q_heat_env = cfg.A_HEAT_ENV * q_heat_des * (dT_heat / cfg.DT_REF_HEAT)
    q_heat_inf = cfg.INFIL_HEAT_FACTOR * q_heat_des * (dT_heat / cfg.DT_REF_HEAT)
    q_internal_credit = cfg.HEAT_INTERNAL_CREDIT * (internal_kw + q_cool_occ)
    q_heat = max(0.0, min(q_heat_env + q_heat_inf - q_internal_credit, 1.20 * q_heat_des))
    mode = "cooling" if dT_cool >= dT_heat else "heating"
    q_hvac = q_cool if mode == "cooling" else q_heat
    return {"Q_cool_kw": q_cool, "Q_heat_kw": q_heat, "Q_HVAC_kw": q_hvac, "mode": mode, "people": n_people, "internal_kw": internal_kw}


def evaluate_controls(bldg: BuildingSpec, cfg: HVACConfig, derived: Dict[str, float], T_mean: float, RH_mean: float, GHI_mean: float, occ: float, year_frac: float, doy: int, rf: float, dust: float, T_sp: float, af: float) -> Dict[str, float]:
    rf_next = cfg.RF_STAR - (cfg.RF_STAR - rf) * math.exp(-cfg.B_FOUL)
    dust_next = dust + cfg.DUST_RATE * af
    dp_next, deg_next = degradation_index(cfg, rf_next, dust_next)
    loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
    mode = loads["mode"]
    current_cop = cop_cooling(cfg, T_mean, year_frac, rf_next) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf_next)
    P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
    P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp_next / max(cfg.FAN_EFF, 1e-6)) / 1000.0
    P_tot = P_hvac + P_fan
    E_day = P_tot * 24.0
    co2 = E_day * cfg.CO2_FACTOR
    T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * deg_next * occ
    comfort_dev = abs(T_zone - cfg.T_SET)
    e_n = E_day / max((derived["Q_cool_des_kw"] * 24.0 * 1.5), 1e-9)
    d_n = deg_next
    c_n = comfort_dev / 3.0
    co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * 24.0 * 1.5), 1e-9)
    J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
    return {"rf_next": rf_next, "dust_next": dust_next, "dp_next": dp_next, "deg_next": deg_next, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "E_day": E_day, "co2": co2, "comfort_dev": comfort_dev, "objective": J}


def optimize_s3(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, prev_T_sp, prev_af, rng):
    center = np.array([prev_T_sp, prev_af], dtype=float)
    sigma = np.array([1.4, 0.18], dtype=float)
    best_x = center.copy()
    best_obj = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, best_x[0], best_x[1])["objective"]
    for _ in range(cfg.APO_ITERS):
        pop = []
        candidates = [best_x, np.array([cfg.T_SET, 1.0]), np.array([prev_T_sp, prev_af])]
        while len(candidates) < cfg.APO_POP:
            x = center + rng.normal(0.0, 1.0, size=2) * sigma
            x[0] = float(np.clip(x[0], cfg.T_SP_MIN, cfg.T_SP_MAX))
            x[1] = float(np.clip(x[1], cfg.AF_MIN, cfg.AF_MAX))
            candidates.append(x)
        for x in candidates:
            obj = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, float(x[0]), float(x[1]))["objective"]
            pop.append((obj, x))
        pop.sort(key=lambda t: t[0])
        elite = pop[: max(3, cfg.APO_POP // 4)]
        elite_x = np.array([e[1] for e in elite])
        center = elite_x.mean(axis=0)
        center[0] = float(np.clip(center[0], cfg.T_SP_MIN, cfg.T_SP_MAX))
        center[1] = float(np.clip(center[1], cfg.AF_MIN, cfg.AF_MAX))
        if elite[0][0] < best_obj:
            best_x = elite[0][1].copy()
            best_obj = elite[0][0]
        sigma *= 0.72
    return float(best_x[0]), float(best_x[1])


def simulate_combo(strategy: str, severity: str, climate_name: str, bldg: BuildingSpec, base_cfg: HVACConfig, base_weather: pd.DataFrame, schedule_profile: Optional[Dict[str, float]] = None, random_state: int = 42, degradation_model: str = "physics"):
    cfg = apply_hvac_preset(apply_severity(base_cfg, severity))
    derived = derive_building_numbers(bldg)
    rng = np.random.default_rng(random_state + sum(ord(c) for c in strategy + severity + climate_name))
    rf = 0.0
    dust = 0.0
    delta_state = 0.0
    T_sp = cfg.T_SET
    af = 1.0
    daily_rows = []
    hx_count = 0
    filter_count = 0

    for d in range(cfg.years * cfg.days_per_year):
        year = d // 365 + 1
        doy = d % 365 + 1
        year_frac = d / 365.0
        T_mean, T_max, RH_mean, GHI_mean, occ = climate_and_operation_for_day(d, base_weather, climate_name, schedule_profile)

        if strategy == "S3":
            T_sp, af = optimize_s3(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af, rng)
        else:
            T_sp = cfg.T_SET
            af = 1.0

        if degradation_model == "physics":
            res = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af)
            rf = res["rf_next"]; dust = res["dust_next"]; dp = res["dp_next"]; deg = res["deg_next"]; delta_state = deg
        elif degradation_model in ["linear_ts", "exponential_ts"]:
            rf, dust, dp, deg = ts_degradation_update(cfg=cfg, severity=severity, prev_delta=delta_state, T_mean=T_mean, RH_mean=RH_mean, GHI_mean=GHI_mean, model_name=degradation_model)
            delta_state = deg
            loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
            mode = loads["mode"]
            current_cop = cop_cooling(cfg, T_mean, year_frac, rf) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf)
            P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
            P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp / max(cfg.FAN_EFF, 1e-6)) / 1000.0
            P_tot = P_hvac + P_fan
            E_day = P_tot * 24.0
            co2 = E_day * cfg.CO2_FACTOR
            T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * deg * occ
            comfort_dev = abs(T_zone - cfg.T_SET)
            e_n = E_day / max((derived["Q_cool_des_kw"] * 24.0 * 1.5), 1e-9); d_n = deg; c_n = comfort_dev / 3.0
            co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * 24.0 * 1.5), 1e-9)
            J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
            res = {"rf_next": rf, "dust_next": dust, "dp_next": dp, "deg_next": deg, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "E_day": E_day, "co2": co2, "comfort_dev": comfort_dev, "objective": J}
        else:
            raise ValueError(f"Unsupported degradation_model: {degradation_model}")

        do_hx = do_filter = False
        maint_cost = 0.0
        if strategy == "S0":
            do_hx = (doy - 1) == 180; do_filter = (doy - 1) in (0, 90, 180, 270)
        elif strategy == "S1":
            do_hx = rf >= cfg.RF_THRESH; do_filter = dp >= cfg.DP_THRESH
        elif strategy == "S2":
            do_hx = (d % cfg.HX_INTERVAL) == 0; do_filter = (d % cfg.FILTER_INTERVAL) == 0
        elif strategy == "S3":
            do_hx = (rf >= cfg.RF_WARN) or (deg >= cfg.DEG_TRIGGER); do_filter = (dp >= cfg.DP_WARN) or (deg >= cfg.DEG_TRIGGER)

        if do_hx:
            rf = 0.0; hx_count += 1; maint_cost += cfg.COST_HX
        if do_filter:
            dust = 0.0; filter_count += 1; maint_cost += cfg.COST_FILTER

        if degradation_model in ["linear_ts", "exponential_ts"]:
            if do_hx and do_filter:
                delta_state *= 0.40
            elif do_hx or do_filter:
                delta_state *= 0.65
            rf = min(cfg.RF_STAR, cfg.RF_STAR * min(delta_state * 1.20, 1.0))
            dp = min(cfg.DP_CLEAN + delta_state * (cfg.DP_MAX - cfg.DP_CLEAN), cfg.DP_MAX)
            dust = max((dp - cfg.DP_CLEAN) / max(cfg.K_CLOG, 1e-9), 0.0)

        if degradation_model == "physics":
            res = evaluate_controls(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, occ, year_frac, doy, rf, dust, T_sp, af)
        else:
            loads = cooling_heating_loads(bldg, cfg, derived, T_mean, RH_mean, GHI_mean, T_sp, occ, doy)
            mode = loads["mode"]
            current_cop = cop_cooling(cfg, T_mean, year_frac, rf) if mode == "cooling" else cop_heating(cfg, T_mean, year_frac, rf)
            P_hvac = loads["Q_HVAC_kw"] / max(current_cop, 0.8)
            P_fan = (derived["Q_air_nom_m3h"] * af / 3600.0 * dp / max(cfg.FAN_EFF, 1e-6)) / 1000.0
            P_tot = P_hvac + P_fan
            E_day = P_tot * 24.0
            co2 = E_day * cfg.CO2_FACTOR
            T_zone = T_sp + 2.2 * (1.0 - af) * occ + 0.08 * max(T_mean - T_sp, 0.0) - 0.06 * max(T_sp - T_mean, 0.0) + cfg.HUMIDITY_COMFORT_FACTOR * max(RH_mean - 60.0, 0.0) + 0.60 * delta_state * occ
            comfort_dev = abs(T_zone - cfg.T_SET)
            e_n = E_day / max((derived["Q_cool_des_kw"] * 24.0 * 1.5), 1e-9); d_n = delta_state; c_n = comfort_dev / 3.0
            co2_n = co2 / max((derived["Q_cool_des_kw"] * cfg.CO2_FACTOR * 24.0 * 1.5), 1e-9)
            J = cfg.W_ENERGY * e_n + cfg.W_DEGRAD * d_n + cfg.W_COMFORT * c_n + cfg.W_CARBON * co2_n
            res = {"rf_next": rf, "dust_next": dust, "dp_next": dp, "deg_next": delta_state, "cop": current_cop, "Q_cool_kw": loads["Q_cool_kw"], "Q_heat_kw": loads["Q_heat_kw"], "Q_HVAC_kw": loads["Q_HVAC_kw"], "mode": mode, "P_tot": P_tot, "E_day": E_day, "co2": co2, "comfort_dev": comfort_dev, "objective": J}

        discomfort_flag = int((occ > 0.5) and (res["comfort_dev"] > 0.3))
        cost_day = res["E_day"] * cfg.E_PRICE + maint_cost
        daily_rows.append({
            "strategy": strategy, "severity": severity, "climate": climate_name, "scenario_combo_3axis": f"{strategy}_{severity}_{climate_name}",
            "building_type": bldg.building_type, "area_m2": bldg.conditioned_area_m2, "floors": bldg.floors, "n_spaces": bldg.n_spaces,
            "hvac_system_type": cfg.hvac_system_type, "Q_cool_des_kw": derived["Q_cool_des_kw"], "Q_heat_des_kw": derived["Q_heat_des_kw"], "Q_air_nom_m3h": derived["Q_air_nom_m3h"],
            "day": d + 1, "year": year, "day_of_year": doy, "T_amb_C": T_mean, "T_max_C": T_max, "RH_mean_pct": RH_mean, "GHI_mean_Wm2": GHI_mean,
            "occ": occ, "T_sp_C": T_sp, "alpha_flow": af, "R_f": rf, "dust_kg": dust, "dP_Pa": res["dp_next"], "delta": res["deg_next"], "COP_eff": res["cop"],
            "mode": res["mode"], "Q_cool_kw": res["Q_cool_kw"], "Q_heat_kw": res["Q_heat_kw"], "Q_HVAC_kw": res["Q_HVAC_kw"], "energy_kwh_day": res["E_day"],
            "co2_kg_day": res["co2"], "comfort_dev_C": res["comfort_dev"], "occupied_discomfort_flag": discomfort_flag, "cost_usd_day": cost_day,
            "hx_cleaned": int(do_hx), "filter_replaced": int(do_filter),
        })

    daily = pd.DataFrame(daily_rows)
    annual = daily.groupby(["strategy", "severity", "climate", "year"], as_index=False).agg(
        annual_energy_MWh=("energy_kwh_day", lambda s: float(s.sum() / 1000.0)),
        annual_cost_usd=("cost_usd_day", "sum"),
        annual_co2_tonne=("co2_kg_day", lambda s: float(s.sum() / 1000.0)),
        mean_COP=("COP_eff", "mean"),
        mean_delta=("delta", "mean"),
        mean_comfort_dev=("comfort_dev_C", "mean"),
        mean_Q_cool_kw=("Q_cool_kw", "mean"),
        mean_Q_heat_kw=("Q_heat_kw", "mean"),
        occupied_discomfort_days=("occupied_discomfort_flag", "sum"),
        filter_replacements=("filter_replaced", "sum"),
        hx_cleanings=("hx_cleaned", "sum"),
    )
    summary = {
        "strategy": strategy, "severity": severity, "climate": climate_name, "scenario_combo_3axis": f"{strategy}_{severity}_{climate_name}",
        "Building Area m2": bldg.conditioned_area_m2, "No. of Spaces": bldg.n_spaces, "HVAC System": cfg.hvac_system_type,
        "Cooling Design kW": derived["Q_cool_des_kw"], "Heating Design kW": derived["Q_heat_des_kw"], "Airflow m3h": derived["Q_air_nom_m3h"],
        "Total Energy MWh": float(daily["energy_kwh_day"].sum() / 1000.0), "Total Cost USD": float(daily["cost_usd_day"].sum()), "Total CO2 tonne": float(daily["co2_kg_day"].sum() / 1000.0),
        "Mean COP": float(daily["COP_eff"].mean()), "Mean Degradation Index": float(daily["delta"].mean()), "Mean Comfort Deviation C": float(daily["comfort_dev_C"].mean()),
        "Mean Cooling Load kW": float(daily["Q_cool_kw"].mean()), "Mean Heating Load kW": float(daily["Q_heat_kw"].mean()), "Occupied Discomfort Days": int(daily["occupied_discomfort_flag"].sum()),
        "Filter Replacements count": int(filter_count), "HX Cleanings count": int(hx_count),
    }
    return daily, annual, summary


def save_figure(df: pd.DataFrame, x: str, y: str, hue: Optional[str], title: str, out_png: Path, out_svg: Optional[Path] = None):
    plt.figure(figsize=(8, 5))
    if hue and hue in df.columns:
        for val, grp in df.groupby(hue):
            plt.plot(grp[x], grp[y], marker="o", label=str(val))
        plt.legend(frameon=False)
    else:
        plt.plot(df[x], df[y], marker="o")
    plt.xlabel(x); plt.ylabel(y); plt.title(title); plt.grid(alpha=0.25); plt.tight_layout(); plt.savefig(out_png, dpi=600)
    if out_svg: plt.savefig(out_svg)
    plt.close()


def save_heatmap(summary_df: pd.DataFrame, climate_name: str, value_col: str, out_png: Path, out_svg: Optional[Path] = None):
    subset = summary_df[summary_df["climate"] == climate_name].copy()
    pivot = subset.pivot(index="severity", columns="strategy", values=value_col)
    order = [s for s in ["Mild", "Moderate", "Severe", "High"] if s in pivot.index]
    pivot = pivot.reindex(order)
    plt.figure(figsize=(7.5, 5.5)); plt.imshow(pivot.values, aspect="auto"); plt.xticks(range(len(pivot.columns)), pivot.columns); plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title(f"{value_col} | {climate_name}"); plt.colorbar(label=value_col)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            plt.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=600)
    if out_svg: plt.savefig(out_svg)
    plt.close()


def export_excel_report(out: Path, summary_df: pd.DataFrame, annual_df: pd.DataFrame, daily_df: pd.DataFrame, meta: Dict[str, object]):
    with pd.ExcelWriter(out / "results_export.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([meta]).to_excel(writer, sheet_name="run_metadata", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        annual_df.to_excel(writer, sheet_name="annual", index=False)
        daily_df.head(5000).to_excel(writer, sheet_name="dataset_head", index=False)


def export_pdf_report(out: Path, summary_df: pd.DataFrame, annual_df: pd.DataFrame, meta: Dict[str, object]):
    pdf_path = out / "results_report.pdf"
    figs = sorted((out / "figures").glob("*.png"))
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis("off")
        txt = ("HVAC Research Modeling Suite v3\n\n"
               f"Axis mode: {meta.get('axis_mode')}\n"
               f"Building type: {meta.get('building_spec', {}).get('building_type')}\n"
               f"HVAC system: {meta.get('hvac_config', {}).get('hvac_system_type')}\n"
               f"Area (m²): {meta.get('building_spec', {}).get('conditioned_area_m2')}\n"
               f"Spaces: {meta.get('building_spec', {}).get('n_spaces')}\n"
               f"Weather: {meta.get('weather_summary', {}).get('source_mode')}\n")
        plt.text(0.08, 0.92, txt, va="top", fontsize=14); pdf.savefig(fig, dpi=300); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis("off"); plt.text(0.05, 0.97, "Summary table (top rows)", va="top", fontsize=14)
        show_df = summary_df.head(18).copy(); tbl = plt.table(cellText=show_df.values, colLabels=show_df.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(6); tbl.scale(1, 1.2); pdf.savefig(fig, dpi=300); plt.close(fig)

        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis("off"); plt.text(0.05, 0.97, "Annual table (top rows)", va="top", fontsize=14)
        show_df = annual_df.head(18).copy(); tbl = plt.table(cellText=show_df.values, colLabels=show_df.columns, loc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(6); tbl.scale(1, 1.2); pdf.savefig(fig, dpi=300); plt.close(fig)

        for img in figs[:8]:
            arr = plt.imread(img); fig = plt.figure(figsize=(8.27, 11.69)); plt.imshow(arr); plt.axis("off"); plt.title(img.name); pdf.savefig(fig, dpi=300); plt.close(fig)


def run_scenario_model(output_dir: str | Path, axis_mode: str, bldg: BuildingSpec, cfg: HVACConfig, weather_mode: str = "synthetic", epw_path: str | None = None, fixed_strategy: str = "S3", fixed_severity: str = "Moderate", fixed_climate: str = "C0_Baseline", zone_df: Optional[pd.DataFrame] = None, random_state: int = 42, include_baseline_layer: bool = True, degradation_model: str = "physics") -> Dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); figures_dir = out / "figures"; figures_dir.mkdir(exist_ok=True)
    bldg, zone_meta = aggregate_zone_occupancy(bldg, zone_df); cfg = apply_hvac_preset(cfg)
    if weather_mode == "epw" and epw_path:
        base_weather = read_epw_daily(epw_path); weather_meta = weather_summary_dict(base_weather, "epw", epw_path)
    else:
        base_weather = synthetic_daily_weather(random_state); weather_meta = weather_summary_dict(base_weather, "synthetic", None)

    if axis_mode == "one_severity":
        combos = [(fixed_strategy, sev, fixed_climate) for sev in SEVERITY_LEVELS.keys()]
        dataset_name, summary_name, annual_name = "one_axis_severity_ml_dataset.csv", "one_axis_severity_summary.csv", "annual_one_axis_severity.csv"
    elif axis_mode == "one_strategy":
        combos = [(stg, fixed_severity, fixed_climate) for stg in SCENARIOS.keys()]
        dataset_name, summary_name, annual_name = "one_axis_strategy_ml_dataset.csv", "one_axis_strategy_summary.csv", "annual_one_axis_strategy.csv"
    elif axis_mode == "two_axis":
        combos = [(stg, sev, fixed_climate) for sev in SEVERITY_LEVELS.keys() for stg in SCENARIOS.keys()]
        dataset_name, summary_name, annual_name = "matrix_ml_dataset.csv", "matrix_summary.csv", "annual_matrix.csv"
    elif axis_mode == "three_axis":
        combos = [(stg, sev, cli) for cli in CLIMATE_LEVELS.keys() for sev in SEVERITY_LEVELS.keys() for stg in SCENARIOS.keys()]
        dataset_name, summary_name, annual_name = "three_axis_ml_dataset.csv", "three_axis_summary.csv", "annual_three_axis.csv"
    else:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")

    all_daily, all_annual, summaries = [], [], []
    schedule_profile = zone_meta.get("schedule_profile", None)
    for strategy, severity, climate_name in combos:
        daily, annual, summary = simulate_combo(strategy, severity, climate_name, bldg, cfg, base_weather, schedule_profile, random_state, degradation_model)
        all_daily.append(daily); all_annual.append(annual); summaries.append(summary)

    daily_df = pd.concat(all_daily, ignore_index=True)
    annual_df = pd.concat(all_annual, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    if include_baseline_layer:
        baseline_daily, baseline_annual, baseline_summary = simulate_baseline_no_degradation(strategy=fixed_strategy if axis_mode == "one_severity" else "S2", climate_name=fixed_climate, bldg=bldg, base_cfg=cfg, base_weather=base_weather, schedule_profile=schedule_profile, random_state=random_state)
        pd.DataFrame(baseline_daily).to_csv(out / "baseline_no_degradation_daily.csv", index=False)
        pd.DataFrame(baseline_annual).to_csv(out / "baseline_no_degradation_annual.csv", index=False)
        pd.DataFrame([baseline_summary]).to_csv(out / "baseline_no_degradation_summary.csv", index=False)

    daily_df.to_csv(out / dataset_name, index=False)
    daily_df.to_csv(out / "matrix_ml_dataset.csv", index=False)
    annual_df.to_csv(out / annual_name, index=False)
    summary_df.to_csv(out / summary_name, index=False)
    base_weather.to_csv(out / "baseline_daily_weather.csv", index=False)

    meta = {
        "building_spec": asdict(bldg), "hvac_config": asdict(cfg), "weather_summary": weather_meta, "zone_occupancy_meta": zone_meta,
        "axis_mode": axis_mode, "fixed_strategy": fixed_strategy, "fixed_severity": fixed_severity, "fixed_climate": fixed_climate,
        "available_hvac_types": list(HVAC_PRESETS.keys()), "available_severity_levels": list(SEVERITY_LEVELS.keys()),
        "available_climate_levels": list(CLIMATE_LEVELS.keys()), "degradation_model": degradation_model, "include_baseline_layer": include_baseline_layer,
    }
    with open(out / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if axis_mode in ["one_severity", "one_strategy"]:
        key = "severity" if axis_mode == "one_severity" else "strategy"
        save_figure(summary_df, key, "Total Energy MWh", None, f"Total Energy vs {key.title()}", figures_dir / "energy_vs_axis.png", figures_dir / "energy_vs_axis.svg")
        save_figure(summary_df, key, "Mean Degradation Index", None, f"Degradation vs {key.title()}", figures_dir / "degradation_vs_axis.png", figures_dir / "degradation_vs_axis.svg")
        save_figure(summary_df, key, "Mean Comfort Deviation C", None, f"Comfort vs {key.title()}", figures_dir / "comfort_vs_axis.png", figures_dir / "comfort_vs_axis.svg")
    elif axis_mode == "two_axis":
        save_figure(summary_df, "strategy", "Total Energy MWh", "severity", "Energy by Strategy and Severity", figures_dir / "energy_by_strategy_severity.png", figures_dir / "energy_by_strategy_severity.svg")
        save_figure(summary_df, "strategy", "Mean Degradation Index", "severity", "Degradation by Strategy and Severity", figures_dir / "degradation_by_strategy_severity.png", figures_dir / "degradation_by_strategy_severity.svg")
    elif axis_mode == "three_axis":
        for cli in CLIMATE_LEVELS.keys():
            save_heatmap(summary_df, cli, "Total Energy MWh", figures_dir / f"heatmap_energy_{cli}.png", figures_dir / f"heatmap_energy_{cli}.svg")
            save_heatmap(summary_df, cli, "Mean Degradation Index", figures_dir / f"heatmap_degradation_{cli}.png", figures_dir / f"heatmap_degradation_{cli}.svg")
            save_heatmap(summary_df, cli, "Mean Comfort Deviation C", figures_dir / f"heatmap_comfort_{cli}.png", figures_dir / f"heatmap_comfort_{cli}.svg")

    export_excel_report(out, summary_df, annual_df, daily_df, meta)
    export_pdf_report(out, summary_df, annual_df, meta)

    return {
        "dataset_csv": str(out / dataset_name), "matrix_ml_dataset_csv": str(out / "matrix_ml_dataset.csv"), "summary_csv": str(out / summary_name),
        "annual_csv": str(out / annual_name), "excel_report": str(out / "results_export.xlsx"), "pdf_report": str(out / "results_report.pdf"),
        "figures_dir": str(figures_dir), "baseline_daily_csv": str(out / "baseline_no_degradation_daily.csv") if include_baseline_layer else "",
        "baseline_summary_csv": str(out / "baseline_no_degradation_summary.csv") if include_baseline_layer else "",
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"RMSE": float(np.sqrt(mse)), "MAE": float(mean_absolute_error(y_true, y_pred)), "R2": float(r2_score(y_true, y_pred))}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "day_of_year" not in out.columns and "day" in out.columns:
        out["day_of_year"] = ((out["day"] - 1) % 365) + 1
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.0)
    if "time_idx" not in out.columns and "day" in out.columns:
        out["time_idx"] = out["day"].astype(int)
    if "scenario_key" not in out.columns:
        if "scenario_combo_3axis" in out.columns:
            out["scenario_key"] = out["scenario_combo_3axis"].astype(str)
        else:
            parts = []
            for c in ["strategy", "severity", "climate"]:
                if c in out.columns:
                    parts.append(out[c].astype(str))
            if parts:
                val = parts[0]
                for p in parts[1:]:
                    val = val + "_" + p
                out["scenario_key"] = val
            else:
                out["scenario_key"] = "case"
    return out


def add_group_lags(df, group_col, cols, lags):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag{lag}"] = out.groupby(group_col)[col].shift(lag)
    return out


def prepare_dataset_for_ml(df):
    out = df.copy()
    for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"]:
        if c in out.columns:
            out[c] = out[c].astype(str)
    lag_cols = ["energy_kwh_day", "delta", "comfort_dev_C", "COP_eff", "T_amb_C", "occ", "R_f", "dP_Pa", "hx_cleaned", "filter_replaced", "alpha_flow", "T_sp_C", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "Q_cool_kw", "Q_heat_kw"]
    out = add_group_lags(out, "scenario_key", lag_cols, [1, 7])
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def feature_map(df):
    cats = [c for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"] if c in df.columns]
    common = cats + [c for c in ["year", "day_of_year", "doy_sin", "doy_cos", "occ", "T_amb_C", "T_sp_C", "alpha_flow", "hx_cleaned_lag1", "filter_replaced_lag1", "hx_cleaned_lag7", "filter_replaced_lag7", "T_max_C", "RH_mean_pct", "GHI_mean_Wm2", "Q_cool_kw", "Q_heat_kw"] if c in df.columns]
    fmap = {}
    if "energy_kwh_day" in df.columns:
        fmap["energy_kwh_day"] = common + [c for c in ["R_f", "dP_Pa", "delta_lag1", "delta_lag7", "energy_kwh_day_lag1", "energy_kwh_day_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    if "delta" in df.columns:
        fmap["delta"] = common + [c for c in ["energy_kwh_day_lag1", "energy_kwh_day_lag7", "delta_lag1", "delta_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "R_f_lag1", "R_f_lag7", "dP_Pa_lag1", "dP_Pa_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    if "comfort_dev_C" in df.columns:
        fmap["comfort_dev_C"] = common + [c for c in ["R_f", "dP_Pa", "delta", "energy_kwh_day_lag1", "energy_kwh_day_lag7", "COP_eff_lag1", "COP_eff_lag7", "T_amb_C_lag1", "T_amb_C_lag7", "occ_lag1", "occ_lag7", "T_sp_C_lag1", "alpha_flow_lag1", "RH_mean_pct_lag1", "GHI_mean_Wm2_lag1"] if c in df.columns]
    return fmap


def auto_year_split(df):
    years = sorted(df["year"].dropna().astype(int).unique().tolist())
    n = len(years)
    if n < 3:
        raise ValueError("Need at least 3 years in the dataset.")
    if n >= 20 and years[:20] == list(range(1, 21)):
        train_years, valid_years, test_years = list(range(1, 15)), [15, 16], [17, 18, 19, 20]
    else:
        n_train = max(1, int(round(n * 0.6)))
        n_valid = max(1, int(round(n * 0.2)))
        n_test = n - n_train - n_valid
        if n_test < 1:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_valid -= 1
        train_years = years[:n_train]
        valid_years = years[n_train:n_train+n_valid]
        test_years = years[n_train+n_valid:]
        if not valid_years:
            valid_years = [train_years.pop()]
        if not test_years:
            test_years = [valid_years.pop()]
            if not valid_years:
                valid_years = [train_years.pop()]
    return (df[df["year"].isin(train_years)].copy(), df[df["year"].isin(valid_years)].copy(), df[df["year"].isin(test_years)].copy(), {"train_years": train_years, "valid_years": valid_years, "test_years": test_years})


def save_scatter(y_true, y_pred, title, out_path):
    plt.figure(figsize=(6, 6)); plt.scatter(y_true, y_pred, alpha=0.35)
    lo = min(float(np.min(y_true)), float(np.min(y_pred))); hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "--"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=600); plt.close()


def export_surrogate_excel_report(out: Path, overall_df: pd.DataFrame):
    with pd.ExcelWriter(out / "surrogate_export.xlsx", engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="overall_metrics", index=False)


def export_surrogate_pdf_report(out: Path, overall_df: pd.DataFrame, shap_notes: List[str]):
    figs = sorted((out / "figures").glob("*.png"))
    with PdfPages(out / "surrogate_report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69)); plt.axis("off")
        plt.text(0.06, 0.96, "Surrogate Model Report", va="top", fontsize=16)
        plt.text(0.06, 0.90, overall_df.to_string(index=False), va="top", family="monospace", fontsize=8)
        if shap_notes:
            plt.text(0.06, 0.52, "Key SHAP notes:\n- " + "\n- ".join(shap_notes), va="top", fontsize=10)
        pdf.savefig(fig, dpi=300); plt.close(fig)
        for img in figs[:8]:
            arr = plt.imread(img); fig = plt.figure(figsize=(8.27, 11.69)); plt.imshow(arr); plt.axis("off"); plt.title(img.name); pdf.savefig(fig, dpi=300); plt.close(fig)


def train_surrogate_models(input_csv: str | Path, output_dir: str | Path, n_iter_search: int = 6, shap_sample: int = 1000, random_state: int = 42) -> Dict[str, str]:
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed. Install dependencies from requirements.txt")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); figs = out / "figures"; figs.mkdir(exist_ok=True)
    raw = pd.read_csv(input_csv); data = prepare_dataset_for_ml(add_time_features(raw)); data.to_csv(out / "prepared_dataset.csv", index=False)
    fmap = feature_map(data); overall_rows = []; shap_notes = []

    for target, feats in fmap.items():
        df = data.dropna(subset=list(set(feats + [target]))).copy()
        train_df, valid_df, test_df, split_info = auto_year_split(df)
        cat_features = [c for c in ["strategy", "severity", "climate", "scenario_key", "hvac_system_type", "mode"] if c in feats]
        param_dist = {"iterations": [300, 600, 1000], "learning_rate": [0.01, 0.03, 0.05, 0.1], "depth": [4, 5, 6, 8], "l2_leaf_reg": [1, 3, 5, 7, 10], "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "random_strength": [0.0, 0.5, 1.0, 2.0]}
        best = None
        for params in ParameterSampler(param_dist, n_iter=n_iter_search, random_state=random_state):
            model = CatBoostRegressor(loss_function="RMSE", eval_metric="RMSE", random_seed=random_state, verbose=False, **params)
            model.fit(train_df[feats], train_df[target], cat_features=cat_features, eval_set=(valid_df[feats], valid_df[target]), use_best_model=True, early_stopping_rounds=80, verbose=False)
            pred_valid = model.predict(valid_df[feats]); valid_metrics = regression_metrics(valid_df[target].to_numpy(), pred_valid)
            if best is None or valid_metrics["RMSE"] < best["valid_metrics"]["RMSE"]:
                best = {"model": model, "params": params, "valid_metrics": valid_metrics}

        model = best["model"]; pred_test = model.predict(test_df[feats]); test_metrics = regression_metrics(test_df[target].to_numpy(), pred_test)
        overall_rows.append({"target": target, **best["valid_metrics"], "test_RMSE": test_metrics["RMSE"], "test_MAE": test_metrics["MAE"], "test_R2": test_metrics["R2"], **split_info})
        model.save_model(str(out / f"{target}_catboost_model.cbm"))

        pred_df = test_df.copy(); pred_df["actual"] = test_df[target].to_numpy(); pred_df["predicted"] = pred_test
        keep_cols = [c for c in ["strategy", "severity", "climate", "scenario_key", "year", "day_of_year", "actual", "predicted"] if c in pred_df.columns]
        pred_df[keep_cols].to_csv(out / f"{target}_test_predictions.csv", index=False)

        save_scatter(pred_df["actual"].to_numpy(), pred_df["predicted"].to_numpy(), f"{target}: Actual vs Predicted", figs / f"{target}_actual_vs_pred.png")
        plt.figure(figsize=(8, 6)); importances = model.get_feature_importance(); imp_df = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)
        top = imp_df.head(15).iloc[::-1]; plt.barh(top["feature"], top["importance"]); plt.xlabel("Importance"); plt.title(f"{target}: CatBoost feature importance"); plt.tight_layout(); plt.savefig(figs / f"{target}_feature_importance.png", dpi=600); plt.close()
        imp_df.to_csv(out / f"{target}_feature_importance.csv", index=False)

        for group_col in ["strategy", "severity", "climate", "scenario_key"]:
            if group_col in pred_df.columns:
                rows = []
                for g, grp in pred_df.groupby(group_col):
                    rows.append({group_col: g, **regression_metrics(grp["actual"].to_numpy(), grp["predicted"].to_numpy()), "n": len(grp)})
                pd.DataFrame(rows).sort_values("RMSE").to_csv(out / f"{target}_metrics_by_{group_col}.csv", index=False)

        if SHAP_AVAILABLE:
            shap_df = test_df[feats].sample(n=min(shap_sample, len(test_df)), random_state=random_state).reset_index(drop=True)
            explainer = shap.TreeExplainer(model); shap_values = np.array(explainer.shap_values(shap_df))
            if shap_values.ndim == 1:
                shap_values = shap_values.reshape(-1, 1)
            mean_abs = np.abs(shap_values).mean(axis=0); shap_imp = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            shap_imp.to_csv(out / f"{target}_mean_abs_shap.csv", index=False)
            plt.figure(figsize=(8, 6)); top = shap_imp.head(15).iloc[::-1]; plt.barh(top["feature"], top["mean_abs_shap"]); plt.xlabel("Mean |SHAP value|"); plt.title(f"{target}: Top SHAP features"); plt.tight_layout(); plt.savefig(figs / f"{target}_shap_bar.png", dpi=600); plt.close()
            shap.summary_plot(shap_values, shap_df, show=False, max_display=15); plt.title(f"{target}: SHAP summary"); plt.tight_layout(); plt.savefig(figs / f"{target}_shap_summary.png", dpi=600, bbox_inches="tight"); plt.close()
            shap_notes.append(f"{target}: top SHAP features = " + ", ".join(shap_imp['feature'].head(5).tolist()))

    overall_df = pd.DataFrame(overall_rows); overall_df.to_csv(out / "axis_catboost_overall_metrics.csv", index=False)
    export_surrogate_excel_report(out, overall_df); export_surrogate_pdf_report(out, overall_df, shap_notes)
    return {"metrics_csv": str(out / "axis_catboost_overall_metrics.csv"), "excel_report": str(out / "surrogate_export.xlsx"), "pdf_report": str(out / "surrogate_report.pdf"), "figures_dir": str(figs)}
