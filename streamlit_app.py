import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from hvac_v3_engine import (
    BuildingSpec, HVACConfig, HVAC_PRESETS,
    run_scenario_model, train_surrogate_models,
)

st.set_page_config(page_title="Reduced-Order-Model-ROM--Degradation", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #08101f 0%, #0b1326 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.0rem;
    max-width: 1280px;
}
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #e8ecf7;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stTabs"] {
    margin-top: 0.35rem;
}
div[data-baseweb="tab-list"] {
    gap: 0.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    padding-bottom: 0.25rem;
}
button[data-baseweb="tab"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px 12px 0 0 !important;
    color: #d7deef !important;
    padding: 0.8rem 1rem !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff5a5f !important;
    border-bottom: 2px solid #ff5a5f !important;
    background: rgba(255,255,255,0.06) !important;
}
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    background: rgba(255,255,255,0.03);
    margin-bottom: 0.85rem;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.5rem 0.7rem;
}
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow: hidden;
}
div.stButton > button {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    padding: 0.55rem 1rem !important;
    font-weight: 600 !important;
}
button[kind="primary"] {
    border-radius: 12px !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown("""
<div style="padding: 0.8rem 0 1.0rem 0;">
  <div style="font-size: 3.0rem; font-weight: 800; letter-spacing: -0.03em; color: #f4f6fb; margin-bottom: 0.35rem;">
    HVAC Research Modeling Suite v3
  </div>
  <div style="font-size: 1.02rem; color: #b9c4da; max-width: 900px;">
    Reduced-Order-Model-ROM--Degradation interface with structured scenario modeling, KPI charts, surrogate analysis, and export-ready results.
  </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# KPI chart helpers
# =========================================================
def build_kpi_summary_from_summary_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in summary_df.iterrows():
        rows.append({
            "scenario_combo_3axis": r.get("scenario_combo_3axis", ""),
            "strategy": r.get("strategy", ""),
            "severity": r.get("severity", ""),
            "climate": r.get("climate", ""),
            "Energy Consumption": r.get("Total Energy MWh", None),
            "Comfort Deviation": r.get("Mean Comfort Deviation C", None),
            "Mean Degradation Index": r.get("Mean Degradation Index", None),
            "Carbon Emissions": r.get("Total CO2 tonne", None),
        })
    return pd.DataFrame(rows)


def _make_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df[x_col].astype(str), pd.to_numeric(df[y_col], errors="coerce"), marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def _save_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path):
    fig = _make_line_chart(df, x_col, y_col, title)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)


def save_kpi_figures_from_summary(summary_csv: str | Path, output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_csv)
    kpi_df = build_kpi_summary_from_summary_csv(summary_df)
    kpi_csv = output_dir / "kpi_summary.csv"
    kpi_df.to_csv(kpi_csv, index=False)

    x_col = "scenario_combo_3axis"
    files = {}
    for y_col, filename in [
        ("Energy Consumption", "kpi_energy.png"),
        ("Comfort Deviation", "kpi_comfort.png"),
        ("Mean Degradation Index", "kpi_degradation.png"),
        ("Carbon Emissions", "kpi_carbon.png"),
    ]:
        out_png = figures_dir / filename
        _save_line_chart(kpi_df, x_col, y_col, y_col, out_png)
        files[y_col] = str(out_png)

    return {"kpi_csv": str(kpi_csv), **files}


def render_kpi_outputs_from_result(result: dict):
    summary_path = Path(result["summary_csv"])
    output_dir = summary_path.parent
    saved = save_kpi_figures_from_summary(summary_path, output_dir)

    st.markdown("### KPI charts")
    kpi_df = pd.read_csv(saved["kpi_csv"])
    st.dataframe(kpi_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(saved["Energy Consumption"], caption="Energy Consumption", use_container_width=True)
        st.image(saved["Mean Degradation Index"], caption="Mean Degradation Index", use_container_width=True)
    with c2:
        st.image(saved["Comfort Deviation"], caption="Comfort Deviation", use_container_width=True)
        st.image(saved["Carbon Emissions"], caption="Carbon Emissions", use_container_width=True)

    with open(saved["kpi_csv"], "rb") as f:
        st.download_button("Download KPI summary CSV", f.read(), file_name="kpi_summary.csv", mime="text/csv")


def render_kpi_outputs_from_folder(folder: str | Path):
    folder = Path(folder)
    summary_candidates = [
        folder / "matrix_summary.csv",
        folder / "one_axis_severity_summary.csv",
        folder / "one_axis_strategy_summary.csv",
        folder / "three_axis_summary.csv",
    ]
    summary_path = None
    for p in summary_candidates:
        if p.exists():
            summary_path = p
            break

    if summary_path is None:
        st.info("No summary CSV found yet for KPI charts.")
        return

    saved = save_kpi_figures_from_summary(summary_path, folder)
    kpi_df = pd.read_csv(saved["kpi_csv"])

    st.markdown("### KPI summary")
    st.dataframe(kpi_df, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.image(saved["Energy Consumption"], caption="Energy Consumption", use_container_width=True)
        st.image(saved["Mean Degradation Index"], caption="Mean Degradation Index", use_container_width=True)
    with c2:
        st.image(saved["Comfort Deviation"], caption="Comfort Deviation", use_container_width=True)
        st.image(saved["Carbon Emissions"], caption="Carbon Emissions", use_container_width=True)


st.markdown("<div style=\"height:0.45rem\"></div>", unsafe_allow_html=True)
# -------- Top bar / setup icon --------
title_col, setup_col = st.columns([8, 1])
with setup_col:
    with st.popover("⚙️ Setup"):
        st.markdown("### Building identity")
        building_type = st.text_input("Building type", value=st.session_state.get("building_type", "Educational / University building"))
        location = st.text_input("Location / Weather source label", value=st.session_state.get("location", "User-defined"))

        st.markdown("### Geometry")
        area_m2 = st.number_input("Conditioned area (m²)", min_value=100.0, value=float(st.session_state.get("area_m2", 5000.0)), step=100.0)
        floors = st.number_input("Floors", min_value=1, value=int(st.session_state.get("floors", 4)), step=1)
        n_spaces = st.number_input("Number of spaces", min_value=1, value=int(st.session_state.get("n_spaces", 40)), step=1)

        st.markdown("### Envelope")
        wall_u = st.number_input("Wall U-value (W/m²K)", min_value=0.1, value=float(st.session_state.get("wall_u", 0.6)), step=0.05)
        roof_u = st.number_input("Roof U-value (W/m²K)", min_value=0.1, value=float(st.session_state.get("roof_u", 0.35)), step=0.05)
        window_u = st.number_input("Window U-value (W/m²K)", min_value=0.5, value=float(st.session_state.get("window_u", 2.7)), step=0.1)
        shgc = st.number_input("SHGC", min_value=0.1, max_value=0.9, value=float(st.session_state.get("shgc", 0.35)), step=0.01)
        glazing_ratio = st.number_input("Glazing ratio", min_value=0.05, max_value=0.95, value=float(st.session_state.get("glazing_ratio", 0.30)), step=0.01)
        infiltration_ach = st.number_input("Infiltration (ACH)", min_value=0.1, value=float(st.session_state.get("infiltration_ach", 0.5)), step=0.1)

        st.markdown("### Internal loads")
        occupancy_density = st.number_input("General occupancy density (person/m²)", min_value=0.01, value=float(st.session_state.get("occupancy_density", 0.08)), step=0.01)
        lighting_w_m2 = st.number_input("Lighting power density (W/m²)", min_value=0.0, value=float(st.session_state.get("lighting_w_m2", 10.0)), step=1.0)
        equipment_w_m2 = st.number_input("Equipment power density (W/m²)", min_value=0.0, value=float(st.session_state.get("equipment_w_m2", 8.0)), step=1.0)
        sensible_w_per_person = st.number_input("Sensible heat per person (W)", min_value=40.0, value=float(st.session_state.get("sensible_w_per_person", 75.0)), step=5.0)

        st.markdown("### HVAC sizing and component")
        hvac_system_type = st.selectbox("HVAC system type", list(HVAC_PRESETS.keys()), index=list(HVAC_PRESETS.keys()).index(st.session_state.get("hvac_system_type", "Chiller_AHU")))
        airflow_m3h_m2 = st.number_input("Airflow intensity (m³/h·m²)", min_value=0.1, value=float(st.session_state.get("airflow_m3h_m2", 4.0)), step=0.1)
        cooling_w_m2 = st.number_input("Cooling design intensity (W/m²)", min_value=10.0, value=float(st.session_state.get("cooling_w_m2", 100.0)), step=5.0)
        heating_w_m2 = st.number_input("Heating design intensity (W/m²)", min_value=10.0, value=float(st.session_state.get("heating_w_m2", 55.0)), step=5.0)

        st.markdown("### Degradation parameters")
        cop_aging_rate = st.number_input("COP aging rate", min_value=0.0001, value=float(st.session_state.get("cop_aging_rate", 0.005)), step=0.001, format="%.4f")
        rf_star = st.number_input("RF* (fouling asymptote)", min_value=1e-6, value=float(st.session_state.get("rf_star", 2e-4)), format="%.6f")
        b_foul = st.number_input("Fouling growth constant B", min_value=0.001, value=float(st.session_state.get("b_foul", 0.015)), step=0.001, format="%.3f")
        dust_rate = st.number_input("Dust accumulation rate", min_value=0.1, value=float(st.session_state.get("dust_rate", 1.2)), step=0.1)
        k_clog = st.number_input("Clogging coefficient", min_value=0.1, value=float(st.session_state.get("k_clog", 6.0)), step=0.1)
        deg_trigger = st.number_input("Degradation trigger", min_value=0.1, max_value=1.5, value=float(st.session_state.get("deg_trigger", 0.55)), step=0.01)

        degradation_model = st.selectbox(
            "Degradation model",
            ["physics", "linear_ts", "exponential_ts"],
            index=["physics", "linear_ts", "exponential_ts"].index(
                st.session_state.get("degradation_model", "physics")
            ),
            format_func=lambda x: {
                "physics": "Physics-based fouling/clogging",
                "linear_ts": "Time-series linear degradation",
                "exponential_ts": "Time-series exponential degradation",
            }[x]
        )
        linear_deg_per_day = st.number_input(
            "Linear degradation slope per day",
            min_value=0.000001,
            value=float(st.session_state.get("linear_deg_per_day", 0.00012)),
            step=0.00001,
            format="%.6f"
        )
        exp_deg_rate_per_day = st.number_input(
            "Exponential degradation rate per day",
            min_value=0.000001,
            value=float(st.session_state.get("exp_deg_rate_per_day", 0.00018)),
            step=0.00001,
            format="%.6f"
        )
        include_baseline_layer = st.checkbox(
            "Export baseline no-degradation layer",
            value=st.session_state.get("include_baseline_layer", True)
        )

        st.markdown("### Occupancy input form")
        use_zone_occ = st.checkbox("Use zone-specific occupancy input", value=st.session_state.get("use_zone_occ", False))
        default_zone_df = pd.DataFrame(
            st.session_state.get(
                "zone_df",
                [
                    {"zone_name": "Lecture_01", "zone_type": "Lecture", "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
                    {"zone_name": "Office_01", "zone_type": "Office", "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
                    {"zone_name": "Lab_01", "zone_type": "Lab", "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
                    {"zone_name": "Corridor", "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
                    {"zone_name": "Service_01", "zone_type": "Service", "area_m2": 80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
                ],
            )
        )
        zone_df = st.data_editor(default_zone_df, num_rows="dynamic", use_container_width=True) if use_zone_occ else None

        st.markdown("### Global simulation")
        years = st.number_input("Simulation years", min_value=1, value=int(st.session_state.get("years", 5)), step=1)
        random_state = st.number_input("Random seed", min_value=1, value=int(st.session_state.get("random_state", 42)), step=1)

        if st.button("Save setup"):
            for k, v in {
                "building_type": building_type, "location": location, "area_m2": area_m2, "floors": floors, "n_spaces": n_spaces,
                "wall_u": wall_u, "roof_u": roof_u, "window_u": window_u, "shgc": shgc, "glazing_ratio": glazing_ratio,
                "infiltration_ach": infiltration_ach, "occupancy_density": occupancy_density, "lighting_w_m2": lighting_w_m2,
                "equipment_w_m2": equipment_w_m2, "sensible_w_per_person": sensible_w_per_person, "hvac_system_type": hvac_system_type,
                "airflow_m3h_m2": airflow_m3h_m2, "cooling_w_m2": cooling_w_m2, "heating_w_m2": heating_w_m2,
                "cop_aging_rate": cop_aging_rate, "rf_star": rf_star, "b_foul": b_foul, "dust_rate": dust_rate,
                "k_clog": k_clog, "deg_trigger": deg_trigger,
                "degradation_model": degradation_model, "linear_deg_per_day": linear_deg_per_day,
                "exp_deg_rate_per_day": exp_deg_rate_per_day, "include_baseline_layer": include_baseline_layer,
                "years": years, "random_state": random_state, "use_zone_occ": use_zone_occ,
            }.items():
                st.session_state[k] = v
            if use_zone_occ:
                st.session_state["zone_df"] = zone_df.to_dict(orient="records")
            st.success("Setup saved")

# pull config from session with defaults
building_type = st.session_state.get("building_type", "Educational / University building")
location = st.session_state.get("location", "User-defined")
area_m2 = float(st.session_state.get("area_m2", 5000.0))
floors = int(st.session_state.get("floors", 4))
n_spaces = int(st.session_state.get("n_spaces", 40))
wall_u = float(st.session_state.get("wall_u", 0.6))
roof_u = float(st.session_state.get("roof_u", 0.35))
window_u = float(st.session_state.get("window_u", 2.7))
shgc = float(st.session_state.get("shgc", 0.35))
glazing_ratio = float(st.session_state.get("glazing_ratio", 0.30))
infiltration_ach = float(st.session_state.get("infiltration_ach", 0.5))
occupancy_density = float(st.session_state.get("occupancy_density", 0.08))
lighting_w_m2 = float(st.session_state.get("lighting_w_m2", 10.0))
equipment_w_m2 = float(st.session_state.get("equipment_w_m2", 8.0))
sensible_w_per_person = float(st.session_state.get("sensible_w_per_person", 75.0))
hvac_system_type = st.session_state.get("hvac_system_type", "Chiller_AHU")
airflow_m3h_m2 = float(st.session_state.get("airflow_m3h_m2", 4.0))
cooling_w_m2 = float(st.session_state.get("cooling_w_m2", 100.0))
heating_w_m2 = float(st.session_state.get("heating_w_m2", 55.0))
cop_aging_rate = float(st.session_state.get("cop_aging_rate", 0.005))
rf_star = float(st.session_state.get("rf_star", 2e-4))
b_foul = float(st.session_state.get("b_foul", 0.015))
dust_rate = float(st.session_state.get("dust_rate", 1.2))
k_clog = float(st.session_state.get("k_clog", 6.0))
deg_trigger = float(st.session_state.get("deg_trigger", 0.55))
degradation_model = st.session_state.get("degradation_model", "physics")
linear_deg_per_day = float(st.session_state.get("linear_deg_per_day", 0.00012))
exp_deg_rate_per_day = float(st.session_state.get("exp_deg_rate_per_day", 0.00018))
include_baseline_layer = bool(st.session_state.get("include_baseline_layer", True))
years = int(st.session_state.get("years", 5))
random_state = int(st.session_state.get("random_state", 42))
use_zone_occ = bool(st.session_state.get("use_zone_occ", False))
zone_df = pd.DataFrame(st.session_state.get("zone_df", [])) if use_zone_occ and st.session_state.get("zone_df") else None

bldg = BuildingSpec(
    building_type=building_type, location=location, conditioned_area_m2=area_m2, floors=floors, n_spaces=n_spaces,
    occupancy_density_p_m2=occupancy_density, lighting_w_m2=lighting_w_m2, equipment_w_m2=equipment_w_m2,
    airflow_m3h_m2=airflow_m3h_m2, infiltration_ach=infiltration_ach, sensible_w_per_person=sensible_w_per_person,
    cooling_intensity_w_m2=cooling_w_m2, heating_intensity_w_m2=heating_w_m2,
    wall_u=wall_u, roof_u=roof_u, window_u=window_u, shgc=shgc, glazing_ratio=glazing_ratio,
)
cfg = HVACConfig(
    years=years, hvac_system_type=hvac_system_type, COP_AGING_RATE=cop_aging_rate, RF_STAR=rf_star,
    B_FOUL=b_foul, DUST_RATE=dust_rate, K_CLOG=k_clog, DEG_TRIGGER=deg_trigger,
    degradation_model=degradation_model, LINEAR_DEG_PER_DAY=linear_deg_per_day, EXP_DEG_RATE_PER_DAY=exp_deg_rate_per_day
)

st.markdown("<div style=\"margin-top:0.35rem; margin-bottom:0.55rem; color:#9fb0cf; font-size:0.95rem;\">Use the tabs below to move between modeling, KPI review, surrogate training, and export workflows.</div>", unsafe_allow_html=True)
tabs = st.tabs([
    "Scenario Modeling",
    "KPI Charts",
    "Surrogate Train / Predict",
    "Exports and Results",
    "Guide"
])

with tabs[0]:
    st.subheader("Run modeling systems")
    st.markdown("<div style=\"height:0.2rem\"></div>", unsafe_allow_html=True)
    axis_mode = st.selectbox("Select modeling level", ["one_severity", "one_strategy", "two_axis", "three_axis"], format_func=lambda x: {
        "one_severity": "1) One-axis severity",
        "one_strategy": "2) One-axis strategy",
        "two_axis": "3) Two-axis severity/strategy",
        "three_axis": "4) Three-axis severity/strategy/weather",
    }[x])
    fixed_strategy = st.selectbox("Fixed strategy (for one-axis severity)", ["S0","S1","S2","S3"], index=3)
    fixed_severity = st.selectbox("Fixed severity (for one-axis strategy)", ["Mild","Moderate","Severe","High"], index=1)
    fixed_climate = st.selectbox("Fixed climate (for one-/two-axis)", ["C0_Baseline","C1_Warm","C2_Heatwave","C3_FutureHot"], index=0)
    weather_mode = st.selectbox("Weather mode", ["synthetic", "epw"], index=0)
    epw_path = st.text_input("EPW file path (optional)", "")
    out_dir = st.text_input("Output folder", "v3_run")

    if st.button("Run selected model"):
        result = run_scenario_model(
            output_dir=out_dir,
            axis_mode=axis_mode,
            bldg=bldg,
            cfg=cfg,
            weather_mode=weather_mode,
            epw_path=epw_path if weather_mode == "epw" and epw_path.strip() else None,
            fixed_strategy=fixed_strategy,
            fixed_severity=fixed_severity,
            fixed_climate=fixed_climate,
            zone_df=zone_df,
            random_state=random_state,
            include_baseline_layer=include_baseline_layer,
            degradation_model=degradation_model,
        )
        st.session_state["last_result"] = result
        st.success("Model run finished")
        st.json(result)
        summary_path = Path(result["summary_csv"])
        if summary_path.exists():
            st.dataframe(pd.read_csv(summary_path), use_container_width=True)

with tabs[1]:
    st.subheader("KPI charts")
    last_result = st.session_state.get("last_result")
    if last_result:
        render_kpi_outputs_from_result(last_result)
    else:
        folder_for_kpis = st.text_input("Result folder for KPI charts", "v3_run")
        render_kpi_outputs_from_folder(folder_for_kpis)

with tabs[2]:
    st.subheader("Train surrogate and predict")
    st.markdown("<div style=\"height:0.2rem\"></div>", unsafe_allow_html=True)
    dataset_path = st.text_input("Input dataset CSV", "v3_run/matrix_ml_dataset.csv")
    surrogate_out = st.text_input("Surrogate output folder", "v3_surrogate")
    n_iter_search = st.number_input("CatBoost search iterations", min_value=2, value=6, step=1)
    shap_sample = st.number_input("SHAP sample size", min_value=100, value=1000, step=100)
    if st.button("Train CatBoost surrogate"):
        result = train_surrogate_models(dataset_path, surrogate_out, int(n_iter_search), int(shap_sample), int(random_state))
        st.success("Surrogate training finished")
        st.json(result)
        p = Path(result["metrics_csv"])
        if p.exists():
            st.dataframe(pd.read_csv(p), use_container_width=True)

with tabs[3]:
    st.subheader("Inspect and export results")
    st.markdown("<div style=\"height:0.2rem\"></div>", unsafe_allow_html=True)
    target_folder = st.text_input("Result folder", "v3_run")
    p = Path(target_folder)
    if p.exists():
        render_kpi_outputs_from_folder(p)

        files = sorted([x.name for x in p.iterdir() if x.is_file()])
        st.write("Files:")
        st.write(files)
        csvs = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".csv"]
        for csvf in csvs[:8]:
            st.markdown(f"**{csvf.name}**")
            try:
                st.dataframe(pd.read_csv(csvf).head(40), use_container_width=True)
            except Exception:
                pass

        baseline_summary = p / "baseline_no_degradation_summary.csv"
        baseline_daily = p / "baseline_no_degradation_daily.csv"
        if baseline_summary.exists():
            st.markdown("### Baseline no-degradation summary")
            st.dataframe(pd.read_csv(baseline_summary), use_container_width=True)
        if baseline_daily.exists():
            st.markdown("### Baseline no-degradation daily preview")
            st.dataframe(pd.read_csv(baseline_daily).head(40), use_container_width=True)
        for special in ["results_export.xlsx", "results_report.pdf", "surrogate_export.xlsx", "surrogate_report.pdf", "kpi_summary.csv"]:
            fp = p / special
            if fp.exists():
                with open(fp, "rb") as f:
                    st.download_button(f"Download {special}", f.read(), file_name=special)

        zones_csv_candidates = sorted(list(p.glob("*zones*.csv")))
        if zones_csv_candidates:
            st.markdown("### Zones analysis tables")
            for zf in zones_csv_candidates[:6]:
                try:
                    st.markdown(f"**{zf.name}**")
                    zdf = pd.read_csv(zf)
                    st.dataframe(zdf.head(60), use_container_width=True)
                    with open(zf, "rb") as f:
                        st.download_button(f"Download {zf.name}", f.read(), file_name=zf.name, key=f"dl_{zf.name}")
                except Exception:
                    pass

        figs_dir = p / "figures"
        if figs_dir.exists():
            img_files = sorted(list(figs_dir.glob("*.png")))[:20]
            cols = st.columns(2)
            for i, img in enumerate(img_files):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("Run a model first or type an existing folder path.")

with tabs[4]:
    st.markdown("""
### Zone-specific occupancy behavior
The zone table now supports distinct densities and distinct term/break/summer schedule factors for lecture rooms, offices, labs, corridors, and service zones. These are aggregated dynamically to the equivalent-building daily occupancy used by the reduced-order load model.

### KPI charts added
This interface now adds direct charts for:
- Energy Consumption
- Comfort Deviation
- Mean Degradation Index
- Carbon Emissions

### How `matrix_ml_dataset.csv` is generated
When you run a scenario model, the app automatically exports a **generic ML dataset alias**:
- one-axis: `matrix_ml_dataset.csv` is created as an alias to the one-axis dataset
- two-axis: `matrix_ml_dataset.csv` is the main two-axis dataset
- three-axis: both `three_axis_ml_dataset.csv` and a generic `matrix_ml_dataset.csv` alias are created

### What this version adds
- UI style aligned with Reduced-Order-Model-ROM--Degradation
- KPI charts tab
- KPI summary CSV export
- zone-specific occupancy input form
- HVAC system type selector
- degradation parameter setup
- integrated CatBoost surrogate training and prediction

### Suggested workflow
1. Configure building + HVAC + degradation + occupancy
2. Run one-/two-/three-axis scenario model
3. Review KPI charts
4. Use exported `matrix_ml_dataset.csv` for surrogate training
5. Export Excel/PDF and figures for reporting
""")
