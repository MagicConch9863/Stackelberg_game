"""Interactive result viewer for the C-A four-follower scenarios.

Run with: streamlit run ca_dashboard.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
CSV_FOLDER = ROOT / "csv_results"
SCENARIO_FILES = {
    "S1 – predefined operation": "s1_timeseries.csv",
    "S2 – fixed tariff": "s2_timeseries.csv",
    "S3 – Stackelberg": "s3_timeseries.csv",
    "S4 – centralized benchmark": "s4_timeseries.csv",
}
BUSES = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")
BUS_CONFIG = {
    "Bus 1": {"BESS capacity [kWh]": 5.0, "BESS power [kW]": 1.0,
              "EV": "Yes", "EV capacity [kWh]": 40.0,
              "EV charge/discharge [kW]": "+3.6 / -1.0", "Trip [kWh]": 4.0},
    "Bus 2": {"BESS capacity [kWh]": 4.0, "BESS power [kW]": 0.8,
              "EV": "No", "EV capacity [kWh]": "–",
              "EV charge/discharge [kW]": "–", "Trip [kWh]": "–"},
    "Bus 3": {"BESS capacity [kWh]": 6.0, "BESS power [kW]": 1.2,
              "EV": "No", "EV capacity [kWh]": "–",
              "EV charge/discharge [kW]": "–", "Trip [kWh]": "–"},
    "Bus 4": {"BESS capacity [kWh]": 6.0, "BESS power [kW]": 1.2,
              "EV": "Yes", "EV capacity [kWh]": 40.0,
              "EV charge/discharge [kW]": "+3.6 / -1.2", "Trip [kWh]": 5.0},
}
VARIABLES = {
    "Net exchange": ("net_kw", "kW"),
    "Import": ("import_kw", "kW"),
    "Export": ("export_kw", "kW"),
    "BESS power": ("bess_kw", "kW"),
    "EV power": ("ev_kw", "kW"),
    "BESS energy": ("bess_energy_kwh", "kWh"),
    "EV energy": ("ev_energy_kwh", "kWh"),
}
COLORS = {"Bus 1": "#1f77b4", "Bus 2": "#ff7f0e",
          "Bus 3": "#2ca02c", "Bus 4": "#d62728"}
DISPLAY_DECIMALS = 5
DISPLAY_ZERO_TOL = 1e-7


def bus_key(bus: str) -> str:
    return bus.lower().replace(" ", "_")


@st.cache_data(show_spinner=False)
def read_csv(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path)


def available_scenarios():
    return {label: CSV_FOLDER/name for label, name in SCENARIO_FILES.items()
            if (CSV_FOLDER/name).is_file()}


def validate(frame: pd.DataFrame):
    if "time_h" not in frame or "aggregate_net_kw" not in frame:
        st.error("CSV must contain time_h and aggregate_net_kw.")
        st.stop()


def display_array(values) -> np.ndarray:
    """Remove solver noise for display without changing the saved optimization."""
    clean = np.asarray(values, dtype=float).copy()
    clean[np.abs(clean) < DISPLAY_ZERO_TOL] = 0.0
    clean = np.round(clean, DISPLAY_DECIMALS)
    # If the remaining peak-to-valley change is only numerical noise, draw a line.
    if clean.size and np.ptp(clean) <= DISPLAY_ZERO_TOL:
        clean[:] = round(float(np.mean(clean)), DISPLAY_DECIMALS)
    return clean


def display_number(value: float) -> float:
    value = float(value)
    if abs(value) < DISPLAY_ZERO_TOL:
        return 0.0
    return round(value, DISPLAY_DECIMALS)


def metric_values(net: np.ndarray, dt=.25):
    net = display_array(net)
    return {key: display_number(value) for key, value in {
        "Peak [kW]": float(np.max(net)),
        "Valley [kW]": float(np.min(net)),
        "Range [kW]": float(np.ptp(net)),
        "Max ramp [kW]": float(np.max(np.abs(np.diff(net)))) if len(net)>1 else 0.0,
        "Import [kWh]": float(np.maximum(net, 0).sum()*dt),
        "Export [kWh]": float(np.maximum(-net, 0).sum()*dt),
    }.items()}


def aggregate_figure(frame, title):
    has_price = "pa_eur_per_kwh" in frame
    rows = 2 if has_price else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        subplot_titles=("Aggregate net exchange", "Common prices") if has_price
                        else ("Aggregate net exchange",))
    fig.add_trace(go.Scatter(x=frame.time_h, y=display_array(frame.aggregate_net_kw),
                             name="Aggregate net", line={"color": "black", "width": 2}), row=1, col=1)
    if has_price:
        fig.add_trace(go.Scatter(x=frame.time_h, y=display_array(frame.pa_eur_per_kwh),
                                 name="PA", line={"color": "#1f77b4"}), row=2, col=1)
        if "pb_eur_per_kwh" in frame:
            fig.add_trace(go.Scatter(x=frame.time_h, y=display_array(frame.pb_eur_per_kwh),
                                     name="PB", line={"color": "#ff7f0e"}), row=2, col=1)
        if "total_buy_price_eur_per_kwh" in frame:
            fig.add_trace(go.Scatter(x=frame.time_h, y=display_array(frame.total_buy_price_eur_per_kwh),
                                     name="PA + PB", line={"color": "#2ca02c"}), row=2, col=1)
        fig.update_yaxes(title_text="EUR/kWh", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_xaxes(title_text="Time [h]", row=rows, col=1)
    fig.update_yaxes(tickformat=".5~f")
    fig.update_layout(title=title, height=650 if has_price else 430,
                      hovermode="x unified", legend={"orientation": "h"})
    return fig


def selected_curves(frame, buses, variable_names, separate):
    traces = []
    for bus in buses:
        prefix = bus_key(bus)
        for label in variable_names:
            if label.startswith("EV") and BUS_CONFIG[bus]["EV"] == "No":
                continue
            suffix, unit = VARIABLES[label]
            column = f"{prefix}_{suffix}"
            if column in frame:
                traces.append((bus, label, unit, column))
    if not traces:
        st.warning("The selected variables are not present for these buses.")
        return None
    rows = len(buses) if separate else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        subplot_titles=buses if separate else None)
    row_for = {bus: i+1 for i, bus in enumerate(buses)}
    dashes = {name: dash for name, dash in zip(VARIABLES, ("solid", "dash", "dot", "solid", "dashdot", "solid", "dot"))}
    for bus, label, unit, column in traces:
        row = row_for[bus] if separate else 1
        values = display_array(frame[column])
        if np.all(np.abs(values) <= DISPLAY_ZERO_TOL):
            continue
        fig.add_trace(go.Scatter(x=frame.time_h, y=values,
                                 name=f"{bus} – {label}",
                                 line={"color": COLORS[bus], "dash": dashes[label]}), row=row, col=1)
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="Time [h]", row=rows, col=1)
    if not fig.data:
        st.info("All selected installed-device curves are zero in this scenario.")
        return None
    fig.update_yaxes(tickformat=".5~f")
    fig.update_layout(height=max(430, 260*rows), hovermode="x unified",
                      legend={"orientation": "h"})
    return fig


def bus_metrics(frame, bus):
    prefix = bus_key(bus)
    column = f"{prefix}_net_kw"
    if column not in frame:
        return None
    values = metric_values(frame[column].to_numpy(float))
    values["Final BESS energy [kWh]"] = display_number(frame[f"{prefix}_bess_energy_kwh"].iloc[-1])
    ev_col = f"{prefix}_ev_energy_kwh"
    if ev_col in frame:
        values["Final EV energy [kWh]"] = display_number(frame[ev_col].iloc[-1])
    return values


def comparison_page():
    files = available_scenarios()
    if len(files) < 2:
        st.info("Run at least two scenarios to display a comparison.")
        return
    rows = []
    fig = go.Figure()
    scenario_colors = ("#555555", "#1f77b4", "#ff7f0e", "#2ca02c")
    for (label, path), color in zip(files.items(), scenario_colors):
        frame = read_csv(str(path), path.stat().st_mtime); validate(frame)
        short = label.split(" ")[0]
        net = display_array(frame.aggregate_net_kw)
        rows.append({"Scenario": short, **metric_values(net)})
        fig.add_trace(go.Scatter(x=frame.time_h, y=net, name=short,
                                 line={"color": color, "width": 2}))
    fig.update_layout(title="Aggregate net-exchange comparison", xaxis_title="Time [h]",
                      yaxis_title="kW", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame(rows).set_index("Scenario").round(DISPLAY_DECIMALS),
                 width="stretch")


def main():
    st.set_page_config(page_title="C-A Scenario Dashboard", layout="wide")
    st.title("C-A Four-Follower Scenario Dashboard")
    st.caption(f"Default data folder: {CSV_FOLDER}")

    files = available_scenarios()
    uploaded = st.sidebar.file_uploader("Optional: upload a scenario CSV", type="csv")
    if uploaded is not None:
        frame = pd.read_csv(uploaded); scenario_label = uploaded.name
    else:
        if not files:
            st.error(f"No scenario CSV files found in {CSV_FOLDER}. Run S1–S4 first.")
            st.stop()
        scenario_label = st.sidebar.selectbox("Scenario", list(files))
        path = files[scenario_label]
        frame = read_csv(str(path), path.stat().st_mtime)
    validate(frame)

    page = st.sidebar.radio("View", ("Scenario overview", "Bus explorer", "Scenario comparison", "Raw data"))
    if page == "Scenario overview":
        values = metric_values(frame.aggregate_net_kw.to_numpy(float))
        columns = st.columns(6)
        for column, (label, value) in zip(columns, values.items()):
            column.metric(label, f"{display_number(value):.{DISPLAY_DECIMALS}f}")
        st.plotly_chart(aggregate_figure(frame, scenario_label), width="stretch")

    elif page == "Bus explorer":
        buses = st.sidebar.multiselect("Followers", BUSES, default=["Bus 1"])
        variables = st.sidebar.multiselect("Variables", list(VARIABLES),
                                           default=["Net exchange", "BESS power", "EV power"])
        separate = st.sidebar.toggle("One subplot per Bus", value=False)
        if not buses:
            st.warning("Select at least one Bus."); return
        fig = selected_curves(frame, buses, variables, separate)
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
        st.subheader("Selected follower details")
        for bus in buses:
            with st.expander(bus, expanded=len(buses)==1):
                left, right = st.columns(2)
                left.dataframe(pd.DataFrame([BUS_CONFIG[bus]]).T.rename(columns={0: "Value"}),
                                   width="stretch")
                metrics = bus_metrics(frame, bus)
                if metrics:
                    right.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "Value"}),
                                    width="stretch")

    elif page == "Scenario comparison":
        comparison_page()

    else:
        shown = frame.copy()
        numeric = shown.select_dtypes(include=np.number).columns
        shown[numeric] = shown[numeric].apply(display_array)
        st.dataframe(shown, width="stretch", height=600)
        st.download_button("Download displayed CSV", frame.to_csv(index=False),
                           file_name="displayed_scenario.csv", mime="text/csv")


if __name__ == "__main__":
    main()
