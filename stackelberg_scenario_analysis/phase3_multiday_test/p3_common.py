"""Shared data loading, configuration, CSV output and plotting for Phase 3.

Phase 3 repeats one validated SimBench load day while PA and PV change from
2026-07-01 through 2026-07-31.  Every scenario is solved as 31 independent
96-step daily problems; device energy is reset at the beginning of each day.
"""
from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simbench as sb

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.facecolor": "white",
})

T, DT = 96, 0.25
PHASE3_CODE_VERSION = "2026-08-16-v3-dst-safe"
BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")

SIMBENCH_MAPPING = {
    "Bus 1": {"name": "BUS6", "index": 6},
    "Bus 2": {"name": "BUS8", "index": 8},
    "Bus 3": {"name": "BUS14", "index": 14},
    "Bus 4": {"name": "BUS16", "index": 17},
}

BUS_CONFIG = {
    "Bus 1": {
        "bess_kwh": 8.0, "bess_kw": 2.5,
        "bess_e_min": 0.8, "bess_e_max": 8.0, "bess_e_init": 4.0,
        "ev_kwh": 40.0, "ev_kw_ch": 3.6, "ev_kw_dis": 1.0,
        "ev_trip_kwh": 4.0,
    },
    "Bus 2": {
        "bess_kwh": 5.0, "bess_kw": 1.5,
        "bess_e_min": 0.5, "bess_e_max": 5.0, "bess_e_init": 2.5,
    },
    "Bus 3": {
        "bess_kwh": 8.0, "bess_kw": 2.5,
        "bess_e_min": 0.8, "bess_e_max": 8.0, "bess_e_init": 4.0,
    },
    "Bus 4": {
        "bess_kwh": 8.0, "bess_kw": 2.5,
        "bess_e_min": 0.8, "bess_e_max": 8.0, "bess_e_init": 4.0,
        "ev_kwh": 40.0, "ev_kw_ch": 3.6, "ev_kw_dis": 1.2,
        "ev_trip_kwh": 5.0,
    },
}

COLORS = {
    "Bus 1": "#4C78A8", "Bus 2": "#F2A65A",
    "Bus 3": "#72A98F", "Bus 4": "#D98282",
}


def resolve_file(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_file():
        return path.resolve()
    here = Path(__file__).resolve()
    # Windows Explorer often hides the final extension.  If a user types
    # "name.csv" while renaming, the physical filename may become
    # "name.csv.csv".  Search both forms and also the Phase-3 input_data
    # subfolder used by this project.
    names = [path]
    if path.suffix.casefold() == ".csv":
        names.append(Path(f"{path}.csv"))
    folders = []
    for folder in (Path.cwd(), here.parent, *here.parents):
        folders.extend((folder, folder / "input_data"))
    tried = []
    for folder in dict.fromkeys(folders):
        for name in names:
            candidate = folder / name
            tried.append(str(candidate))
            if candidate.is_file():
                if candidate.name.casefold().endswith(".csv.csv"):
                    print(f"NOTICE: using duplicate-extension file {candidate.name}. "
                          "It may safely be renamed to a single .csv later.")
                return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find input file: {filename}\n"
        f"Place it in {here.parent / 'input_data'} or change the setting.\n"
        f"Also checked the Windows duplicate-extension form: {path.name}.csv"
    )


def output_folders() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    csv_folder = root / "csv_results"
    figure_folder = root / "figures"
    csv_folder.mkdir(parents=True, exist_ok=True)
    figure_folder.mkdir(parents=True, exist_ok=True)
    return csv_folder, figure_folder


def scenario_figure_folder(scenario: str, style: str | None = None) -> Path:
    """Return figures/s1[/line|bar] and create it when needed."""
    _, figure_root = output_folders()
    folder = figure_root / str(scenario).strip().lower()
    if style is not None:
        folder = folder / str(style).strip().lower()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def experiment_dates(settings: dict) -> pd.DatetimeIndex:
    start = pd.Timestamp(settings["START_DATE"]).normalize()
    end = pd.Timestamp(settings["END_DATE"]).normalize()
    if end < start:
        raise ValueError("END_DATE must not be before START_DATE")
    dates = pd.date_range(start, end, freq="D")
    maximum = settings.get("MAX_DAYS")
    return dates if maximum is None else dates[:int(maximum)]


def expected_quarter_hours(settings: dict) -> pd.DatetimeIndex:
    dates = experiment_dates(settings)
    if len(dates) == 0:
        raise ValueError("The experiment date range is empty")
    return pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=1),
                         freq="15min", inclusive="left")


def _find_column(frame: pd.DataFrame, wanted: str) -> str:
    exact = {str(column).strip().casefold(): column for column in frame.columns}
    key = wanted.strip().casefold()
    if key in exact:
        return exact[key]
    candidates = [column for column in frame.columns
                  if key in str(column).strip().casefold()]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"Cannot uniquely find column containing {wanted!r}. "
        f"Available columns: {list(frame.columns)}"
    )


def _numeric_text(series: pd.Series) -> pd.Series:
    """Convert English or German formatted number strings to float."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().strip('"').replace("\u00a0", "").replace(" ", "")
        if not text:
            return np.nan
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            return float(text)
        except ValueError:
            match = re.search(r"[-+]?\d+(?:[.,]\d+)?", text)
            return float(match.group(0).replace(",", ".")) if match else np.nan
    return series.map(convert).astype(float)


def _parse_interval_start(series: pd.Series) -> pd.Series:
    left = (series.astype(str)
            .str.split(r"\s+-\s+", n=1, regex=True).str[0]
            .str.strip())
    # ENTSO-E/SMARD may append CET, CEST or another timezone label in
    # parentheses.  Phase 3 uses local market clock time, therefore remove
    # any final parenthesized timezone label before parsing.  Using the broad
    # pattern also survives spelling/capitalization changes in future files.
    cleaned = left.str.replace(r"\s*\([^)]*\)\s*$", "", regex=True).str.strip()
    parsed = pd.to_datetime(
        cleaned, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    # Fallback keeps the reader compatible with exports that omit seconds or
    # use an English month name.
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            cleaned.loc[missing], dayfirst=True, errors="coerce")
    if parsed.isna().any():
        samples = left.loc[parsed.isna()].head(3).tolist()
        raise ValueError(f"Cannot parse interval start times, examples: {samples}")
    return parsed


def _coarse_interval_date(series: pd.Series) -> pd.Series:
    """Extract only DD/MM/YYYY without parsing DST-sensitive clock text."""
    left = (series.astype(str)
            .str.split(r"\s+-\s+", n=1, regex=True).str[0]
            .str.strip())
    parts = left.str.extract(
        r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>\d{4})")
    return pd.to_datetime(
        {
            "year": pd.to_numeric(parts["year"], errors="coerce"),
            "month": pd.to_numeric(parts["month"], errors="coerce"),
            "day": pd.to_numeric(parts["day"], errors="coerce"),
        },
        errors="coerce",
    )


def read_pa_month(settings: dict) -> pd.Series:
    """Read Sequence 1 from the full-year price CSV and select July 2026."""
    path = resolve_file(settings["PA_FILE"])
    try:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    mtu_col = _find_column(frame, settings["PA_TIME_COLUMN"])
    sequence_col = _find_column(frame, settings["PA_SEQUENCE_COLUMN"])
    price_col = _find_column(frame, settings["PA_PRICE_COLUMN"])

    sequence = frame[sequence_col].astype(str).str.strip().str.casefold()
    wanted = str(settings["PA_SEQUENCE"]).strip().casefold()
    selected = frame.loc[sequence == wanted, [mtu_col, price_col]].copy()
    if selected.empty:
        raise ValueError(f"No rows with {settings['PA_SEQUENCE']!r} in {path.name}")

    # The source CSV contains a full year.  Filter by its date token before
    # parsing the clock time.  Consequently March/October DST transition rows
    # can never break a July-only simulation, even if their local clock labels
    # are unusual or duplicated.
    dates = experiment_dates(settings)
    target_start = dates[0]
    target_stop = dates[-1] + pd.Timedelta(days=1)
    coarse = _coarse_interval_date(selected[mtu_col])
    if coarse.notna().any():
        keep = coarse.ge(target_start) & coarse.lt(target_stop)
        selected = selected.loc[keep].copy()
    if selected.empty:
        raise ValueError(
            f"No {settings['PA_SEQUENCE']!r} price rows found from "
            f"{target_start.date()} through {(target_stop - pd.Timedelta(days=1)).date()} "
            f"in {path.name}"
        )
    selected["timestamp"] = _parse_interval_start(selected[mtu_col])
    selected["pa_eur_per_mwh"] = _numeric_text(selected[price_col])
    selected = selected.dropna(subset=["pa_eur_per_mwh"])

    expected = expected_quarter_hours(settings)
    series = selected.groupby("timestamp")["pa_eur_per_mwh"].mean().sort_index()
    series = series.reindex(expected)
    missing = int(series.isna().sum())
    if missing:
        examples = series.index[series.isna()][:5].astype(str).tolist()
        raise ValueError(
            f"Price data misses {missing} required quarter-hours; examples: {examples}"
        )
    series = series / 1000.0
    series.name = "pa_eur_per_kwh"
    print(f"Phase 3 reader: {PHASE3_CODE_VERSION}")
    print(f"PA: {path.name}, {len(series)} rows, Sequence 1, "
          f"range={series.min():.5f}..{series.max():.5f} EUR/kWh")
    return series


def read_pv_month(settings: dict) -> pd.Series:
    """Read SMARD Photovoltaics and normalize once across the entire month."""
    path = resolve_file(settings["PV_FILE"])
    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            frame = pd.read_csv(path, sep=";", engine="python", dtype=str,
                                encoding=encoding)
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise last_error
    frame.columns = [str(column).strip().lstrip("\ufeff") for column in frame.columns]
    if len(frame.columns) < 3:
        raise ValueError(
            "The SMARD file was not parsed as a semicolon CSV. Use the original "
            "downloaded file, not an Excel-resaved incorrectly split copy."
        )
    time_col = _find_column(frame, settings["PV_TIME_COLUMN"])
    pv_col = _find_column(frame, settings["PV_COLUMN_KEYWORD"])
    timestamp = pd.to_datetime(frame[time_col].astype(str).str.strip(),
                               dayfirst=True, errors="coerce")
    if timestamp.isna().any():
        samples = frame.loc[timestamp.isna(), time_col].head(3).tolist()
        raise ValueError(f"Cannot parse SMARD start dates, examples: {samples}")
    values = _numeric_text(frame[pv_col]).clip(lower=0.0)
    raw = pd.DataFrame({"timestamp": timestamp, "pv_raw_mwh": values})
    raw = raw.dropna().groupby("timestamp")["pv_raw_mwh"].mean().sort_index()

    expected = expected_quarter_hours(settings)
    raw = raw.reindex(expected)
    missing = int(raw.isna().sum())
    if missing:
        examples = raw.index[raw.isna()][:5].astype(str).tolist()
        raise ValueError(
            f"PV data misses {missing} required quarter-hours; examples: {examples}"
        )
    reference = float(raw.max())
    if reference <= 0:
        raise ValueError("SMARD Photovoltaics contains no positive value")
    pv_pu = (raw / reference).clip(0.0, 1.0)
    pv_pu.name = "pv_pu"
    print(f"PV: {path.name}, column={pv_col!r}, {len(pv_pu)} rows, "
          f"monthly normalization reference={reference:.6g} MWh")
    return pv_pu


def load_fixed_simbench_day(settings: dict) -> dict[str, np.ndarray]:
    """Load one SimBench day once; this same load is repeated for all 31 days."""
    net = sb.get_simbench_net(settings["SIMBENCH_CODE"])
    profiles = sb.get_absolute_profiles_from_relative_profiles(
        net=net, element="load", multiplying_column="p_mw", time_as_index=False)
    start = int(settings["LOAD_DAY_INDEX"]) * T
    day = profiles.iloc[start:start + T]
    if len(day) != T:
        raise ValueError("LOAD_DAY_INDEX exceeds the SimBench time series")
    column_map = {str(column): column for column in day.columns}
    loads = {}
    for bus in BUS_ORDER:
        mapping = SIMBENCH_MAPPING[bus]
        sb_index = mapping["index"]
        actual_name = str(net.bus.at[sb_index, "name"])
        load_indices = net.load.index[net.load["bus"] == sb_index].tolist()
        columns = [column_map[str(i)] for i in load_indices if str(i) in column_map]
        if not columns:
            raise RuntimeError(
                f"{bus}: no load profile at {actual_name} (index {sb_index})"
            )
        loads[bus] = day[columns].sum(axis=1).to_numpy(float) * 1000.0
        print(f"{bus} <- {actual_name} (index {sb_index}), loads {load_indices}")
    return loads


def day_inputs(date: pd.Timestamp, loads: dict[str, np.ndarray],
               pa_month: pd.Series | None, pv_month: pd.Series,
               settings: dict):
    index = pd.date_range(date, date + pd.Timedelta(days=1),
                          freq="15min", inclusive="left")
    pv_shape = pv_month.reindex(index).to_numpy(float)
    pv = {
        bus: pv_shape * float(settings["PV_CAPACITY_KW"].get(bus, 0.0))
        for bus in BUS_ORDER
    }
    pa = None if pa_month is None else pa_month.reindex(index).to_numpy(float)
    return loads, pv, pa


def trip_vector(bus: str) -> np.ndarray:
    trip = np.zeros(T)
    if "ev_kwh" in BUS_CONFIG[bus]:
        slots = np.r_[np.arange(28, 32), np.arange(68, 72)]
        trip[slots] = BUS_CONFIG[bus]["ev_trip_kwh"] / len(slots)
    return trip


def clean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[np.abs(values) < 1e-8] = 0.0
    return np.round(values, 8)


def select_export_price(pa, pb, mode: str) -> np.ndarray:
    choice = str(mode).strip().upper()
    if choice == "PA":
        return np.asarray(pa, dtype=float)
    if choice == "PB":
        return np.asarray(pb, dtype=float)
    raise ValueError("EXPORT_PRICE_MODE must be 'PA' or 'PB'")


def day_frame(date, loads, pv, result, pa=None, pb=None) -> pd.DataFrame:
    timestamps = pd.date_range(date, periods=T, freq="15min")
    data = {
        "timestamp": timestamps,
        "date": pd.Timestamp(date).date().isoformat(),
        "step": np.arange(T),
        "time_h": np.arange(T) * DT,
    }
    aggregate_load = np.zeros(T)
    aggregate_pv = np.zeros(T)
    aggregate_net = np.zeros(T)
    for bus in BUS_ORDER:
        key = bus.lower().replace(" ", "_")
        item = result[bus]
        data[f"{key}_load_kw"] = clean(loads[bus])
        data[f"{key}_pv_kw"] = clean(pv[bus])
        for field in ("bess", "ev", "import", "export", "net"):
            data[f"{key}_{field}_kw"] = clean(item[field])
        data[f"{key}_bess_energy_kwh"] = clean(item["bess_energy"][:T])
        if "ev_energy" in item:
            data[f"{key}_ev_energy_kwh"] = clean(item["ev_energy"][:T])
        aggregate_load += loads[bus]
        aggregate_pv += pv[bus]
        aggregate_net += item["net"]
    data["aggregate_load_kw"] = clean(aggregate_load)
    data["aggregate_pv_kw"] = clean(aggregate_pv)
    data["aggregate_baseline_net_kw"] = clean(aggregate_load - aggregate_pv)
    data["aggregate_net_kw"] = clean(aggregate_net)
    if pa is not None:
        data["pa_eur_per_kwh"] = clean(pa)
    if pb is not None:
        data["pb_eur_per_kwh"] = clean(pb)
        data["total_buy_price_eur_per_kwh"] = clean(np.asarray(pa) + np.asarray(pb))
    if pa is not None and pb is not None:
        buy_price = np.asarray(pa, dtype=float) + np.asarray(pb, dtype=float)
        for bus in BUS_ORDER:
            key = bus.lower().replace(" ", "_")
            item = result[bus]
            sell_price = np.asarray(item.get("sell_price", pa), dtype=float)
            purchase = buy_price * np.asarray(item["import"], dtype=float) * DT
            revenue = sell_price * np.asarray(item["export"], dtype=float) * DT
            data[f"{key}_buy_price_eur_per_kwh"] = clean(buy_price)
            data[f"{key}_sell_price_eur_per_kwh"] = clean(sell_price)
            data[f"{key}_purchase_cost_eur"] = clean(purchase)
            data[f"{key}_sales_revenue_eur"] = clean(revenue)
            data[f"{key}_net_cost_eur"] = clean(purchase - revenue)
    return pd.DataFrame(data)


def day_metrics(scenario: str, date, frame: pd.DataFrame,
                result: dict, pa=None, pb=None) -> dict:
    net = frame["aggregate_net_kw"].to_numpy(float)
    baseline = frame["aggregate_baseline_net_kw"].to_numpy(float)
    import_energy = sum(float(np.sum(result[b]["import"]) * DT) for b in BUS_ORDER)
    export_energy = sum(float(np.sum(result[b]["export"]) * DT) for b in BUS_ORDER)
    row = {
        "scenario": scenario.upper(),
        "date": pd.Timestamp(date).date().isoformat(),
        "peak_kw": float(np.max(net)),
        "valley_kw": float(np.min(net)),
        "peak_valley_kw": float(np.ptp(net)),
        "maximum_ramp_kw_per_15min": float(np.max(np.abs(np.diff(net)))),
        "baseline_peak_kw": float(np.max(baseline)),
        "baseline_peak_valley_kw": float(np.ptp(baseline)),
        "import_energy_kwh": import_energy,
        "export_energy_kwh": export_energy,
        "purchase_cost_eur": np.nan,
        "sales_revenue_eur": np.nan,
        "net_electricity_cost_eur": np.nan,
    }
    if pa is not None and pb is not None:
        buy_price = np.asarray(pa) + np.asarray(pb)
        purchase = 0.0
        revenue = 0.0
        for bus in BUS_ORDER:
            item = result[bus]
            sell = np.asarray(item.get("sell_price", pa), dtype=float)
            purchase += float(np.dot(buy_price, item["import"]) * DT)
            revenue += float(np.dot(sell, item["export"]) * DT)
        row["purchase_cost_eur"] = purchase
        row["sales_revenue_eur"] = revenue
        row["net_electricity_cost_eur"] = purchase - revenue
    for bus in BUS_ORDER:
        key = bus.lower().replace(" ", "_")
        item = result[bus]
        bus_import = float(np.sum(item["import"]) * DT)
        bus_export = float(np.sum(item["export"]) * DT)
        row[f"{key}_import_energy_kwh"] = bus_import
        row[f"{key}_export_energy_kwh"] = bus_export
        if pa is not None and pb is not None:
            buy_price = np.asarray(pa, dtype=float) + np.asarray(pb, dtype=float)
            sell_price = np.asarray(item.get("sell_price", pa), dtype=float)
            bus_purchase = float(np.dot(buy_price, item["import"]) * DT)
            bus_revenue = float(np.dot(sell_price, item["export"]) * DT)
            row[f"{key}_purchase_cost_eur"] = bus_purchase
            row[f"{key}_sales_revenue_eur"] = bus_revenue
            row[f"{key}_net_cost_eur"] = bus_purchase - bus_revenue
            row[f"{key}_average_buy_price_eur_per_kwh"] = (
                bus_purchase / bus_import if bus_import > 1e-9 else 0.0)
            row[f"{key}_average_sell_price_eur_per_kwh"] = (
                bus_revenue / bus_export if bus_export > 1e-9 else 0.0)
    return row


def save_month(scenario: str, frames: list[pd.DataFrame],
               metrics: list[dict]) -> tuple[Path, Path]:
    csv_folder, _ = output_folders()
    timeseries_path = csv_folder / f"{scenario}_timeseries_month.csv"
    metrics_path = csv_folder / f"{scenario}_daily_metrics.csv"
    pd.concat(frames, ignore_index=True).to_csv(timeseries_path, index=False)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    return timeseries_path, metrics_path


PLOT_COLORS = {
    "load": "#505050", "net": "#245A8D", "bess": "#6A4C93",
    "ev": "#008C95", "pv": "#3A7D44", "import": "#2868A9",
    "export": "#B23A48", "pa": "#245A8D", "pb": "#A64B2A",
    "total_price": "#2F6B3C",
}


def _power_plot(axis, x, values, style, color, label, *, dashed=False, alpha=1.0):
    """Draw either a conventional line or one bar for every 15-minute value."""
    values = np.asarray(values, dtype=float)
    if style == "bar":
        axis.bar(x, values, width=DT / 24.0 * 0.90, color=color,
                 edgecolor="none", alpha=0.82 * alpha, label=label)
    else:
        axis.plot(x, values, linestyle="--" if dashed else "-", color=color,
                  lw=1.00 if not dashed else 0.85, alpha=alpha, label=label)


def _finish_axis(axis, title, ylabel="kW", zero=False):
    axis.set_title(title, loc="left", fontweight="normal")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D8D8D8", alpha=0.55, linewidth=0.55)
    if zero:
        axis.axhline(0.0, color="#6F6F6F", lw=0.65)
    axis.margins(y=0.12)


def _empty_device_axis(axis, title, message):
    _finish_axis(axis, title)
    axis.text(0.5, 0.5, message, transform=axis.transAxes,
              ha="center", va="center", color="#666666")
    axis.set_yticks([])


def _economic_values(frame, key):
    imported = float(frame[f"{key}_import_kw"].sum() * DT)
    exported = float(frame[f"{key}_export_kw"].sum() * DT)
    purchase_col = f"{key}_purchase_cost_eur"
    if purchase_col not in frame:
        return [imported, exported, np.nan, np.nan, np.nan, np.nan, np.nan]
    purchase = float(frame[purchase_col].sum())
    revenue = float(frame[f"{key}_sales_revenue_eur"].sum())
    return [imported, exported, purchase, revenue, purchase - revenue,
            purchase / imported if imported > 1e-9 else 0.0,
            revenue / exported if exported > 1e-9 else 0.0]


def _add_economic_table(axis, values):
    axis.axis("off")
    headers = ["Import\n[kWh]", "Export\n[kWh]", "Purchase\n[EUR]",
               "Revenue\n[EUR]", "Net cost\n[EUR]", "Avg buy\n[EUR/kWh]",
               "Avg sell\n[EUR/kWh]"]
    text = [["-" if not np.isfinite(value) else f"{value:.1f}" for value in values]]
    table = axis.table(cellText=text, colLabels=headers, cellLoc="center",
                       colLoc="center", loc="center", bbox=[0.01, 0.05, 0.98, 0.88])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        cell.set_linewidth(0.6)
        cell.set_facecolor("#EAF1F7" if row == 0 else "white")


def _plot_price(axis, x, frame, style):
    if "pa_eur_per_kwh" not in frame:
        _empty_device_axis(axis, "Electricity prices", "No price data")
        return
    # Prices remain lines in both figure editions.  Overlaying three price
    # columns would hide PA/PB and would not improve interval readability.
    _power_plot(axis, x, frame["pa_eur_per_kwh"], "line",
                PLOT_COLORS["pa"], "PA")
    if "pb_eur_per_kwh" in frame:
        _power_plot(axis, x, frame["pb_eur_per_kwh"], "line",
                    PLOT_COLORS["pb"], "PB", alpha=0.85)
        _power_plot(axis, x, frame["total_buy_price_eur_per_kwh"], "line",
                    PLOT_COLORS["total_price"], "PA + PB", alpha=0.75)
    _finish_axis(axis, "Electricity prices", "EUR/kWh")
    axis.legend(ncol=3, frameon=False)


def _plot_bus_figure(scenario, bus_number, bus, frame, x, style, folder):
    key = bus.lower().replace(" ", "_")
    fig = plt.figure(figsize=(14, 17))
    grid = fig.add_gridspec(7, 1, height_ratios=[1.5, 1, 1, 1, 1, 1, 0.72],
                            hspace=0.34)
    axes = [fig.add_subplot(grid[index, 0]) for index in range(6)]
    table_axis = fig.add_subplot(grid[6, 0])

    _power_plot(axes[0], x, frame[f"{key}_load_kw"], "line",
                PLOT_COLORS["load"], "SimBench load", dashed=True)
    _power_plot(axes[0], x, frame[f"{key}_net_kw"], style,
                PLOT_COLORS["net"], "Total net load (all devices)")
    _finish_axis(axes[0], "Load and total net exchange", zero=True)
    axes[0].legend(ncol=2, frameon=False)

    _power_plot(axes[1], x, frame[f"{key}_bess_kw"], style,
                PLOT_COLORS["bess"], "BESS power")
    _finish_axis(axes[1], "BESS power (+ charge, - discharge)", zero=True)
    axes[1].legend(frameon=False)

    ev = frame[f"{key}_ev_kw"].to_numpy(float)
    if "ev_kwh" in BUS_CONFIG[bus]:
        _power_plot(axes[2], x, ev, style, PLOT_COLORS["ev"], "EV power")
        _finish_axis(axes[2], "EV power (+ charge, - discharge)", zero=True)
        axes[2].legend(frameon=False)
    else:
        _empty_device_axis(axes[2], "EV power", "No EV on this Bus")

    pv = frame[f"{key}_pv_kw"].to_numpy(float)
    if np.max(np.abs(pv)) > 1e-9:
        _power_plot(axes[3], x, pv, style, PLOT_COLORS["pv"], "PV generation")
        _finish_axis(axes[3], "PV generation")
        axes[3].legend(frameon=False)
    else:
        _empty_device_axis(axes[3], "PV generation", "No PV on this Bus")

    _power_plot(axes[4], x, frame[f"{key}_import_kw"], style,
                PLOT_COLORS["import"], "Import (+)")
    _power_plot(axes[4], x, -frame[f"{key}_export_kw"], style,
                PLOT_COLORS["export"], "Export (-)", alpha=0.88)
    _finish_axis(axes[4], "Grid transactions", zero=True)
    axes[4].legend(ncol=2, frameon=False)

    _plot_price(axes[5], x, frame, style)
    axes[5].set_xlabel("Day of July 2026")
    _add_economic_table(table_axis, _economic_values(frame, key))

    for axis in axes[:-1]:
        axis.tick_params(labelbottom=False)
    cfg = BUS_CONFIG[bus]
    fig.suptitle(
        f"{scenario.upper()} | {bus} ({SIMBENCH_MAPPING[bus]['name']}) | "
        f"BESS {cfg['bess_kwh']:.1f} kWh / {cfg['bess_kw']:.1f} kW",
        y=0.995,
    )
    path = folder / f"{scenario}_bus{bus_number}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(scenario, title, frame, x, style, folder):
    fig, axes = plt.subplots(6, 1, figsize=(15, 15), sharex=True)
    _power_plot(axes[0], x, frame["aggregate_load_kw"], "line",
                PLOT_COLORS["load"], "SimBench load", dashed=True)
    _power_plot(axes[0], x, frame["aggregate_net_kw"], style,
                PLOT_COLORS["net"], "Aggregate total net load")
    _finish_axis(axes[0], "Aggregate load and net exchange", zero=True)
    axes[0].legend(ncol=2, frameon=False)

    aggregate_bess = sum(frame[f"{b.lower().replace(' ', '_')}_bess_kw"]
                         for b in BUS_ORDER)
    aggregate_ev = sum(frame[f"{b.lower().replace(' ', '_')}_ev_kw"]
                       for b in BUS_ORDER)
    aggregate_import = sum(frame[f"{b.lower().replace(' ', '_')}_import_kw"]
                           for b in BUS_ORDER)
    aggregate_export = sum(frame[f"{b.lower().replace(' ', '_')}_export_kw"]
                           for b in BUS_ORDER)
    for axis, values, color, axis_title, legend in (
        (axes[1], aggregate_bess, PLOT_COLORS["bess"],
         "Aggregate BESS power (+ charge, - discharge)", "BESS"),
        (axes[2], aggregate_ev, PLOT_COLORS["ev"],
         "Aggregate EV power (+ charge, - discharge)", "EV"),
        (axes[3], frame["aggregate_pv_kw"], PLOT_COLORS["pv"],
         "Aggregate PV generation", "PV"),
    ):
        _power_plot(axis, x, values, style, color, legend)
        _finish_axis(axis, axis_title, zero="discharge" in axis_title)
        axis.legend(frameon=False)
    _power_plot(axes[4], x, aggregate_import, style,
                PLOT_COLORS["import"], "Import (+)")
    _power_plot(axes[4], x, -aggregate_export, style,
                PLOT_COLORS["export"], "Export (-)")
    _finish_axis(axes[4], "Aggregate grid transactions", zero=True)
    axes[4].legend(ncol=2, frameon=False)
    _plot_price(axes[5], x, frame, style)
    axes[5].set_xlabel("Day of July 2026")
    fig.suptitle(f"{title} — {style.capitalize()} view", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.982))
    path = folder / f"{scenario}_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_month(scenario: str, title: str, frame: pd.DataFrame,
               metrics: pd.DataFrame, show_price: bool) -> Path:
    """Create both readable line figures and 15-minute column figures."""
    del metrics, show_price  # all needed information is already in the timeseries
    x = np.arange(len(frame)) * DT / 24.0
    returned_path = None
    for style in ("line", "bar"):
        folder = scenario_figure_folder(scenario, style)
        overview_path = _plot_overview(scenario, title, frame, x, style, folder)
        for bus_number, bus in enumerate(BUS_ORDER, 1):
            _plot_bus_figure(scenario, bus_number, bus, frame, x, style, folder)
        if style == "line":
            returned_path = overview_path
    return returned_path


DEFAULT_SETTINGS = {
    "SIMBENCH_CODE": "1-LV-urban6--0-no_sw",
    # The same SimBench day is repeated on all experiment dates.
    "LOAD_DAY_INDEX": 0,
    "START_DATE": "2026-07-01",
    "END_DATE": "2026-07-31",
    # Use an integer such as 2 for a quick test; None runs all 31 days.
    "MAX_DAYS": 2,
    "PA_FILE": "pa_de_lu_seq1_2026.csv",
    "PA_TIME_COLUMN": "MTU (CET/CEST)",
    "PA_SEQUENCE_COLUMN": "Sequence",
    "PA_SEQUENCE": "Sequence 1",
    "PA_PRICE_COLUMN": "Day-ahead Price (EUR/MWh)",
    "PV_FILE": "pv_smard_de_lu_2026_07.csv",
    "PV_TIME_COLUMN": "Start date",
    "PV_COLUMN_KEYWORD": "Photovoltaics",
    "PV_CAPACITY_KW": {
        "Bus 1": 3.0, "Bus 2": 2.5, "Bus 3": 0.0, "Bus 4": 0.0,
    },
    "EXPORT_PRICE_MODE": "PA",
    "FLOW_TIE_BREAK_EUR_PER_KWH": 1e-5,
    "TERMINAL_ENERGY_TOL_FRAC": 0.05,
}
