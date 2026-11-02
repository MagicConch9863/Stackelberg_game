from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import simbench as sb


# ============================================================
# Core model constants
# ============================================================

T = 96
DT = 0.25

BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")

# Keep the same C-B SimBench buses used in the previous phase.
SIMBENCH_MAPPING = {
    "Bus 1": {"name": "BUS6", "index": 6},
    "Bus 2": {"name": "BUS8", "index": 8},
    "Bus 3": {"name": "BUS14", "index": 14},
    "Bus 4": {"name": "BUS16", "index": 17},
}

# Phase 4 intentionally removes EV completely.
BUS_CONFIG = {
    "Bus 1": {
        "bess_kwh": 8.0,
        "bess_kw": 2.5,
        "bess_e_min": 0.0,
        "bess_e_max": 8.0,
        "bess_e_init": 4.0,
    },
    "Bus 2": {
        "bess_kwh": 5.0,
        "bess_kw": 1.5,
        "bess_e_min": 0.0,
        "bess_e_max": 5.0,
        "bess_e_init": 2.5,
    },
    "Bus 3": {
        "bess_kwh": 8.0,
        "bess_kw": 2.5,
        "bess_e_min": 0.0,
        "bess_e_max": 8.0,
        "bess_e_init": 4.0,
    },
    "Bus 4": {
        "bess_kwh": 8.0,
        "bess_kw": 2.5,
        "bess_e_min": 0.0,
        "bess_e_max": 8.0,
        "bess_e_init": 4.0,
    },
}

# Exactly the six metrics requested for the main scientific figure/table.
PAPER_METRICS = (
    "export_energy_kwh",
    "import_energy_kwh",
    "maximum_ramp_kw_per_15min",
    "net_electricity_cost_eur",
    "peak_kw",
    "valley_kw",
)

PAPER_METRIC_LABELS = {
    "export_energy_kwh": "Export Energy [kWh]",
    "import_energy_kwh": "Import Energy [kWh]",
    "maximum_ramp_kw_per_15min": "Maximum Ramp [kW/15 min]",
    "net_electricity_cost_eur": "Net Electricity Cost [EUR]",
    "peak_kw": "Peak [kW]",
    "valley_kw": "Valley [kW]",
}


# ============================================================
# Paths
# ============================================================

def phase4_root() -> Path:
    return Path(__file__).resolve().parent


def input_folder() -> Path:
    return phase4_root() / "input_data"


def result_folder() -> Path:
    folder = phase4_root() / "results"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def figure_folder() -> Path:
    folder = result_folder() / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _candidate_names(filename: str | Path) -> list[Path]:
    """Accept normal *.csv and Windows files accidentally named *.csv.csv."""
    path = Path(filename)
    text = str(path)
    names = [path]

    if text.lower().endswith(".csv"):
        names.append(Path(text + ".csv"))
    else:
        names.extend((Path(text + ".csv"), Path(text + ".csv.csv")))

    return list(dict.fromkeys(names))


def resolve_input_file(filename: str | Path) -> Path:
    path = Path(filename)
    if path.is_file():
        return path.resolve()

    folders = [input_folder(), phase4_root(), Path.cwd()]
    tried = []

    for folder in folders:
        for name in _candidate_names(filename):
            candidate = folder / name
            tried.append(str(candidate))
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot find input file {filename!r}.\n"
        f"Expected it under: {input_folder()}\n"
        f"Examples tried: {tried[:8]}"
    )


# ============================================================
# Generic CSV helpers
# ============================================================

def _find_column(frame: pd.DataFrame, wanted: str) -> str:
    exact = {str(c).strip().casefold(): c for c in frame.columns}
    key = wanted.strip().casefold()

    if key in exact:
        return exact[key]

    candidates = [
        c for c in frame.columns
        if key in str(c).strip().casefold()
    ]

    if len(candidates) == 1:
        return candidates[0]

    raise KeyError(
        f"Cannot uniquely find column containing {wanted!r}. "
        f"Available columns: {list(frame.columns)}"
    )


def _numeric_text(series: pd.Series) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan

        text = (
            str(value)
            .strip()
            .strip('"')
            .replace("\u00a0", "")
            .replace(" ", "")
        )

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


# ============================================================
# PA reader: actual 2025 price data
# ============================================================

def _parse_pa_timestamp(series: pd.Series) -> pd.Series:
    left = (
        series.astype(str)
        .str.split(r"\s+-\s+", n=1, regex=True)
        .str[0]
        .str.strip()
    )
    cleaned = left.str.replace(r"\s*\([^)]*\)\s*$", "", regex=True).str.strip()

    parsed = pd.to_datetime(
        cleaned,
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            cleaned.loc[missing],
            dayfirst=True,
            errors="coerce",
        )

    if parsed.isna().any():
        examples = left.loc[parsed.isna()].head(5).tolist()
        raise ValueError(f"Cannot parse PA timestamps. Examples: {examples}")

    return parsed


def read_pa_year(settings: dict) -> tuple[pd.Series, pd.Series]:
    path = resolve_input_file(settings["PA_FILE"])

    try:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, sep=None, engine="python", encoding="latin1")

    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]

    time_col = _find_column(frame, settings["PA_TIME_COLUMN"])
    sequence_col = _find_column(frame, settings["PA_SEQUENCE_COLUMN"])
    price_col = _find_column(frame, settings["PA_PRICE_COLUMN"])

    wanted_sequence = str(settings["PA_SEQUENCE"]).strip().casefold()
    sequence = frame[sequence_col].astype(str).str.strip().str.casefold()

    selected = frame.loc[
        sequence == wanted_sequence,
        [time_col, price_col],
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No rows with {settings['PA_SEQUENCE']!r} in {path.name}."
        )

    selected["timestamp"] = _parse_pa_timestamp(selected[time_col])
    selected["value"] = _numeric_text(selected[price_col])

    year = int(settings["YEAR"])
    selected = selected.loc[
        selected["timestamp"].dt.year == year
    ].dropna(subset=["value"])

    if selected.empty:
        raise ValueError(f"No PA data for year {year} in {path.name}.")

    raw_counts = (
        selected.assign(date=selected["timestamp"].dt.normalize())
        .groupby("date")
        .size()
    )

    # EUR/MWh -> EUR/kWh
    series = (
        selected.groupby("timestamp")["value"]
        .mean()
        .sort_index()
        / 1000.0
    )
    series.name = "pa_eur_per_kwh"

    print(f"PA source: {path.name}, rows for {year}: {len(selected)}")
    return series, raw_counts


# ============================================================
# PV readers
# ============================================================

def _read_one_pv_file(path: Path, settings: dict) -> pd.DataFrame:
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            frame = pd.read_csv(
                path,
                sep=";",
                engine="python",
                dtype=str,
                encoding=encoding,
            )
            break
        except UnicodeDecodeError as error:
            last_error = error
    else:
        raise last_error

    frame.columns = [str(c).strip().lstrip("\ufeff") for c in frame.columns]

    time_col = _find_column(frame, settings["PV_TIME_COLUMN"])
    pv_col = _find_column(frame, settings["PV_COLUMN_KEYWORD"])

    timestamp = pd.to_datetime(
        frame[time_col].astype(str).str.strip(),
        dayfirst=True,
        errors="coerce",
    )

    if timestamp.isna().any():
        examples = frame.loc[timestamp.isna(), time_col].head(5).tolist()
        raise ValueError(
            f"Cannot parse PV timestamps in {path.name}. Examples: {examples}"
        )

    return pd.DataFrame({
        "timestamp": timestamp,
        "value": _numeric_text(frame[pv_col]).clip(lower=0.0),
    })


def _actual_2025_pv(settings: dict) -> dict:
    """
    Use actual calendar-matched 2025 PV data.

    Supports either one annual file or monthly files for only the months enabled
    in EXPERIMENTS.
    """
    frames = []
    used_files = []

    annual_name = settings.get("PV_FILE")
    annual_path = None

    if annual_name:
        try:
            annual_path = resolve_input_file(annual_name)
        except FileNotFoundError:
            annual_path = None

    if annual_path is not None:
        frames.append(_read_one_pv_file(annual_path, settings))
        used_files.append(annual_path.name)
    else:
        pattern = settings.get("PV_MONTHLY_PATTERN")
        if not pattern:
            raise FileNotFoundError(
                "PV_MODE='actual_2025' but no annual PV file was found and "
                "PV_MONTHLY_PATTERN is empty."
            )

        months_needed = sorted({
            int(month)
            for experiment in settings["EXPERIMENTS"].values()
            if experiment.get("enabled", True)
            for month in experiment["months"]
        })

        missing = []
        for month in months_needed:
            filename = pattern.format(month=month)
            try:
                path = resolve_input_file(filename)
            except FileNotFoundError:
                missing.append(filename)
                continue

            frames.append(_read_one_pv_file(path, settings))
            used_files.append(path.name)

        if missing:
            raise FileNotFoundError(
                "PV_MODE='actual_2025' requires 2025 PV data for every selected "
                f"month. Missing: {missing}.\n"
                "If you only have the existing 2026-07 PV file, change "
                "PV_MODE to 'reference_profile'."
            )

    raw = pd.concat(frames, ignore_index=True)
    year = int(settings["YEAR"])
    raw = raw.loc[
        raw["timestamp"].dt.year == year
    ].dropna(subset=["value"])

    if raw.empty:
        raise ValueError(
            f"Actual PV input does not contain year {year}. "
            "Use PV_MODE='reference_profile' if the available PV file is only "
            "a reference month from another year."
        )

    raw_counts = (
        raw.assign(date=raw["timestamp"].dt.normalize())
        .groupby("date")
        .size()
    )

    series = raw.groupby("timestamp")["value"].mean().sort_index()
    reference = float(series.max())
    if reference <= 0:
        raise ValueError("PV source contains no positive values.")

    pv_pu = (series / reference).clip(0.0, 1.0)
    pv_pu.name = "pv_pu"

    print(
        f"PV mode: actual_2025 | files={used_files} | "
        f"normalization reference={reference:.6g}"
    )

    return {
        "mode": "actual_2025",
        "series": pv_pu,
        "counts": raw_counts,
        "files": used_files,
    }


def _reference_pv_profile(settings: dict) -> dict:
    """
    Temporary fallback for the currently available PV file.

    The existing reference file (e.g. 2026-07) is used ONLY to obtain a normalised
    96-step daily PV shape. Phase 4 then generates a 2025 daily PV shape using:

        reference shape x user-defined month factor x deterministic day factor

    Therefore PA remains real 2025 data, while PV is a synthetic sensitivity input.
    Switch PV_MODE to 'actual_2025' once actual 2025 PV files are available.
    """
    path = resolve_input_file(settings["PV_REFERENCE_FILE"])
    raw = _read_one_pv_file(path, settings)

    raw["date"] = raw["timestamp"].dt.normalize()
    raw["slot"] = (
        raw["timestamp"].dt.hour * 4
        + raw["timestamp"].dt.minute // 15
    )

    # Keep only full 96-step reference days.
    counts = raw.groupby("date").size()
    full_dates = counts.index[counts == T]
    usable = raw.loc[raw["date"].isin(full_dates)].copy()

    if usable.empty:
        raise ValueError(
            f"Reference PV file {path.name} has no complete 96-step day."
        )

    profile = usable.groupby("slot")["value"].mean().reindex(range(T))
    if profile.isna().any():
        raise ValueError(
            f"Reference PV file {path.name} cannot form a complete 96-step profile."
        )

    profile = profile.to_numpy(float)
    maximum = float(np.max(profile))
    if maximum <= 0:
        raise ValueError("Reference PV profile contains no positive PV generation.")

    profile = np.clip(profile / maximum, 0.0, 1.0)

    # Synthetic PV is always generated as 96 steps for every day in target year.
    year = int(settings["YEAR"])
    synthetic_counts = pd.Series(
        T,
        index=pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"),
        dtype=int,
    )

    print(
        f"PV mode: reference_profile | source={path.name} | "
        f"full reference days={len(full_dates)}"
    )

    return {
        "mode": "reference_profile",
        "profile": profile,
        "counts": synthetic_counts,
        "files": [path.name],
    }


def read_pv_source(settings: dict) -> dict:
    mode = str(settings.get("PV_MODE", "reference_profile")).strip().lower()

    if mode == "actual_2025":
        return _actual_2025_pv(settings)

    if mode == "reference_profile":
        return _reference_pv_profile(settings)

    raise ValueError(
        "PV_MODE must be either 'actual_2025' or 'reference_profile'."
    )


def pv_shape_for_date(date: pd.Timestamp, pv_source: dict, settings: dict) -> np.ndarray:
    date = pd.Timestamp(date).normalize()

    if pv_source["mode"] == "actual_2025":
        series = pv_source["series"]
        stop = date + pd.Timedelta(days=1)
        day = series.loc[(series.index >= date) & (series.index < stop)]
        if day.index.has_duplicates:
            day = day.groupby(level=0).mean()
        if len(day) != T:
            raise ValueError(
                f"PV: {date.date()} has {len(day)} usable intervals, expected {T}."
            )
        return day.to_numpy(float)

    profile = np.asarray(pv_source["profile"], dtype=float)
    month = int(date.month)

    month_factors = settings["PV_MONTH_FACTORS"]
    month_factor = float(month_factors.get(month, 1.0))

    low, high = settings["PV_DAY_FACTOR_RANGE"]
    seed = int(settings["PV_RANDOM_SEED"]) + int(date.strftime("%Y%m%d"))
    rng = np.random.default_rng(seed)
    day_factor = float(rng.uniform(float(low), float(high)))

    return np.clip(profile * month_factor * day_factor, 0.0, 1.0)


# ============================================================
# SimBench load profiles
# ============================================================

def load_simbench_profiles(settings: dict) -> tuple[pd.DataFrame, int]:
    net = sb.get_simbench_net(settings["SIMBENCH_CODE"])

    profiles = sb.get_absolute_profiles_from_relative_profiles(
        net=net,
        element="load",
        multiplying_column="p_mw",
        time_as_index=False,
    )

    full_days = len(profiles) // T
    if full_days < 1:
        raise ValueError("SimBench returned no complete 96-step load day.")

    mode = str(settings["LOAD_MODE"]).strip().lower()
    if mode == "annual" and full_days < 300:
        raise ValueError(
            f"LOAD_MODE='annual' was requested, but SimBench returned only "
            f"{full_days} complete days. Set LOAD_MODE='fixed_day' instead."
        )

    profiles = profiles.iloc[: full_days * T].reset_index(drop=True)
    column_map = {str(c): c for c in profiles.columns}

    result = {}
    for bus in BUS_ORDER:
        mapping = SIMBENCH_MAPPING[bus]
        sb_index = mapping["index"]
        actual_name = str(net.bus.at[sb_index, "name"])

        load_indices = net.load.index[
            net.load["bus"] == sb_index
        ].tolist()

        columns = [
            column_map[str(index)]
            for index in load_indices
            if str(index) in column_map
        ]

        if not columns:
            raise RuntimeError(
                f"{bus}: no SimBench load profile at {actual_name} "
                f"(bus index {sb_index})."
            )

        result[bus] = profiles[columns].sum(axis=1).to_numpy(float) * 1000.0
        print(
            f"{bus} <- {actual_name} (index {sb_index}), "
            f"load elements={load_indices}"
        )

    print(f"SimBench complete days available: {full_days} | LOAD_MODE={mode}")
    return pd.DataFrame(result), full_days


def simbench_load_for_date(
    date: pd.Timestamp,
    profiles: pd.DataFrame,
    full_days: int,
    settings: dict,
) -> dict[str, np.ndarray]:
    mode = str(settings["LOAD_MODE"]).strip().lower()

    if mode == "annual":
        day_index = (int(pd.Timestamp(date).dayofyear) - 1) % full_days
    elif mode == "fixed_day":
        day_index = int(settings["FIXED_LOAD_DAY_INDEX"])
        if not 0 <= day_index < full_days:
            raise ValueError(
                f"FIXED_LOAD_DAY_INDEX={day_index} is outside 0..{full_days - 1}."
            )
    else:
        raise ValueError("LOAD_MODE must be 'annual' or 'fixed_day'.")

    start = day_index * T
    day = profiles.iloc[start : start + T]

    if len(day) != T:
        raise RuntimeError(f"Cannot extract SimBench day index {day_index}.")

    return {bus: day[bus].to_numpy(float) for bus in BUS_ORDER}


# ============================================================
# Daily input extraction
# ============================================================

def _extract_pa_day(series: pd.Series, date: pd.Timestamp) -> np.ndarray:
    date = pd.Timestamp(date).normalize()
    stop = date + pd.Timedelta(days=1)
    day = series.loc[(series.index >= date) & (series.index < stop)]

    if day.index.has_duplicates:
        day = day.groupby(level=0).mean()

    if len(day) != T:
        raise ValueError(
            f"PA: {date.date()} has {len(day)} usable 15-min intervals; expected {T}."
        )

    return day.to_numpy(float)


def day_inputs(
    date: pd.Timestamp,
    simbench_profiles: pd.DataFrame,
    full_days: int,
    pa_year: pd.Series,
    pv_source: dict,
    settings: dict,
):
    loads = simbench_load_for_date(
        date,
        simbench_profiles,
        full_days,
        settings,
    )

    pa = _extract_pa_day(pa_year, date)
    pv_shape = pv_shape_for_date(date, pv_source, settings)

    pv = {
        bus: pv_shape * float(settings["PV_CAPACITY_KW"].get(bus, 0.0))
        for bus in BUS_ORDER
    }

    return loads, pv, pa


# ============================================================
# Random date sampling
# ============================================================

def valid_dates(
    pa_counts: pd.Series,
    pv_counts: pd.Series,
    settings: dict,
) -> pd.DatetimeIndex:
    year = int(settings["YEAR"])
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    valid = []

    for date in dates:
        if settings["WEEKDAYS_ONLY"] and date.weekday() >= 5:
            continue

        if settings["REQUIRE_96_PRICE_INTERVALS"]:
            if int(pa_counts.get(date, 0)) != T:
                continue

        if settings["REQUIRE_96_PV_INTERVALS"]:
            if int(pv_counts.get(date, 0)) != T:
                continue

        valid.append(date)

    if not valid:
        raise RuntimeError("No valid 96-step dates remain after filtering.")

    return pd.DatetimeIndex(valid)


def _sample_month(
    valid: pd.DatetimeIndex,
    month: int,
    n_days: int,
    rng: np.random.Generator,
) -> list[pd.Timestamp]:
    month_dates = valid[valid.month == month]

    if len(month_dates) < n_days:
        raise ValueError(
            f"Month {month}: only {len(month_dates)} valid dates are available, "
            f"but {n_days} were requested."
        )

    ids = np.sort(rng.choice(len(month_dates), size=n_days, replace=False))
    return [pd.Timestamp(month_dates[i]) for i in ids]


def build_sampling_plan(valid: pd.DatetimeIndex, settings: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(settings["RANDOM_SEED"]))
    rows = []

    for experiment_name, config in settings["EXPERIMENTS"].items():
        if not config.get("enabled", True):
            continue

        months = [int(m) for m in config["months"]]
        days_per_month = int(config["days_per_month"])
        repetitions = int(config.get("repetitions", 1))

        if any(month < 1 or month > 12 for month in months):
            raise ValueError(
                f"{experiment_name}: months must be integers 1..12, got {months}."
            )

        for repetition in range(1, repetitions + 1):
            for month in months:
                selected = _sample_month(
                    valid,
                    month,
                    days_per_month,
                    rng,
                )

                for date in selected:
                    rows.append({
                        "experiment": experiment_name,
                        "replication": repetition,
                        "month": month,
                        "date": date.normalize(),
                    })

    plan = pd.DataFrame(rows)

    if plan.empty:
        raise ValueError("Sampling plan is empty. Enable at least one experiment.")

    return plan.sort_values(
        ["experiment", "replication", "month", "date"]
    ).reset_index(drop=True)


# ============================================================
# Scenario result helpers
# ============================================================

def make_result_item(bess, import_kw, export_kw, bess_energy):
    bess = np.asarray(bess, dtype=float)
    imp = np.asarray(import_kw, dtype=float)
    exp = np.asarray(export_kw, dtype=float)

    return {
        "bess": bess,
        "import": imp,
        "export": exp,
        "net": imp - exp,
        "bess_energy": np.asarray(bess_energy, dtype=float),
    }


def baseline_result(loads, pv):
    """S1 = C-B physical baseline: SimBench load - PV, no price, no optimization."""
    result = {}

    for bus in BUS_ORDER:
        net = np.asarray(loads[bus], dtype=float) - np.asarray(pv[bus], dtype=float)

        result[bus] = make_result_item(
            bess=np.zeros(T),
            import_kw=np.maximum(net, 0.0),
            export_kw=np.maximum(-net, 0.0),
            bess_energy=np.full(T + 1, BUS_CONFIG[bus]["bess_e_init"]),
        )

    return result


def aggregate_net(result: dict) -> np.ndarray:
    return sum(np.asarray(result[bus]["net"], dtype=float) for bus in BUS_ORDER)


def daily_metrics(
    scenario: str,
    date: pd.Timestamp,
    result: dict,
    pa=None,
    pb=None,
) -> dict:
    """
    S1 and S4 intentionally have no monetary accounting.
    Their net_electricity_cost_eur is NaN, not zero.
    """
    net = aggregate_net(result)
    date = pd.Timestamp(date)

    row = {
        "scenario": scenario.upper(),
        "date": date.date().isoformat(),
        "month": int(date.month),
        "export_energy_kwh": float(
            sum(np.sum(result[bus]["export"]) * DT for bus in BUS_ORDER)
        ),
        "import_energy_kwh": float(
            sum(np.sum(result[bus]["import"]) * DT for bus in BUS_ORDER)
        ),
        "maximum_ramp_kw_per_15min": float(
            np.max(np.abs(np.diff(net))) if len(net) > 1 else 0.0
        ),
        "net_electricity_cost_eur": np.nan,
        "peak_kw": float(np.max(net)),
        "valley_kw": float(np.min(net)),
    }

    if scenario.lower() in ("s2", "s3") and pa is not None and pb is not None:
        pa = np.asarray(pa, dtype=float)
        pb = np.asarray(pb, dtype=float)
        buy_price = pa + pb

        total_cost = 0.0
        for bus in BUS_ORDER:
            imp = np.asarray(result[bus]["import"], dtype=float)
            exp = np.asarray(result[bus]["export"], dtype=float)
            total_cost += float(
                np.dot(buy_price, imp) * DT
                - np.dot(pa, exp) * DT
            )

        row["net_electricity_cost_eur"] = total_cost

    return row


# ============================================================
# Statistics
# ============================================================

def bootstrap_mean_ci(values, settings: dict) -> tuple[float, float]:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)

    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])

    rng = np.random.default_rng(int(settings["BOOTSTRAP_SEED"]))
    samples = int(settings["BOOTSTRAP_SAMPLES"])

    ids = rng.integers(0, len(x), size=(samples, len(x)))
    means = x[ids].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def descriptive_row(values, settings: dict) -> dict | None:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)

    if len(x) == 0:
        return None

    p05, p25, p50, p75, p95 = np.quantile(
        x,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    ci_low, ci_high = bootstrap_mean_ci(x, settings)
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    return {
        "n": int(len(x)),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": (
            std / abs(mean) if abs(mean) > 1e-12 else np.nan
        ),
        "median": float(p50),
        "min": float(np.min(x)),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(np.max(x)),
        "iqr": float(p75 - p25),
        "p05_p95_width": float(p95 - p05),
        "mean_ci95_low": ci_low,
        "mean_ci95_high": ci_high,
    }


def descriptive_table(frame, group_cols, metrics, settings):
    rows = []

    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        base = dict(zip(group_cols, key))

        for metric in metrics:
            if metric not in group.columns:
                continue

            stats = descriptive_row(group[metric], settings)
            if stats is None:
                continue

            rows.append({**base, "metric": metric, **stats})

    return pd.DataFrame(rows)


def histogram_frequency_table(frame, group_cols, metrics, bins):
    rows = []

    for key, group in frame.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)

        base = dict(zip(group_cols, key))

        for metric in metrics:
            if metric not in group.columns:
                continue

            x = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if len(x) == 0:
                continue

            if np.allclose(x, x[0]):
                edges = np.array([x[0] - 0.5, x[0] + 0.5])
            else:
                edges = np.histogram_bin_edges(x, bins=int(bins))

            counts, edges = np.histogram(x, bins=edges)
            total = int(counts.sum())

            for i, count in enumerate(counts):
                rows.append({
                    **base,
                    "metric": metric,
                    "bin_index": i + 1,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(count),
                    "relative_frequency": float(count / total) if total else np.nan,
                })

    return pd.DataFrame(rows)


def paired_improvement_table(
    frame,
    left_scenario,
    right_scenario,
    metrics,
    settings,
):
    keys = ["experiment", "replication", "date"]

    left = frame.loc[
        frame["scenario"] == left_scenario.upper()
    ].set_index(keys)

    right = frame.loc[
        frame["scenario"] == right_scenario.upper()
    ].set_index(keys)

    common = left.index.intersection(right.index)
    rows = []

    for metric in metrics:
        if metric not in left.columns or metric not in right.columns:
            continue

        a = pd.to_numeric(left.loc[common, metric], errors="coerce")
        b = pd.to_numeric(right.loc[common, metric], errors="coerce")

        valid = a.notna() & b.notna()
        a = a.loc[valid].to_numpy(float)
        b = b.loc[valid].to_numpy(float)

        if len(a) == 0:
            continue

        difference = a - b
        ci_low, ci_high = bootstrap_mean_ci(difference, settings)

        percent = np.full(len(a), np.nan, dtype=float)
        denominator = np.abs(a)
        mask = denominator > 1e-12
        percent[mask] = difference[mask] / denominator[mask] * 100.0

        rows.append({
            "comparison": f"{left_scenario.upper()}_to_{right_scenario.upper()}",
            "metric": metric,
            "n_pairs": int(len(difference)),
            "mean_left_minus_right": float(np.mean(difference)),
            "median_left_minus_right": float(np.median(difference)),
            "std_difference": (
                float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
            ),
            "mean_relative_change_percent": (
                float(np.nanmean(percent)) if np.isfinite(percent).any() else np.nan
            ),
            "median_relative_change_percent": (
                float(np.nanmedian(percent)) if np.isfinite(percent).any() else np.nan
            ),
            "probability_right_lower_than_left": float(np.mean(difference > 0)),
            "mean_difference_ci95_low": ci_low,
            "mean_difference_ci95_high": ci_high,
        })

    return pd.DataFrame(rows)
