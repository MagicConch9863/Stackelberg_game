
from pathlib import Path
import re
import numpy as np
import pandas as pd
import simbench as sb

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input_data"
RESULTS = ROOT / "results"
DATA_RESULTS = RESULTS / "data"
FORECAST_MODEL_RESULTS = RESULTS / "forecast_model_tests"
NOISE_MECHANISM_RESULTS = RESULTS / "noise_mechanism_tests"
CONTROL_INPUT_RESULTS = RESULTS / "control_inputs"
CONTROL_EXPERIMENT_RESULTS = RESULTS / "control_experiments"

DIRECT_ARX_RESULTS = FORECAST_MODEL_RESULTS / "direct_arx"
GAUSSIAN_NOISE_RESULTS = NOISE_MECHANISM_RESULTS / "iid_gaussian"
CORRELATED_GAUSSIAN_RESULTS = NOISE_MECHANISM_RESULTS / "correlated_gaussian"
RANDOM_WALK_RESULTS = NOISE_MECHANISM_RESULTS / "random_walk"

for output_dir in (
    DATA_RESULTS,
    DIRECT_ARX_RESULTS,
    GAUSSIAN_NOISE_RESULTS,
    CORRELATED_GAUSSIAN_RESULTS,
    RANDOM_WALK_RESULTS,
    CONTROL_INPUT_RESULTS,
    CONTROL_EXPERIMENT_RESULTS,
):
    output_dir.mkdir(parents=True, exist_ok=True)

PV_FILE = INPUT / "pv_smard_de_lu_2024_2025.csv"

SIMBENCH_CODE = "1-LV-urban6--0-no_sw"
BUS_INDEX = [6, 8, 14, 17]
PV_CAPACITY_KW = 5.5
STEPS_PER_DAY = 96

LAGS = (1, 2, 4, 8, 12, 24, 48, 96, 192, 672)
RIDGE = 1e-2


def _number(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace("\u00a0", "").replace(" ", "")
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def load_pv():
    if not PV_FILE.exists():
        raise FileNotFoundError(
            f"PV file not found: {PV_FILE}\n"
            "Put pv_smard_de_lu_2024_2025.csv into forecast_mpc/input_data/"
        )

    df = pd.read_csv(PV_FILE, sep=";", dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    time_col = next(c for c in df.columns if "Datum von" in c)
    pv_col = next(c for c in df.columns if "Photovoltaik" in c)

    time = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    pv = df[pv_col].map(_number)

    s = pd.Series(pv.to_numpy(), index=time).dropna().sort_index()
    s = s.groupby(level=0).mean().resample("15min").mean().interpolate(limit=4)

    # Germany-wide SMARD PV is used only as a temporal shape.
    shape = (s / s.max()).clip(0.0, 1.0)
    return (shape * PV_CAPACITY_KW).rename("pv_kw").clip(0.0, PV_CAPACITY_KW)


def load_simbench():
    net = sb.get_simbench_net(SIMBENCH_CODE)
    profiles = sb.get_absolute_profiles_from_relative_profiles(
        net=net,
        element="load",
        multiplying_column="p_mw",
        time_as_index=False,
    )

    colmap = {str(c): c for c in profiles.columns}
    total = np.zeros(len(profiles), dtype=float)

    for bus in BUS_INDEX:
        ids = net.load.index[net.load["bus"] == bus].tolist()
        cols = [colmap[str(i)] for i in ids if str(i) in colmap]
        if not cols:
            raise RuntimeError(f"No SimBench load found for bus index {bus}")
        total += profiles[cols].sum(axis=1).to_numpy(float) * 1000.0

    total = total[: (len(total) // STEPS_PER_DAY) * STEPS_PER_DAY]

    idx = pd.date_range(
        "2024-01-01",
        "2025-12-31 23:45",
        freq="15min",
    )
    values = np.resize(total, len(idx))

    return pd.Series(values, index=idx, name="load_kw").clip(lower=0.0)


def dataset():
    """
    Build the combined Load/PV dataset directly from:
      - SimBench annual load profile
      - SMARD 2024/2025 realized PV CSV

    No pre-existing results CSV is required.
    """
    load = load_simbench()
    pv = load_pv()

    idx = load.index.intersection(pv.index)
    data = pd.concat([load.loc[idx], pv.loc[idx]], axis=1).dropna()

    data.to_csv(
        DATA_RESULTS / "actual_load_pv_2024_2025.csv",
        index_label="timestamp",
    )
    return data


def score(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    e = p - a

    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    return mae, rmse


class DirectARX:
    """
    Direct multi-step ARX.

    A separate linear regression is fitted for each horizon h:
        y(k+h) = theta_h^T phi(k,h)

    No recursive feeding of predicted values is used.
    """

    def __init__(self, horizon, kind, ridge=RIDGE):
        self.H = int(horizon)
        self.kind = kind
        self.ridge = float(ridge)
        self.models = []

    def _features(self, y, k, h):
        target = k + h

        x = [1.0]

        # Recent / seasonal information known at forecast origin.
        x.extend(y[k - lag] for lag in LAGS)

        # Same target slot from previous day, two days ago, previous week.
        x.extend([
            y[target - 96],
            y[target - 192],
            y[target - 672],
        ])

        # Recent local level.
        x.extend([
            float(np.mean(y[k - 4:k])),
            float(np.mean(y[k - 8:k - 4])),
        ])

        return np.asarray(x, dtype=float)

    def fit(self, series):
        raw = np.asarray(series, dtype=float)

        self.mean = float(raw.mean())
        self.std = float(raw.std())
        if self.std <= 1e-12:
            self.std = 1.0

        y = (raw - self.mean) / self.std
        k0 = max(LAGS) + 1

        self.models = []

        for h in range(1, self.H + 1):
            X = []
            Y = []

            for k in range(k0, len(y) - h):
                X.append(self._features(y, k, h))
                Y.append(y[k + h])

            X = np.asarray(X)
            Y = np.asarray(Y)

            reg = self.ridge * np.eye(X.shape[1])
            reg[0, 0] = 0.0

            theta = np.linalg.solve(
                X.T @ X + reg,
                X.T @ Y,
            )
            self.models.append(theta)

        return self

    def forecast(self, full_series, origin_pos):
        """
        Forecast future H steps using only observations before origin_pos.
        """
        hist = np.asarray(full_series.iloc[:origin_pos], dtype=float)
        zhist = (hist - self.mean) / self.std

        # Padding is only for indexing; all target-aligned lags remain historical
        # for H <= 96.
        y = np.concatenate([zhist, np.zeros(self.H + 1)])

        out = []
        k = origin_pos

        for h, theta in enumerate(self.models, start=1):
            zhat = float(self._features(y, k, h) @ theta)
            pred = zhat * self.std + self.mean
            pred = max(0.0, float(pred))

            if self.kind == "pv":
                t = k + h
                prev_day = y[t - 96] * self.std + self.mean
                prev_2day = y[t - 192] * self.std + self.mean

                if prev_day <= 1e-9 and prev_2day <= 1e-9:
                    pred = 0.0

                pred = min(pred, PV_CAPACITY_KW)

            out.append(pred)

        return np.asarray(out)
