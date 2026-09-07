from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")
STEPS_PER_DAY = 96


def _stable_rng(base_seed, date, bus, variable, scenario):
    text = f"{base_seed}|{pd.Timestamp(date).date()}|{bus}|{variable}|{scenario}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    return np.random.default_rng(seed)


def _iid_error(length, sigma_epsilon, rng):
    return rng.normal(0.0, sigma_epsilon, length)


def _stationary_ar1_error(length, rho, sigma_epsilon, rng):
    if not 0.0 < abs(rho) < 1.0:
        raise ValueError("Stationary AR(1) requires 0 < abs(rho) < 1.")
    error = np.empty(length, dtype=float)
    error[0] = rng.normal(0.0, sigma_epsilon / np.sqrt(1.0 - rho ** 2))
    innovations = rng.normal(0.0, sigma_epsilon, length)
    for step in range(1, length):
        error[step] = rho * error[step - 1] + innovations[step]
    return error


def _lag1(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.std(values[:-1]) <= 1e-12 or np.std(values[1:]) <= 1e-12:
        return np.nan
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _physical_forecast(realization, raw_error, variable, capacity_kw, pv_threshold_kw):
    realization = np.asarray(realization, dtype=float)
    raw_forecast = realization + raw_error

    if variable == "load":
        physical = np.maximum(raw_forecast, 0.0)
        active = np.ones(len(realization), dtype=bool)
        lower_clipped = raw_forecast < 0.0
        upper_clipped = np.zeros(len(realization), dtype=bool)
    elif variable == "pv":
        active = realization >= pv_threshold_kw
        physical = np.zeros(len(realization), dtype=float)
        physical[active] = np.clip(raw_forecast[active], 0.0, capacity_kw)
        lower_clipped = active & (raw_forecast < 0.0)
        upper_clipped = active & (raw_forecast > capacity_kw)
    else:
        raise ValueError(f"Unknown variable: {variable}")

    return physical, raw_forecast, active, lower_clipped, upper_clipped


def generate_bus_scenarios(
    date,
    loads,
    pv,
    pv_capacities,
    *,
    base_seed,
    sigma_epsilon_kw=0.3,
    ar1_rho=0.9,
    pv_active_threshold_kw=0.01,
):
    date = pd.Timestamp(date).normalize()
    timestamps = pd.date_range(date, periods=STEPS_PER_DAY, freq="15min")
    rows = []
    validation = []

    for bus in BUS_ORDER:
        realizations = {
            "load": np.asarray(loads[bus], dtype=float),
            "pv": np.asarray(pv[bus], dtype=float),
        }
        capacity_kw = float(pv_capacities.get(bus, 0.0))

        for variable, realization in realizations.items():
            if len(realization) != STEPS_PER_DAY:
                raise ValueError(
                    f"{date.date()} {bus} {variable}: expected 96 values, got {len(realization)}."
                )

            scenario_values = {}
            for scenario in ("iid", "ar1"):
                rng = _stable_rng(base_seed, date, bus, variable, scenario)
                if variable == "pv" and capacity_kw <= 0.0:
                    raw_error = np.zeros(STEPS_PER_DAY, dtype=float)
                elif scenario == "iid":
                    raw_error = _iid_error(STEPS_PER_DAY, sigma_epsilon_kw, rng)
                else:
                    raw_error = _stationary_ar1_error(
                        STEPS_PER_DAY, ar1_rho, sigma_epsilon_kw, rng
                    )

                physical, raw_forecast, active, lower_clip, upper_clip = (
                    _physical_forecast(
                        realization,
                        raw_error,
                        variable,
                        capacity_kw,
                        pv_active_threshold_kw,
                    )
                )
                scenario_values[scenario] = {
                    "forecast": physical,
                    "raw_forecast": raw_forecast,
                    "raw_error": raw_error,
                    "effective_error": physical - realization,
                    "active": active,
                    "lower_clip": lower_clip,
                    "upper_clip": upper_clip,
                }
                validation.append({
                    "date": date.date().isoformat(),
                    "bus": bus,
                    "variable": variable,
                    "scenario": scenario,
                    "sigma_epsilon_kW": sigma_epsilon_kw,
                    "rho": 0.0 if scenario == "iid" else ar1_rho,
                    "raw_error_mean_kW": float(np.mean(raw_error)),
                    "raw_error_std_kW": float(np.std(raw_error)),
                    "empirical_lag1_correlation": _lag1(raw_error),
                    "lower_clip_rate": float(np.mean(lower_clip)),
                    "upper_clip_rate": float(np.mean(upper_clip)),
                    "inactive_rate": float(np.mean(~active)) if variable == "pv" else 0.0,
                    "inactive_nonzero_violations": int(np.count_nonzero(physical[~active])),
                    "physical_min_kW": float(np.min(physical)),
                    "physical_max_kW": float(np.max(physical)),
                })

            for step, timestamp in enumerate(timestamps):
                row = {
                    "date": date.date().isoformat(),
                    "timestamp": timestamp,
                    "step": step,
                    "bus": bus,
                    "variable": variable,
                    "realization_kW": realization[step],
                    "forecast_perfect_kW": realization[step],
                }
                for scenario in ("iid", "ar1"):
                    values = scenario_values[scenario]
                    row[f"forecast_{scenario}_kW"] = values["forecast"][step]
                    row[f"raw_forecast_{scenario}_kW"] = values["raw_forecast"][step]
                    row[f"raw_error_{scenario}_kW"] = values["raw_error"][step]
                    row[f"effective_error_{scenario}_kW"] = values["effective_error"][step]
                    row[f"lower_clipped_{scenario}"] = values["lower_clip"][step]
                    row[f"upper_clipped_{scenario}"] = values["upper_clip"][step]
                row["pv_active"] = scenario_values["iid"]["active"][step]
                rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(validation)


@dataclass
class ForecastScenarioStore:
    frame: pd.DataFrame

    def __post_init__(self):
        required = {"date", "step", "bus", "variable", "realization_kW"}
        missing = required.difference(self.frame.columns)
        if missing:
            raise ValueError(f"Scenario frame is missing columns: {sorted(missing)}")
        self.frame = self.frame.copy()
        self.frame["date"] = pd.to_datetime(self.frame["date"]).dt.normalize()

    def _day(self, date):
        wanted = pd.Timestamp(date).normalize()
        day = self.frame[self.frame["date"] == wanted]
        if day.empty:
            raise KeyError(f"No scenario data for {wanted.date()}.")
        return day

    def realization(self, date, step):
        day = self._day(date)
        rows = day[day["step"] == int(step)]
        if len(rows) != len(BUS_ORDER) * 2:
            raise IndexError(f"Invalid realization step {step}.")
        return {
            variable: {
                bus: float(rows[(rows["bus"] == bus) & (rows["variable"] == variable)][
                    "realization_kW"
                ].iloc[0])
                for bus in BUS_ORDER
            }
            for variable in ("load", "pv")
        }

    def day_ahead(self, date, scenario):
        return self.window(date, 0, STEPS_PER_DAY, scenario)

    def mpc_window(self, date, step, horizon, scenario):
        return self.window(date, step, horizon, scenario)

    def window(self, date, step, horizon, scenario):
        scenario = str(scenario).lower()
        column = (
            "forecast_perfect_kW" if scenario in ("none", "perfect")
            else f"forecast_{scenario}_kW"
        )
        if column not in self.frame.columns:
            raise ValueError(f"Unknown forecast scenario: {scenario}")
        start = int(step)
        stop = min(STEPS_PER_DAY, start + int(horizon))
        if not 0 <= start < STEPS_PER_DAY or stop <= start:
            raise IndexError(f"Invalid window: step={step}, horizon={horizon}.")
        day = self._day(date)
        selected = day[(day["step"] >= start) & (day["step"] < stop)]
        return {
            variable: {
                bus: selected[
                    (selected["bus"] == bus) & (selected["variable"] == variable)
                ].sort_values("step")[column].to_numpy(float)
                for bus in BUS_ORDER
            }
            for variable in ("load", "pv")
        }
