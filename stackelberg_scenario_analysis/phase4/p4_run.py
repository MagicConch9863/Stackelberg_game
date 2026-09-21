from __future__ import annotations

import json

import pandas as pd

from p4_common import (
    PAPER_METRICS,
    build_sampling_plan,
    day_inputs,
    daily_metrics,
    descriptive_table,
    histogram_frequency_table,
    load_simbench_profiles,
    paired_improvement_table,
    read_pa_year,
    read_pv_source,
    result_folder,
    valid_dates,
)
from p4_statistics_plot import plot_all_experiments
from p4_s1_baseline import solve_day as solve_s1
from p4_s2_fixed_tariff import solve_day as solve_s2
from p4_s3_stackelberg import solve_day as solve_s3
from p4_s4_centralized import solve_day as solve_s4


# ============================================================
# MODIFY PHASE 4 SETTINGS HERE
# ============================================================

SETTINGS = {
    "FORMULATION_VERSION": "soft_soc_20_80_v1",
    # --------------------------------------------------------
    # Year / external data
    # --------------------------------------------------------
    "YEAR": 2025,

    # Real 2025 day-ahead price file.
    "PA_FILE": "pa_de_lu_seq1_2025.csv",
    "PA_TIME_COLUMN": "MTU (CET/CEST)",
    "PA_SEQUENCE_COLUMN": "Sequence",
    "PA_SEQUENCE": "Sequence 1",
    "PA_PRICE_COLUMN": "Day-ahead Price (EUR/MWh)",

    # --------------------------------------------------------
    # PV source mode
    # --------------------------------------------------------
    # Current screenshot only contains pv_smard_de_lu_2026_07.csv.csv.
    # Therefore the default mode below RUNS with the files you currently have.
    # It uses that file only as a normalised reference shape and generates
    # date-dependent PV sensitivity profiles for 2025.
    #
    # Later, when actual 2025 PV files are available, change to:
    #     "PV_MODE": "actual_2025"
    # --------------------------------------------------------
    "PV_MODE": "reference_profile",

    # Used only when PV_MODE == 'reference_profile'.
    "PV_REFERENCE_FILE": "pv_smard_de_lu_2026_07.csv",

    # Used only when PV_MODE == 'actual_2025'.
    "PV_FILE": "pv_smard_de_lu_2025.csv",
    "PV_MONTHLY_PATTERN": "pv_smard_de_lu_2025_{month:02d}.csv",

    "PV_TIME_COLUMN": "Start date",
    "PV_COLUMN_KEYWORD": "Photovoltaics",

    # Month-dependent scale for reference_profile mode.
    # Easy to modify. July is the reference maximum (=1.0).
    "PV_MONTH_FACTORS": {
        1: 0.25,
        2: 0.35,
        3: 0.55,
        4: 0.75,
        5: 0.90,
        6: 1.00,
        7: 1.00,
        8: 0.90,
        9: 0.70,
        10: 0.50,
        11: 0.30,
        12: 0.20,
    },

    # Random daily PV multiplier in reference_profile mode.
    # Same date always gets the same factor because the seed is date-based.
    "PV_DAY_FACTOR_RANGE": (1, 1),
    "PV_RANDOM_SEED": 20250817,

    # --------------------------------------------------------
    # C-B SimBench load
    # --------------------------------------------------------
    "SIMBENCH_CODE": "1-LV-urban6--0-no_sw",

    # Safe default: keep one SimBench 24h load profile fixed while PA/PV vary.
    # If your installed SimBench returns a full-year profile, you can change to:
    #     "LOAD_MODE": "annual"
    "LOAD_MODE": "fixed_day",
    "FIXED_LOAD_DAY_INDEX": 0,

    # --------------------------------------------------------
    # PV capacities on the selected four buses
    # --------------------------------------------------------
    "PV_CAPACITY_KW": {
        "Bus 1": 3.0,
        "Bus 2": 2.5,
        "Bus 3": 0.0,
        "Bus 4": 0.0,
    },

    # --------------------------------------------------------
    # BESS / S2 / S3
    # --------------------------------------------------------
    # SOC is physically constrained to [0%, 100%].  The preferred interval is
    # soft and time invariant; no terminal/day-ahead SOC reference is used.
    "SOC_SOFT_MIN_FRAC": 0.20,
    "SOC_SOFT_MAX_FRAC": 0.80,
    "SOC_SOFT_PENALTY_EUR_PER_KWH": 1.0,
    "FIXED_PB_EUR_PER_KWH": 0.10,

    "PB_MIN": 0.0,
    "PB_MAX": 0.30,
    "PB_RAMP": 0.15,
    "ENFORCE_PB_DAILY_MEAN": False,
    "PB_DAILY_MEAN": 0.10,
    "GRID_BOUND_KW": 20.0,
    "DUAL_BIG_M": 100.0, 
    "MIP_GAP": 0.05,
    "TIME_LIMIT_S_PER_DAY": 1200,
    "SOLVER_OUTPUT": 0,
    "GUROBI_DISPLAY_INTERVAL_S": 2,
    "FLOW_TIE_BREAK_EUR_PER_KWH": 1e-5,

    # --------------------------------------------------------
    # Scenarios
    # --------------------------------------------------------
    "SCENARIOS": ("s1", "s2", "s3", "s4"),

    # --------------------------------------------------------
    # CUSTOM RANDOM MONTH SAMPLING
    # --------------------------------------------------------
    # Only enabled groups are executed.
    # Default below = exactly 6 months: Jan/Mar/May/Jul/Sep/Nov,
    # 2 random days per month = 12 sampled day occurrences.
    # --------------------------------------------------------
    "EXPERIMENTS": {
        "six_month_test": {
            "enabled": True,
            "months": [ 1, 5, 10],
            "days_per_month": 1,
            "repetitions": 1,
        },

        # Examples kept disabled. Turn on only when needed.
        "winter": {
            "enabled": False,
            "months": [12, 1, 2],
            "days_per_month": 10,
            "repetitions": 1,
        },
        "spring": {
            "enabled": False,
            "months": [3, 4, 5],
            "days_per_month": 10,
            "repetitions": 1,
        },
        "summer": {
            "enabled": False,
            "months": [6, 7, 8],
            "days_per_month": 10,
            "repetitions": 1,
        },
        "autumn": {
            "enabled": False,
            "months": [9, 10, 11],
            "days_per_month": 10,
            "repetitions": 1,
        },
    },

    "RANDOM_SEED": 20250817,
    "WEEKDAYS_ONLY": False,

    # Skip price DST/incomplete days so every optimization is 96 steps.
    "REQUIRE_96_PRICE_INTERVALS": True,
    "REQUIRE_96_PV_INTERVALS": True,

    # For a very quick remote-desktop test set 1 or 2.
    # None = solve every distinct date selected above.
    "MAX_UNIQUE_DAYS": None,

    # --------------------------------------------------------
    # Statistics / figures
    # --------------------------------------------------------
    "BOOTSTRAP_SAMPLES": 5000,
    "BOOTSTRAP_SEED": 42,
    "HISTOGRAM_BINS": 10,

    # One scientific 2x3 boxplot figure per enabled experiment.
    # No individual per-bus/per-metric images are generated.
    "CREATE_DISTRIBUTION_FIGURES": True,
}


# ============================================================
# Simulation helpers
# ============================================================

def solve_unique_dates(plan, settings, pa_year, pv_source):
    simbench_profiles, full_days = load_simbench_profiles(settings)

    unique_dates = sorted(
        pd.to_datetime(plan["date"]).dt.normalize().unique()
    )

    maximum = settings.get("MAX_UNIQUE_DAYS")
    if maximum is not None:
        unique_dates = unique_dates[: int(maximum)]

    rows = []
    scenarios = tuple(str(s).lower() for s in settings["SCENARIOS"])

    for index, date_value in enumerate(unique_dates, start=1):
        date = pd.Timestamp(date_value)
        print(
            f"\n=== Phase 4 day {index}/{len(unique_dates)}: "
            f"{date.date()} ==="
        )

        loads, pv, pa = day_inputs(
            date,
            simbench_profiles,
            full_days,
            pa_year,
            pv_source,
            settings,
        )

        # S1: pure physical baseline. No PA/PB and no optimization.
        if "s1" in scenarios:
            print("[S1] Baseline calculation (no Gurobi optimization)...", flush=True)
            result = solve_s1(loads, pv)
            rows.append(daily_metrics("s1", date, result))

        # S2: real PA + fixed PB; BESS economic response.
        if "s2" in scenarios:
            print("[S2] Fixed-tariff BESS optimization starting...", flush=True)
            result, pb = solve_s2(loads, pv, pa, settings)
            print("[S2] Completed.", flush=True)
            rows.append(daily_metrics("s2", date, result, pa=pa, pb=pb))

        # S3: Stackelberg optimized PB.
        if "s3" in scenarios:
            print("[S3] Stackelberg KKT + Big-M model starting...", flush=True)
            result, pb, mip_gap = solve_s3(date, loads, pv, pa, settings)
            print(f"[S3] Completed. Final reported MIP gap = {mip_gap:.6g}", flush=True)
            row = daily_metrics("s3", date, result, pa=pa, pb=pb)
            row["solver_mip_gap"] = mip_gap  # diagnostic only, not paper metric
            rows.append(row)

        # S4: centralized BESS benchmark. No prices and no monetary accounting.
        if "s4" in scenarios:
            print("[S4] Centralized BESS benchmark optimization starting...", flush=True)
            result = solve_s4(date, loads, pv, settings)
            print("[S4] Completed.", flush=True)
            rows.append(daily_metrics("s4", date, result))

    return pd.DataFrame(rows)


def expand_by_sampling_plan(physical_daily, plan):
    plan = plan.copy()
    plan["date"] = pd.to_datetime(plan["date"]).dt.date.astype(str)

    physical = physical_daily.copy()
    physical["date"] = pd.to_datetime(physical["date"]).dt.date.astype(str)

    return plan.merge(
        physical,
        on=["date", "month"],
        how="inner",
        validate="many_to_many",
    )


def save_statistical_outputs(expanded, settings):
    out = result_folder()
    metrics = tuple(metric for metric in PAPER_METRICS if metric in expanded.columns)

    descriptive_table(
        expanded,
        ["experiment", "scenario"],
        metrics,
        settings,
    ).to_csv(out / "statistics_by_experiment.csv", index=False)

    descriptive_table(
        expanded,
        ["experiment", "month", "scenario"],
        metrics,
        settings,
    ).to_csv(out / "statistics_by_month.csv", index=False)

    histogram_frequency_table(
        expanded,
        ["experiment", "scenario"],
        metrics,
        int(settings["HISTOGRAM_BINS"]),
    ).to_csv(out / "histogram_frequency_table.csv", index=False)

    paired_improvement_table(
        expanded,
        "s2",
        "s3",
        metrics,
        settings,
    ).to_csv(out / "paired_s2_to_s3.csv", index=False)

    paired_improvement_table(
        expanded,
        "s3",
        "s4",
        metrics,
        settings,
    ).to_csv(out / "paired_s3_to_s4.csv", index=False)


# ============================================================
# Main
# ============================================================

def main():
    settings = SETTINGS
    out = result_folder()

    # Actual 2025 PA.
    pa_year, pa_counts = read_pa_year(settings)

    # Actual 2025 PV OR temporary reference-profile mode.
    pv_source = read_pv_source(settings)
    pv_counts = pv_source["counts"]

    valid = valid_dates(pa_counts, pv_counts, settings)
    print(f"Valid 96-step dates in {settings['YEAR']}: {len(valid)}")

    plan = build_sampling_plan(valid, settings)
    plan.to_csv(out / "sampling_plan.csv", index=False)

    physical_daily = solve_unique_dates(plan, settings, pa_year, pv_source)
    physical_daily.to_csv(out / "physical_daily_performance.csv", index=False)

    expanded = expand_by_sampling_plan(physical_daily, plan)
    expanded.to_csv(out / "sampled_daily_performance.csv", index=False)

    save_statistical_outputs(expanded, settings)
    plot_all_experiments(expanded, settings)

    with open(out / "phase4_settings.json", "w", encoding="utf-8") as file:
        json.dump(settings, file, indent=2, ensure_ascii=False, default=str)

    print("\n====================================================")
    print("Phase 4 completed")
    print("====================================================")
    print(f"Results: {out}")
    print(f"Sampling occurrences: {len(plan)}")
    print(f"Distinct physical dates solved: {physical_daily['date'].nunique()}")
    print(f"PV_MODE: {settings['PV_MODE']}")
    print(f"LOAD_MODE: {settings['LOAD_MODE']}")
    print("S1/S4 electricity cost is intentionally NaN: no price is defined.")


if __name__ == "__main__":
    main()
