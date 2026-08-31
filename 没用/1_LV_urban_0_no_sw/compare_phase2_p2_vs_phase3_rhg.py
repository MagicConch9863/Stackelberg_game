# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEEDER = "1_LV_urban_0_no_sw"
PHASE2_DIR = "phase2_outputs1"
PHASE3_DIR = "phase3_outputs1"

PHASE2_CANDIDATES = ["phase2_P2_solution.csv", "phase2_solution.csv", "phase2_stackelberg_solution.csv"]
PHASE3_CANDIDATES = ["phase3_rhg_solution.csv", "phase3_solution.csv"]

OUT_DIR = "compare_outputs"
OUT_FIG = "phase2_P2_vs_phase3_RHG_full.png"
OUT_HOURLY = "phase2_P2_vs_phase3_RHG_hourly.csv"
OUT_SUMMARY = "phase2_P2_vs_phase3_RHG_summary.csv"

# ---------- leader cost weights ----------
# 如果你现在只关心削峰填谷，就保持下面这样：
W_GAP = 1.0
W_RAMP = 0.0
W_VAR = 0.0
W_PB = 0.0


def find_feeder_folder():
    here = Path(__file__).resolve().parent
    for p in [here] + list(here.parents):
        if p.name == FEEDER:
            return p
        c = p / FEEDER
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find feeder folder: {FEEDER}")


def find_csv(folder, candidates):
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    csvs = sorted(folder.glob("*.csv"))
    raise FileNotFoundError(
        f"No expected csv found in: {folder}\n"
        f"Expected one of: {candidates}\n"
        f"Available: {[x.name for x in csvs]}"
    )


def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Cannot find columns {candidates} in {list(df.columns)}")
    return None


def load_solution(path, label):
    df = pd.read_csv(path)

    hour_col = pick_col(df, ["hour", "Hour", "t"], required=False)
    net_col = pick_col(df, ["net_exchange_kw", "net_kw", "net_exchange"], required=True)
    base_col = pick_col(df, ["net_baseline_kw", "baseline_kw", "net_baseline"], required=False)
    pa_col = pick_col(df, ["pa_eur_per_kwh", "pa", "PA"], required=False)
    pb_col = pick_col(df, ["pb_eur_per_kwh", "pb", "PB"], required=False)
    total_col = pick_col(df, ["total_price_eur_per_kwh", "pa_plus_pb", "total_price"], required=False)

    out = pd.DataFrame()
    out["hour"] = df[hour_col].to_numpy(float) if hour_col else np.arange(len(df), dtype=float)
    out[f"{label}_net_exchange_kw"] = df[net_col].to_numpy(float)

    if base_col:
        out["net_baseline_kw"] = df[base_col].to_numpy(float)
    if pa_col:
        out[f"{label}_pa_eur_per_kwh"] = df[pa_col].to_numpy(float)
    if pb_col:
        out[f"{label}_pb_eur_per_kwh"] = df[pb_col].to_numpy(float)

    if total_col:
        out[f"{label}_total_price_eur_per_kwh"] = df[total_col].to_numpy(float)
    elif pa_col and pb_col:
        out[f"{label}_total_price_eur_per_kwh"] = df[pa_col].to_numpy(float) + df[pb_col].to_numpy(float)

    return out


def curve_metrics(x):
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    return {
        "peak_kw": float(np.max(x)),
        "valley_kw": float(np.min(x)),
        "gap_kw": float(np.max(x) - np.min(x)),
        "ramp_sum_kw": float(np.sum(np.abs(dx))),
        "variation": float(np.var(x)),
        "std_kw": float(np.std(x)),
        "mean_kw": float(np.mean(x)),
    }


def leader_cost(net, pb=None, w_gap=1.0, w_ramp=0.0, w_var=0.0, w_pb=0.0):
    net = np.asarray(net, dtype=float)
    gap = np.max(net) - np.min(net)
    ramp = np.sum(np.abs(np.diff(net)))
    var = np.var(net)
    pb_term = np.sum(pb) if pb is not None else 0.0
    return float(w_gap * gap + w_ramp * ramp + w_var * var + w_pb * pb_term)


def build_summary(hourly):
    base = hourly["net_baseline_kw"].to_numpy(float)
    p2 = hourly["P2_net_exchange_kw"].to_numpy(float)
    rhg = hourly["RHG_net_exchange_kw"].to_numpy(float)

    m_base = curve_metrics(base)
    m_p2 = curve_metrics(p2)
    m_rhg = curve_metrics(rhg)

    p2_pb = hourly["P2_pb_eur_per_kwh"].to_numpy(float) if "P2_pb_eur_per_kwh" in hourly else None
    rhg_pb = hourly["RHG_pb_eur_per_kwh"].to_numpy(float) if "RHG_pb_eur_per_kwh" in hourly else None

    p2_cost = leader_cost(p2, p2_pb, W_GAP, W_RAMP, W_VAR, W_PB)
    rhg_cost = leader_cost(rhg, rhg_pb, W_GAP, W_RAMP, W_VAR, W_PB)

    def row(name, m):
        return {
            "case": name,
            "peak_kw": m["peak_kw"],
            "valley_kw": m["valley_kw"],
            "gap_kw": m["gap_kw"],
            "ramp_sum_kw": m["ramp_sum_kw"],
            "variation": m["variation"],
            "mean_kw": m["mean_kw"],
            "std_kw": m["std_kw"],
        }

    summary = pd.DataFrame([
        row("baseline", m_base),
        row("P2", m_p2),
        row("RHG", m_rhg),
    ])

    summary["peak_reduction_pct"] = np.nan
    summary["valley_fill_kw"] = np.nan
    summary["gap_reduction_pct"] = np.nan
    summary["ramp_reduction_pct"] = np.nan
    summary["variation_reduction_pct"] = np.nan
    summary["leader_cost"] = np.nan

    base_peak = m_base["peak_kw"]
    base_gap = m_base["gap_kw"]
    base_ramp = m_base["ramp_sum_kw"]
    base_var = m_base["variation"]
    base_valley = m_base["valley_kw"]

    for case, m, cost in [("P2", m_p2, p2_cost), ("RHG", m_rhg, rhg_cost)]:
        idx = summary["case"] == case
        summary.loc[idx, "peak_reduction_pct"] = 100.0 * (base_peak - m["peak_kw"]) / max(base_peak, 1e-9)
        summary.loc[idx, "valley_fill_kw"] = m["valley_kw"] - base_valley
        summary.loc[idx, "gap_reduction_pct"] = 100.0 * (base_gap - m["gap_kw"]) / max(base_gap, 1e-9)
        summary.loc[idx, "ramp_reduction_pct"] = 100.0 * (base_ramp - m["ramp_sum_kw"]) / max(base_ramp, 1e-9)
        summary.loc[idx, "variation_reduction_pct"] = 100.0 * (base_var - m["variation"]) / max(base_var, 1e-9)
        summary.loc[idx, "leader_cost"] = cost

    if "P2_total_price_eur_per_kwh" in hourly:
        summary.loc[summary["case"] == "P2", "avg_total_price"] = hourly["P2_total_price_eur_per_kwh"].mean()
    if "RHG_total_price_eur_per_kwh" in hourly:
        summary.loc[summary["case"] == "RHG", "avg_total_price"] = hourly["RHG_total_price_eur_per_kwh"].mean()

    return summary


def plot_all(hourly, summary, out_png):
    h = hourly["hour"].to_numpy(float)

    fig = plt.figure(figsize=(16, 24))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.4, 1.6, 1.6, 1.8, 2.2])

    # 1) Net exchange
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(h, hourly["net_baseline_kw"], "--", linewidth=2.0, label="net baseline")
    ax0.plot(h, hourly["P2_net_exchange_kw"], linewidth=2.0, label="P2 from Phase 2")
    ax0.plot(h, hourly["RHG_net_exchange_kw"], linewidth=2.0, label="RHG from Phase 3")
    ax0.set_title("Phase 2 P2 vs Phase 3 RHG", fontsize=18)
    ax0.set_ylabel("net exchange [kW]")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    # 2) Price P2
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if "P2_pa_eur_per_kwh" in hourly:
        ax1.step(h, hourly["P2_pa_eur_per_kwh"], where="post", linestyle="--", label="PA")
    if "P2_pb_eur_per_kwh" in hourly:
        ax1.step(h, hourly["P2_pb_eur_per_kwh"], where="post", linestyle=":", label="PB P2")
    if "P2_total_price_eur_per_kwh" in hourly:
        ax1.step(h, hourly["P2_total_price_eur_per_kwh"], where="post", linewidth=2.0, label="PA + PB P2")
    ax1.set_ylabel("price [EUR/kWh]")
    ax1.set_title("Price comparison: Phase 2 P2")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    # 3) Price RHG
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    if "RHG_pa_eur_per_kwh" in hourly:
        ax2.step(h, hourly["RHG_pa_eur_per_kwh"], where="post", linestyle="--", label="PA")
    if "RHG_pb_eur_per_kwh" in hourly:
        ax2.step(h, hourly["RHG_pb_eur_per_kwh"], where="post", linestyle=":", label="PB RHG")
    if "RHG_total_price_eur_per_kwh" in hourly:
        ax2.step(h, hourly["RHG_total_price_eur_per_kwh"], where="post", linewidth=2.0, label="PA + PB RHG")
    ax2.set_ylabel("price [EUR/kWh]")
    ax2.set_title("Price comparison: Phase 3 RHG")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right")

    # 4) Metric bars
    ax3 = fig.add_subplot(gs[3, 0])
    cmp = summary[summary["case"].isin(["P2", "RHG"])].copy()
    metrics = [
        "peak_kw", "valley_kw", "gap_kw", "ramp_sum_kw",
        "variation", "leader_cost"
    ]
    labels = ["peak", "valley", "gap", "ramp", "variation", "leader cost"]
    x = np.arange(len(metrics))
    w = 0.35
    p2_vals = [cmp.loc[cmp["case"] == "P2", m].iloc[0] for m in metrics]
    rhg_vals = [cmp.loc[cmp["case"] == "RHG", m].iloc[0] for m in metrics]
    ax3.bar(x - w/2, p2_vals, w, label="P2")
    ax3.bar(x + w/2, rhg_vals, w, label="RHG")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title("Metric comparison")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="upper right")

    # 5) Table
    ax4 = fig.add_subplot(gs[4, 0])
    ax4.axis("off")
    table_cols = [
        "case", "peak_kw", "valley_kw", "gap_kw", "ramp_sum_kw",
        "variation", "peak_reduction_pct", "gap_reduction_pct",
        "ramp_reduction_pct", "leader_cost", "avg_total_price"
    ]
    show = summary.copy()
    for c in table_cols:
        if c not in show.columns:
            show[c] = np.nan
    show = show[table_cols].copy()

    for c in show.columns[1:]:
        show[c] = show[c].apply(lambda v: "" if pd.isna(v) else f"{v:.4f}")

    tbl = ax4.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax4.set_title("Numerical summary", pad=20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    feeder = find_feeder_folder()
    phase2_dir = feeder / PHASE2_DIR
    phase3_dir = feeder / PHASE3_DIR
    out_dir = feeder / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    p2_csv = find_csv(phase2_dir, PHASE2_CANDIDATES)
    rhg_csv = find_csv(phase3_dir, PHASE3_CANDIDATES)

    p2 = load_solution(p2_csv, "P2")
    rhg = load_solution(rhg_csv, "RHG")

    n = min(len(p2), len(rhg))
    p2 = p2.iloc[:n].reset_index(drop=True)
    rhg = rhg.iloc[:n].reset_index(drop=True)

    hourly = pd.DataFrame({"hour": p2["hour"]})
    hourly["P2_net_exchange_kw"] = p2["P2_net_exchange_kw"]
    hourly["RHG_net_exchange_kw"] = rhg["RHG_net_exchange_kw"]

    if "net_baseline_kw" in p2.columns:
        hourly["net_baseline_kw"] = p2["net_baseline_kw"]
    elif "net_baseline_kw" in rhg.columns:
        hourly["net_baseline_kw"] = rhg["net_baseline_kw"]
    else:
        raise KeyError("No net_baseline_kw found in either phase2 or phase3 csv")

    for col in ["P2_pa_eur_per_kwh", "P2_pb_eur_per_kwh", "P2_total_price_eur_per_kwh"]:
        if col in p2.columns:
            hourly[col] = p2[col]

    for col in ["RHG_pa_eur_per_kwh", "RHG_pb_eur_per_kwh", "RHG_total_price_eur_per_kwh"]:
        if col in rhg.columns:
            hourly[col] = rhg[col]

    summary = build_summary(hourly)

    hourly.to_csv(out_dir / OUT_HOURLY, index=False)
    summary.to_csv(out_dir / OUT_SUMMARY, index=False)
    plot_all(hourly, summary, out_dir / OUT_FIG)

    print("\n[loaded]")
    print("P2 :", p2_csv)
    print("RHG:", rhg_csv)

    print("\n[leader cost function used for comparison]")
    print(f"J = {W_GAP}*gap + {W_RAMP}*ramp + {W_VAR}*variation + {W_PB}*sum(PB)")

    print("\n[summary]")
    print(summary.to_string(index=False))

    print("\n[done]")
    print("Hourly CSV :", out_dir / OUT_HOURLY)
    print("Summary CSV:", out_dir / OUT_SUMMARY)
    print("Figure     :", out_dir / OUT_FIG)


if __name__ == "__main__":
    main()