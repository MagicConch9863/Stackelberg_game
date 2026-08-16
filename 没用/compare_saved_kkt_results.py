import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Adjustable settings
# ============================================================

CONFIG = {
    # 如果 run_dir = None，会自动读取 root_dir 下最新的一个结果文件夹
    "root_dir": "outputs/auto_kkt_price_modes_bus_configurable",
    "run_dir": None,

    # 想比较哪些模式
    "modes": ["free", "binary", "ternary"],

    # 想比较哪些 K
    "K_list": [1, 2, 4, 6],

    # 重点横向比较 K=4 和 K=6
    "focus_K": [4, 6],

    # 指标计算用的峰谷分位数
    "peak_q": 0.70,
    "valley_q": 0.30,

    # 输出文件夹
    "output_subdir": "comparison_figures",
}


# ============================================================
# 2. Load results
# ============================================================

def find_latest_run(root_dir):
    folders = [
        p for p in glob.glob(os.path.join(root_dir, "*"))
        if os.path.isdir(p)
    ]

    if not folders:
        raise RuntimeError(f"No result folders found under: {root_dir}")

    folders.sort(key=os.path.getmtime)
    return folders[-1]


def load_case(run_dir, mode, K):
    path = os.path.join(run_dir, f"{mode}_K{K}_timeseries.csv")

    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None

    df = pd.read_csv(path)

    required = ["hour", "baseline", "net", "import", "PB"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"{path} missing column: {c}")

    df["mode"] = mode
    df["K"] = K

    return df


def load_all_cases(cfg):
    run_dir = cfg["run_dir"]

    if run_dir is None:
        run_dir = find_latest_run(cfg["root_dir"])

    print("Reading result folder:")
    print(run_dir)

    cases = {}

    for mode in cfg["modes"]:
        for K in cfg["K_list"]:
            df = load_case(run_dir, mode, K)
            if df is not None:
                cases[(mode, K)] = df

    if not cases:
        raise RuntimeError("No valid case csv files were found.")

    out_dir = os.path.join(run_dir, cfg["output_subdir"])
    os.makedirs(out_dir, exist_ok=True)

    return run_dir, out_dir, cases


# ============================================================
# 3. Metrics
# ============================================================

def calc_metrics(df, peak_q=0.70, valley_q=0.30):
    baseline = df["baseline"].to_numpy(dtype=float)
    net = df["net"].to_numpy(dtype=float)
    imp = df["import"].to_numpy(dtype=float)

    eps = 1e-9

    peak_mask = baseline >= np.quantile(baseline, peak_q)
    valley_mask = baseline <= np.quantile(baseline, valley_q)

    peak_shave = 100.0 * (
        1.0 - np.mean(imp[peak_mask]) / (np.mean(baseline[peak_mask]) + eps)
    )

    valley_fill = 100.0 * (
        np.mean(net[valley_mask]) / (np.mean(baseline[valley_mask]) + eps) - 1.0
    )

    ramp_reduction = 100.0 * (
        1.0 - np.sum(np.diff(net) ** 2) / (np.sum(np.diff(baseline) ** 2) + eps)
    )

    variance_reduction = 100.0 * (
        1.0
        - np.sum((net - np.mean(net)) ** 2)
        / (np.sum((baseline - np.mean(baseline)) ** 2) + eps)
    )

    global_peak_reduction = 100.0 * (
        1.0 - np.max(net) / (np.max(baseline) + eps)
    )

    global_valley_increase = 100.0 * (
        np.min(net) / (np.min(baseline) + eps) - 1.0
    )

    mean_abs_change = np.mean(np.abs(net - baseline))

    return {
        "peak_shave_percent": peak_shave,
        "valley_fill_percent": valley_fill,
        "ramp_reduction_percent": ramp_reduction,
        "variance_reduction_percent": variance_reduction,
        "global_peak_reduction_percent": global_peak_reduction,
        "global_valley_increase_percent": global_valley_increase,
        "mean_abs_net_change_kw": mean_abs_change,
        "baseline_peak_kw": np.max(baseline),
        "optimized_peak_kw": np.max(net),
        "baseline_valley_kw": np.min(baseline),
        "optimized_valley_kw": np.min(net),
    }


def build_metric_table(cfg, cases):
    rows = []

    for (mode, K), df in cases.items():
        m = calc_metrics(df, cfg["peak_q"], cfg["valley_q"])
        m["mode"] = mode
        m["K"] = K
        rows.append(m)

    metric_df = pd.DataFrame(rows)
    metric_df = metric_df.sort_values(["mode", "K"]).reset_index(drop=True)

    return metric_df


# ============================================================
# 4. Plot: net exchange inside each mode
# ============================================================

def plot_net_by_mode(cfg, out_dir, cases):
    modes = cfg["modes"]
    K_list = cfg["K_list"]

    fig, axes = plt.subplots(len(modes), 1, figsize=(14, 4 * len(modes)), sharex=True)

    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        plotted_baseline = False

        for K in K_list:
            key = (mode, K)

            if key not in cases:
                continue

            df = cases[key]
            h = df["hour"].to_numpy()
            baseline = df["baseline"].to_numpy()
            net = df["net"].to_numpy()

            if not plotted_baseline:
                ax.plot(h, baseline, "--", color="0.35", linewidth=2.2, label="Baseline")
                plotted_baseline = True

            ax.plot(h, net, linewidth=2.0, label=f"K={K}")

        ax.set_title(f"{mode}: net exchange for different K")
        ax.set_ylabel("kW")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5)

    axes[-1].set_xlabel("Hour")

    fig.tight_layout()
    path = os.path.join(out_dir, "01_net_by_mode.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return path


# ============================================================
# 5. Plot: PB inside each mode
# ============================================================

def plot_pb_by_mode(cfg, out_dir, cases):
    modes = cfg["modes"]
    K_list = cfg["K_list"]

    fig, axes = plt.subplots(len(modes), 1, figsize=(14, 3.5 * len(modes)), sharex=True)

    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        for K in K_list:
            key = (mode, K)

            if key not in cases:
                continue

            df = cases[key]
            h = df["hour"].to_numpy()
            pb = df["PB"].to_numpy()

            ax.step(h, pb, where="post", linewidth=2.0, label=f"K={K}")

        ax.set_title(f"{mode}: optimized PB for different K")
        ax.set_ylabel("EUR/kWh")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5)

    axes[-1].set_xlabel("Hour")

    fig.tight_layout()
    path = os.path.join(out_dir, "02_pb_by_mode.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return path


# ============================================================
# 6. Plot: mode comparison for K=4 and K=6
# ============================================================

def plot_mode_compare_focus_K(cfg, out_dir, cases):
    modes = cfg["modes"]
    focus_K = cfg["focus_K"]

    fig, axes = plt.subplots(len(focus_K), 2, figsize=(16, 4.5 * len(focus_K)), sharex="col")

    if len(focus_K) == 1:
        axes = np.array([axes])

    for row, K in enumerate(focus_K):
        ax_net = axes[row, 0]
        ax_pb = axes[row, 1]

        plotted_baseline = False

        for mode in modes:
            key = (mode, K)

            if key not in cases:
                continue

            df = cases[key]
            h = df["hour"].to_numpy()
            baseline = df["baseline"].to_numpy()
            net = df["net"].to_numpy()
            pb = df["PB"].to_numpy()

            if not plotted_baseline:
                ax_net.plot(h, baseline, "--", color="0.35", linewidth=2.2, label="Baseline")
                plotted_baseline = True

            ax_net.plot(h, net, linewidth=2.0, label=mode)
            ax_pb.step(h, pb, where="post", linewidth=2.0, label=mode)

        ax_net.set_title(f"K={K}: net exchange comparison")
        ax_net.set_ylabel("kW")
        ax_net.grid(alpha=0.25)
        ax_net.legend()

        ax_pb.set_title(f"K={K}: PB comparison")
        ax_pb.set_ylabel("EUR/kWh")
        ax_pb.grid(alpha=0.25)
        ax_pb.legend()

    axes[-1, 0].set_xlabel("Hour")
    axes[-1, 1].set_xlabel("Hour")

    fig.tight_layout()
    path = os.path.join(out_dir, "03_mode_compare_K4_K6.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return path


# ============================================================
# 7. Plot: metric heatmaps
# ============================================================

def metric_matrix(metric_df, modes, K_list, metric_name):
    mat = np.full((len(modes), len(K_list)), np.nan)

    for i, mode in enumerate(modes):
        for j, K in enumerate(K_list):
            row = metric_df[(metric_df["mode"] == mode) & (metric_df["K"] == K)]

            if len(row) > 0:
                mat[i, j] = float(row.iloc[0][metric_name])

    return mat


def plot_metrics_heatmap(cfg, out_dir, metric_df):
    metrics = [
        ("peak_shave_percent", "Peak shave (%)"),
        ("valley_fill_percent", "Valley fill (%)"),
        ("global_peak_reduction_percent", "Global peak reduction (%)"),
        ("ramp_reduction_percent", "Ramp reduction (%)"),
        ("variance_reduction_percent", "Variance reduction (%)"),
        ("mean_abs_net_change_kw", "Mean abs. net change (kW)"),
    ]

    modes = cfg["modes"]
    K_list = cfg["K_list"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    axes = axes.ravel()

    for ax, (metric_name, title) in zip(axes, metrics):
        mat = metric_matrix(metric_df, modes, K_list, metric_name)

        im = ax.imshow(mat, aspect="auto")

        ax.set_title(title)
        ax.set_xticks(np.arange(len(K_list)))
        ax.set_xticklabels([f"K={k}" for k in K_list])
        ax.set_yticks(np.arange(len(modes)))
        ax.set_yticklabels(modes)

        for i in range(len(modes)):
            for j in range(len(K_list)):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Performance metrics: mode × K", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(out_dir, "04_metrics_heatmap.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return path


# ============================================================
# 8. Plot: focused metric bars for K=4 / K=6
# ============================================================

def plot_metrics_bar_focus_K(cfg, out_dir, metric_df):
    focus_K = cfg["focus_K"]
    modes = cfg["modes"]

    metrics = [
        ("peak_shave_percent", "Peak shave"),
        ("valley_fill_percent", "Valley fill"),
        ("global_peak_reduction_percent", "Global peak reduction"),
        ("ramp_reduction_percent", "Ramp reduction"),
        ("variance_reduction_percent", "Variance reduction"),
    ]

    fig, axes = plt.subplots(len(focus_K), 1, figsize=(14, 4.5 * len(focus_K)))

    if len(focus_K) == 1:
        axes = [axes]

    for ax, K in zip(axes, focus_K):
        x = np.arange(len(metrics))
        width = 0.8 / len(modes)

        for i, mode in enumerate(modes):
            row = metric_df[(metric_df["mode"] == mode) & (metric_df["K"] == K)]

            if len(row) == 0:
                continue

            vals = [float(row.iloc[0][m[0]]) for m in metrics]

            ax.bar(
                x + (i - (len(modes) - 1) / 2) * width,
                vals,
                width,
                label=mode,
            )

        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_title(f"K={K}: performance comparison")
        ax.set_ylabel("% vs baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics], rotation=15)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "05_metrics_bar_K4_K6.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return path


# ============================================================
# 9. Optional compact best table
# ============================================================

def print_best_cases(metric_df):
    print("\n=== Best cases by metric ===")

    targets = [
        ("peak_shave_percent", "max"),
        ("valley_fill_percent", "max"),
        ("global_peak_reduction_percent", "max"),
        ("ramp_reduction_percent", "max"),
        ("variance_reduction_percent", "max"),
        ("mean_abs_net_change_kw", "min"),
    ]

    for metric, direction in targets:
        if metric not in metric_df.columns:
            continue

        if direction == "max":
            idx = metric_df[metric].idxmax()
        else:
            idx = metric_df[metric].idxmin()

        row = metric_df.loc[idx]

        print(
            f"{metric}: {row['mode']} K={int(row['K'])}, "
            f"value={row[metric]:.3f}"
        )


# ============================================================
# 10. Main
# ============================================================

def main(cfg):
    run_dir, out_dir, cases = load_all_cases(cfg)

    metric_df = build_metric_table(cfg, cases)
    metric_path = os.path.join(out_dir, "comparison_metrics.csv")
    metric_df.to_csv(metric_path, index=False)

    saved = []
    saved.append(plot_net_by_mode(cfg, out_dir, cases))
    saved.append(plot_pb_by_mode(cfg, out_dir, cases))
    saved.append(plot_mode_compare_focus_K(cfg, out_dir, cases))
    saved.append(plot_metrics_heatmap(cfg, out_dir, metric_df))
    saved.append(plot_metrics_bar_focus_K(cfg, out_dir, metric_df))

    print_best_cases(metric_df)

    print("\nSaved comparison metrics:")
    print(metric_path)

    print("\nSaved figures:")
    for p in saved:
        print(p)

    print("\nDone.")


if __name__ == "__main__":
    main(CONFIG)