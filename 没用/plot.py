import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 需要你改的地方
# ============================================================

OUT_DIR = r"outputs/bigm_kkt_no_revenue_clean/20260629_123502"
SAVE_NAME = "paper_style_performance.png"

# 选择哪几个结果画到上面的曲线图里
# 文件名按照你当前输出里的 csv 名字写
CURVE_FILES = {
    "Stackelberg (Proposed)": "static_K6_timeseries.csv",
    # "RHG": "rhg_K6_timeseries.csv",
    # "Heuristic Pricing": "static_K4_timeseries.csv",
}

# 如果你想把 rhg_K6 也画进去，就改成：
# CURVE_FILES = {
#     "RHG": "rhg_K6_timeseries.csv",
#     "Stackelberg (Proposed)": "static_K6_timeseries.csv",
# }

METRICS_FILE = "summary_metrics.csv"

# 选择哪些方法进入下面柱状图
BAR_CASES = [
    ("static", 1, "Static K=1"),
    ("static", 4, "Static K=4"),
    ("static", 6, "Static K=6"),
    ("static", 12, "Static K=12"),
    ("rhg", 1, "RHG K=1"),
    ("rhg", 4, "RHG K=4"),
    ("rhg", 6, "RHG K=6"),
    ("rhg", 12, "RHG K=12"),
]

# 下面柱状图展示哪些指标
BAR_METRICS = [
    ("Ramp red. %", "Ramp reduction"),
    ("Var red. %", "Variance reduction"),
    ("Peak red. %", "Peak reduction"),
]


# ============================================================
# 自动识别列名
# ============================================================

def find_col(df, candidates):
    lower_map = {c.lower().strip(): c for c in df.columns}
    for name in candidates:
        key = name.lower().strip()
        if key in lower_map:
            return lower_map[key]
    for c in df.columns:
        cl = c.lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    raise KeyError(f"Cannot find column from candidates: {candidates}\nExisting columns: {list(df.columns)}")


def load_timeseries(path):
    df = pd.read_csv(path)

    t_col = find_col(df, ["hour", "time", "t"])
    base_col = find_col(df, ["baseline", "simbench baseline", "base", "nominal"])
    net_col = find_col(df, ["net", "optimized net", "net_exchange", "N"])

    return df[t_col].to_numpy(), df[base_col].to_numpy(), df[net_col].to_numpy()


# ============================================================
# 读取曲线数据
# ============================================================

curves = {}
baseline_x = None
baseline_y = None

for label, file_name in CURVE_FILES.items():
    path = os.path.join(OUT_DIR, file_name)
    if not os.path.exists(path):
        print(f"Warning: file not found, skipped: {path}")
        continue

    x, base, net = load_timeseries(path)
    curves[label] = (x, net)

    if baseline_x is None:
        baseline_x = x
        baseline_y = base

if baseline_x is None:
    raise RuntimeError("No valid timeseries csv found. Please check CURVE_FILES and OUT_DIR.")


# ============================================================
# 读取性能指标
# ============================================================

metrics_path = os.path.join(OUT_DIR, METRICS_FILE)
metrics = pd.read_csv(metrics_path)

method_col = find_col(metrics, ["method"])
k_col = find_col(metrics, ["K"])

bar_labels = []
bar_values = {title: [] for _, title in BAR_METRICS}

for method, k, label in BAR_CASES:
    row = metrics[
        (metrics[method_col].astype(str).str.lower() == method.lower()) &
        (metrics[k_col].astype(int) == int(k))
    ]

    if row.empty:
        print(f"Warning: no metric row found for {method}, K={k}, skipped.")
        continue

    row = row.iloc[0]
    bar_labels.append(label)

    for col_name, title in BAR_METRICS:
        real_col = find_col(metrics, [col_name])
        bar_values[title].append(float(row[real_col]))


# ============================================================
# 绘图
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.28)

# ------------------------------------------------------------
# 上图：net exchange 曲线
# ------------------------------------------------------------

ax1 = fig.add_subplot(gs[0])

ax1.plot(
    baseline_x,
    baseline_y,
    linestyle="--",
    linewidth=2.0,
    color="0.65",
    label="Nominal Load"
)

for label, (x, y) in curves.items():
    ax1.plot(x, y, linewidth=2.6, label=label)

ax1.set_title("Load Smoothing Performance", fontsize=18, fontweight="bold", pad=12)
ax1.set_xlabel("Hour", fontsize=13)
ax1.set_ylabel("Total Power (kW)", fontsize=13)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper right", ncol=2, frameon=True, fontsize=12)

# ------------------------------------------------------------
# 下图：性能指标柱状图
# ------------------------------------------------------------

ax2 = fig.add_subplot(gs[1])

x = np.arange(len(BAR_METRICS))
width = 0.8 / max(len(bar_labels), 1)

for i, label in enumerate(bar_labels):
    values = [bar_values[title][i] for _, title in BAR_METRICS]
    offset = (i - (len(bar_labels) - 1) / 2) * width
    bars = ax2.bar(x + offset, values, width, label=label, alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + 1.0 if h >= 0 else h - 3.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom" if h >= 0 else "top",
            fontsize=9
        )

ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("Performance Metrics Quantification", fontsize=18, fontweight="bold", pad=12)
ax2.set_ylabel("Improvement (%)", fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([title for _, title in BAR_METRICS], fontsize=12)
ax2.grid(True, axis="y", linestyle="--", alpha=0.45)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=10)

plt.tight_layout()

save_path = os.path.join(OUT_DIR, SAVE_NAME)
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved to: {save_path}")