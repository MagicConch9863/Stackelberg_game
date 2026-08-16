"""S3: one leader/four followers, single-level Big-M KKT reformulation."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

T, DT = 96, 0.25
BUS_ORDER = ("Bus 1", "Bus 2", "Bus 3", "Bus 4")
BUS_CONFIG = {
    "Bus 1": {"bess_kwh": 5.0, "bess_kw": 1.0, "ev_kwh": 40.0,
              "ev_kw_ch": 3.6, "ev_kw_dis": 1.0, "ev_trip_kwh": 4.0},
    "Bus 2": {"bess_kwh": 4.0, "bess_kw": 0.8},
    "Bus 3": {"bess_kwh": 6.0, "bess_kw": 1.2},
    "Bus 4": {"bess_kwh": 6.0, "bess_kw": 1.2, "ev_kwh": 40.0,
              "ev_kw_ch": 3.6, "ev_kw_dis": 1.2, "ev_trip_kwh": 5.0},
}


def resolve_input_file(filename: str) -> Path:
    path = Path(filename)
    if path.is_file():
        return path
    script = Path(__file__).resolve()
    for folder in (script.parent, *script.parents):
        candidate = folder / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot find input file: {filename}")


def read_series(filename: str, column: str, scale: float = 1.0):
    path = resolve_input_file(filename)
    frame = pd.read_csv(path)
    if column not in frame:
        raise KeyError(f"{path.name} has no '{column}'. Columns: {list(frame.columns)}")
    values = pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(float)*scale
    if not len(values):
        raise ValueError(f"No numeric values in {path.name}:{column}")
    if len(values) != T:
        values = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, len(values)), values)
    return values


def load_inputs(settings):
    # C-A remains DER-only: no SimBench and no fixed base load.
    loads = {bus: np.zeros(T) for bus in BUS_ORDER}
    pv = {bus: np.zeros(T) for bus in BUS_ORDER}
    pa_scale = .001 if settings["PRICE_UNIT"].upper() == "EUR/MWH" else 1.0
    pa = read_series(settings["PRICE_FILE"], settings["PRICE_COLUMN"], pa_scale)
    return loads, pv, pa


def trip_vector(bus):
    trip = np.zeros(T)
    if "ev_kwh" in BUS_CONFIG[bus]:
        slots = np.r_[np.arange(28, 32), np.arange(68, 72)]
        trip[slots] = BUS_CONFIG[bus]["ev_trip_kwh"]/len(slots)
    return trip


def output_folders():
    root = Path(__file__).resolve().parent
    csv_folder, figure_folder = root/"csv_results", root/"figures"
    csv_folder.mkdir(exist_ok=True); figure_folder.mkdir(exist_ok=True)
    return csv_folder, figure_folder


def save_timeseries(path, loads, pv, result, pa, pb):
    def clean(values):
        values = np.asarray(values, dtype=float).copy()
        values[np.abs(values) < 1e-8] = 0.0
        return np.round(values, 8)

    pa, pb = clean(pa), clean(pb)
    data = {"time_h": np.arange(T)*DT, "pa_eur_per_kwh": pa,
            "pb_eur_per_kwh": pb, "total_buy_price_eur_per_kwh": pa+pb}
    aggregate = np.zeros(T)
    for bus in BUS_ORDER:
        key = bus.lower().replace(" ", "_"); item = result[bus]
        data[f"{key}_load_kw"] = loads[bus]; data[f"{key}_pv_kw"] = pv[bus]
        for field in ("bess", "ev", "import", "export", "net"):
            data[f"{key}_{field}_kw"] = clean(item[field])
        data[f"{key}_bess_energy_kwh"] = clean(item["bess_energy"][:T])
        if "ev_energy" in item:
            data[f"{key}_ev_energy_kwh"] = clean(item["ev_energy"][:T])
        aggregate += item["net"]
    data["aggregate_net_kw"] = clean(aggregate)
    pd.DataFrame(data).to_csv(path, index=False)
    return aggregate


def plot_outputs(folder, result, pa, pb):
    time = np.arange(T)*DT
    aggregate = sum(result[bus]["net"] for bus in BUS_ORDER)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(time, aggregate, color="black", linewidth=2)
    ax[0].axhline(0, color="grey", linewidth=.8)
    ax[0].set(title="Aggregate net exchange of four followers", ylabel="kW")
    ax[1].step(time, pa, where="post", color="black", label="PA")
    ax[1].step(time, pb, where="post", color="tab:orange", label="PB")
    ax[1].step(time, pa+pb, where="post", color="tab:red", label="PA + PB")
    ax[1].set(title="Common prices for all followers", ylabel="EUR/kWh", xlabel="Time [h]")
    ax[1].legend(ncol=3)
    for a in ax: a.grid(alpha=.3)
    fig.suptitle("S3: Stackelberg coordination", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(folder/"s3_aggregate.png", dpi=180)
    plt.close(fig)

    for number, bus in enumerate(BUS_ORDER, 1):
        item, cfg = result[bus], BUS_CONFIG[bus]
        fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        ax[0, 0].plot(time, item["net"], label="net")
        ax[0, 0].plot(time, item["import"], label="import")
        ax[0, 0].plot(time, -item["export"], label="- export")
        ax[0, 0].set_title("Grid exchange"); ax[0, 0].legend()
        ax[0, 1].step(time, item["bess"], where="post", label="BESS")
        if "ev_kwh" in cfg: ax[0, 1].step(time, item["ev"], where="post", label="EV")
        ax[0, 1].set_title("Device power (+ charge, - discharge)"); ax[0, 1].legend()
        ax[1, 0].plot(time, item["bess_energy"][:T], label="BESS energy")
        if "ev_energy" in item: ax[1, 0].plot(time, item["ev_energy"][:T], label="EV energy")
        ax[1, 0].set_title("Stored energy"); ax[1, 0].legend()
        ax[1, 1].step(time, pa, where="post", label="PA", color="black")
        ax[1, 1].step(time, pb, where="post", label="PB", color="tab:orange")
        ax[1, 1].step(time, pa+pb, where="post", label="PA + PB", color="tab:red")
        ax[1, 1].set_title("Common prices"); ax[1, 1].legend()
        for a in ax.flat:
            a.axhline(0, color="grey", linewidth=.6); a.grid(alpha=.3)
            a.set_xlabel("Time [h]")
        fig.suptitle(f"S3: Stackelberg coordination – {bus}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(folder/f"s3_bus{number}.png", dpi=180)
        plt.close(fig)


def follower_lp(bus: str, load: np.ndarray, pv: np.ndarray, grid_bound: float):
    """Build Aeq*x=beq, Aub*x<=bub and bounds for one follower.

    x = [BESS_power, EV_power, import, export].
    """
    cfg = BUS_CONFIG[bus]; q = 4*T
    aeq, beq, aub, bub = [], [], [], []
    for t in range(T):
        row = np.zeros(q)
        row[t], row[T+t], row[2*T+t], row[3*T+t] = 1, 1, -1, 1
        aeq.append(row); beq.append(-(load[t]-pv[t]))
    row = np.zeros(q); row[:T] = DT
    aeq.append(row); beq.append(0.0)  # terminal BESS energy = initial
    if "ev_kwh" in cfg:
        row = np.zeros(q); row[T:2*T] = DT
        aeq.append(row); beq.append(cfg["ev_trip_kwh"])

    # Cumulative energy bounds.
    for k in range(1, T+1):
        r = np.zeros(q); r[:k] = DT
        aub += [r, -r]
        bub += [.9*cfg["bess_kwh"]-.5*cfg["bess_kwh"],
                .5*cfg["bess_kwh"]-.1*cfg["bess_kwh"]]
    if "ev_kwh" in cfg:
        cumulative_trip = np.r_[0.0, np.cumsum(trip_vector(bus))]
        for k in range(1, T+1):
            r = np.zeros(q); r[T:T+k] = DT
            aub += [r, -r]
            bub += [.9*cfg["ev_kwh"]-.6*cfg["ev_kwh"]+cumulative_trip[k],
                    .6*cfg["ev_kwh"]-.2*cfg["ev_kwh"]-cumulative_trip[k]]

    lb = np.r_[np.full(T, -cfg["bess_kw"]), np.zeros(T),
               np.zeros(T), np.zeros(T)]
    ub = np.r_[np.full(T, cfg["bess_kw"]), np.zeros(T),
               np.full(T, grid_bound), np.full(T, grid_bound)]
    if "ev_kwh" in cfg:
        for t in range(T):
            hour = t*DT
            if hour < 7 or hour >= 18:
                lb[T+t], ub[T+t] = 0, cfg["ev_kw_ch"]
            elif 8 <= hour < 17:
                lb[T+t], ub[T+t] = -cfg["ev_kw_dis"], 0
    return tuple(map(np.asarray, (aeq, beq, aub, bub))) + (lb, ub)


def complementarity(m, slack_expr, slack_max, dual, name, dual_max):
    z = m.addVar(vtype=GRB.BINARY, name=f"z_{name}")
    slack = m.addVar(lb=0, ub=max(float(slack_max), 0.0), name=f"slack_{name}")
    m.addConstr(slack == slack_expr, name=f"slack_def_{name}")
    m.addConstr(slack <= max(float(slack_max), 0.0)*z, name=f"comp_slack_{name}")
    m.addConstr(dual <= dual_max*(1-z), name=f"comp_dual_{name}")


def add_follower_kkt(m, bus, load, pv, pa, pb, settings):
    Aeq, beq, Aub, bub, lb, ub = follower_lp(bus, load, pv, settings["GRID_BOUND_KW"])
    q = 4*T; tag = bus.replace(" ", "_")
    x = m.addVars(q, lb=-GRB.INFINITY, name=f"x_{tag}")
    lam = m.addVars(len(beq), lb=-GRB.INFINITY, name=f"lambda_{tag}")
    mu = m.addVars(len(bub), lb=0, ub=settings["DUAL_BIG_M"], name=f"mu_{tag}")

    # Primal feasibility and complementary slackness of general inequalities.
    for i, row in enumerate(Aeq):
        nz = np.flatnonzero(row)
        m.addConstr(gp.quicksum(row[j]*x[j] for j in nz) == beq[i], name=f"eq_{tag}_{i}")
    for i, row in enumerate(Aub):
        nz = np.flatnonzero(row)
        lhs = gp.quicksum(row[j]*x[j] for j in nz)
        m.addConstr(lhs <= bub[i], name=f"ineq_{tag}_{i}")
        minimum_lhs = sum(row[j]*(lb[j] if row[j] >= 0 else ub[j]) for j in nz)
        complementarity(m, bub[i]-lhs, bub[i]-minimum_lhs, mu[i],
                        f"ineq_{tag}_{i}", settings["DUAL_BIG_M"])

    # Bounds are explicit because their duals enter stationarity.
    alpha, beta = {}, {}
    for j in range(q):
        m.addConstr(x[j] >= lb[j]); m.addConstr(x[j] <= ub[j])
        alpha[j] = m.addVar(lb=0, ub=settings["DUAL_BIG_M"], name=f"dl_{tag}_{j}")
        beta[j] = m.addVar(lb=0, ub=settings["DUAL_BIG_M"], name=f"du_{tag}_{j}")
        span = ub[j]-lb[j]
        complementarity(m, x[j]-lb[j], span, alpha[j], f"lb_{tag}_{j}", settings["DUAL_BIG_M"])
        complementarity(m, ub[j]-x[j], span, beta[j], f"ub_{tag}_{j}", settings["DUAL_BIG_M"])

    # Stationarity: grad_x L = 0. PB appears only in the import coefficient.
    for j in range(q):
        if 2*T <= j < 3*T:
            t = j-2*T; cost = (pa[t]+pb[t])*DT
        elif 3*T <= j:
            t = j-3*T; cost = -pa[t]*DT
        else:
            cost = 0
        eq_terms = gp.quicksum(Aeq[i, j]*lam[i] for i in np.flatnonzero(Aeq[:, j]))
        ineq_terms = gp.quicksum(Aub[i, j]*mu[i] for i in np.flatnonzero(Aub[:, j]))
        m.addConstr(cost + eq_terms + ineq_terms - alpha[j] + beta[j] == 0,
                    name=f"stationarity_{tag}_{j}")
    return x


def main(settings: dict):
    csv_folder, figure_folder = output_folders()
    loads, pv, pa = load_inputs(settings)
    m = gp.Model("S3_one_leader_four_followers_KKT")
    m.Params.OutputFlag = settings["SOLVER_OUTPUT"]
    m.Params.MIPGap = settings["MIP_GAP"]
    m.Params.TimeLimit = settings["TIME_LIMIT_S"]
    # One common PB decision for every 15-minute interval (96 values/day).
    # All four followers see the same PB[t], but PB may change at every t.
    pb = m.addVars(T, lb=settings["PB_MIN"], ub=settings["PB_MAX"], name="PB")
    avg = gp.quicksum(pb[t] for t in range(T))/T
    m.addConstr(avg >= settings["PB_AVERAGE"]-settings["PB_AVERAGE_TOL"])
    m.addConstr(avg <= settings["PB_AVERAGE"]+settings["PB_AVERAGE_TOL"])
    for t in range(1, T):
        m.addConstr(pb[t]-pb[t-1] <= settings["PB_RAMP"])
        m.addConstr(pb[t-1]-pb[t] <= settings["PB_RAMP"])

    x = {}
    for bus in BUS_ORDER:
        x[bus] = add_follower_kkt(m, bus, loads[bus], pv[bus], pa, pb, settings)

    peak = m.addVar(lb=-GRB.INFINITY, name="aggregate_peak")
    valley = m.addVar(lb=-GRB.INFINITY, name="aggregate_valley")
    for t in range(T):
        aggregate = gp.quicksum(x[bus][2*T+t]-x[bus][3*T+t] for bus in BUS_ORDER)
        m.addConstr(peak >= aggregate); m.addConstr(valley <= aggregate)

    # Primary objective remains exactly the leader peak-valley objective.
    # The lower-priority tie-break only chooses a readable solution when many
    # follower-optimal responses have the same leader objective value.
    abs_device = {}
    movement = gp.LinExpr()
    for bus in BUS_ORDER:
        tag = bus.replace(" ", "_")
        for j in range(2*T):
            abs_device[bus, j] = m.addVar(lb=0, name=f"abs_device_{tag}_{j}")
            m.addGenConstrAbs(abs_device[bus, j], x[bus][j])
            movement += abs_device[bus, j]
    m.ModelSense = GRB.MINIMIZE
    m.setObjectiveN(peak-valley, index=0, priority=2, weight=1.0,
                    abstol=1e-8, name="leader_peak_valley")
    m.setObjectiveN(movement*DT, index=1, priority=1, weight=1.0,
                    name="readable_optimal_response")
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"S3 found no feasible incumbent; Gurobi status {m.Status}")

    result = {}
    for bus in BUS_ORDER:
        xv = np.array([x[bus][j].X for j in range(4*T)])
        bess, ev, imp, exp = np.split(xv, 4)
        cfg = BUS_CONFIG[bus]
        item = {"bess": bess, "ev": ev, "import": imp, "export": exp,
                "net": imp-exp,
                "bess_energy": .5*cfg["bess_kwh"] + np.r_[0.0, np.cumsum(bess*DT)]}
        if "ev_kwh" in cfg:
            item["ev_energy"] = .6*cfg["ev_kwh"] + np.r_[0.0, np.cumsum(ev*DT-trip_vector(bus))]
        result[bus] = item
    pb_values = np.array([pb[t].X for t in range(T)])
    agg = save_timeseries(csv_folder / "s3_timeseries.csv", loads, pv, result, pa, pb_values)
    plot_outputs(figure_folder, result, pa, pb_values)
    print(f"S3 complete: range={np.ptp(agg):.4f} kW; incumbent gap={m.MIPGap:.6f}")
    print(f"CSV: {csv_folder / 's3_timeseries.csv'}")
    print(f"Common PB values: {T} (one value per {DT*60:.0f}-minute interval)")
    total_price = pa + pb_values
    if np.ptp(total_price) < 1e-6:
        print("NOTE: PA+PB is constant. With DER-only zero base load, Bus 2/3 "
              "have no EV demand, so their minimum-movement optimal response is zero.")
    print(f"Figures: {figure_folder / 's3_aggregate.png'} and s3_bus1.png ... s3_bus4.png")


if __name__ == "__main__":
    SETTINGS = {
        "ENABLE_PV": False, "PV_FILE": "pv_2019-07-01_local.csv", "PV_COLUMN": "electricity",
        "PRICE_FILE": "price_data.csv", "PRICE_COLUMN": "price_eur_per_mwh_1", "PRICE_UNIT": "EUR/MWH",
        "PB_MIN": 0.0, "PB_MAX": 0.30, "PB_RAMP": 0.15,
        "PB_AVERAGE": 0.10, "PB_AVERAGE_TOL": 0.01,
        "GRID_BOUND_KW": 20.0, "DUAL_BIG_M": 1000.0,
        "MIP_GAP": 0.001, "TIME_LIMIT_S": 1200, "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
