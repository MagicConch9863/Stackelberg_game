from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_core.phase1_closed_loop import START_DATE, _forecast, _metrics
from experiment_core.stackelberg_mpc import (
    BUS_CONFIG, BUS_ORDER, DT, HORIZON, PHASE4_SETTINGS, _solve_stackelberg_window,
)
from experiment_core.ol_experiment import solve_day_with_feasibility_focus

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))
from p4_s3_stackelberg import solve_day as solve_stackelberg_day
from p4_common import day_inputs, load_simbench_profiles, read_pa_year, read_pv_source

OUTPUT = ROOT / "t5_perfect_soc_reference" / "results_1day"
SIM_STEPS = 96
SOC_BAND_FRAC = 0.15
FIRST_STEP_LIMIT_S = 180.0
USE_NET_ANCHOR = False


def _load_data():
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({"LOAD_MODE": "annual", "SOLVER_OUTPUT": 1})
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    profiles, full_days = load_simbench_profiles(settings)
    reference_days = SIM_STEPS // 96 + 1
    parts = []
    for offset in range(reference_days):
        date = START_DATE + pd.Timedelta(days=offset)
        parts.append(day_inputs(date, profiles, full_days, pa_year, pv_source, settings))
    actual = {v: {b: np.concatenate([p[0 if v == "load" else 1][b] for p in parts])
                  for b in BUS_ORDER} for v in ("load", "pv")}
    for bus in BUS_ORDER:
        actual["pv"][bus][actual["pv"][bus] < 0.01] = 0.0
    pa = np.concatenate([p[2] for p in parts])
    timestamps = pd.date_range(START_DATE, periods=len(pa), freq="15min")
    return actual, pa, timestamps


def _reference(actual, pa, timestamps):
    path = OUTPUT / "day_ahead_soc_reference.csv"
    if path.exists():
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        reference_days = SIM_STEPS // 96 + 1
        if len(frame) == reference_days * 97 * len(BUS_ORDER):
            return frame
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({"SOLVER_OUTPUT": 1, "TIME_LIMIT_S_PER_DAY": 1200, "MIP_GAP": 0.05})
    rows = []
    reference_days = SIM_STEPS // 96 + 1
    for day in range(reference_days):
        origin = day * 96
        fc = _forecast(actual, "perfect", origin, horizon=96)
        print(f"[SOC reference] OL day {day+1}/{reference_days}: {timestamps[origin].date()}", flush=True)
        started = time.perf_counter()
        result, pb, gap = solve_day_with_feasibility_focus(
            solve_stackelberg_day, timestamps[origin], fc["load"], fc["pv"],
            pa[origin:origin+96], settings)
        runtime = time.perf_counter() - started
        for bus in BUS_ORDER:
            energy = np.asarray(result[bus]["bess_energy"], float)
            for j in range(97):
                rows.append({"day":day,"energy_step":origin+j,"timestamp":timestamps[origin]+pd.Timedelta(minutes=15*j),
                             "bus":bus,"soc_reference_kwh":energy[j],"pb_eur_per_kwh":pb[min(j,95)],
                             "ol_gap":gap,"ol_runtime_s":runtime})
        pd.DataFrame(rows).to_csv(path,index=False)
        print(f"[SOC reference] saved day {day+1}: gap={100*gap:.2f}%, runtime={runtime:.1f}s",flush=True)
    return pd.DataFrame(rows)


def _plot(frame, refs):
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":9})
    x=frame.step*DT
    duration_h = SIM_STEPS * DT
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.2),sharex=True)
    axes[0,0].plot(x,frame.uncontrolled_net_kw,color="#777777",ls=":",lw=1,label="Uncontrolled")
    axes[0,0].plot(x,frame.realized_net_kw,color="#0072B2",lw=1.1,label="Controlled")
    axes[0,0].set_title("Grid exchange"); axes[0,0].legend(frameon=False)
    axes[0,1].plot(x,frame.forecast_net_kw,color="#E69F00",ls="--",lw=1,label="Perfect prediction")
    axes[0,1].plot(x,frame.uncontrolled_net_kw,color="#0072B2",lw=.9,label="Realization")
    axes[0,1].set_title("Prediction vs realization"); axes[0,1].legend(frameon=False)
    axes[1,0].plot(x,frame.bess_kw,color="#009E73",lw=1); axes[1,0].axhline(0,color=".4",lw=.5)
    axes[1,0].set_title("Aggregate BESS action")
    axes[1,1].step(x,frame.pb_eur_per_kwh,where="post",color="#CC79A7",lw=1)
    axes[1,1].set_title("Leader tariff PB")
    for ax in axes.flat:
        ax.set_xlabel("Time (h)"); ax.set_xlim(0,duration_h); ax.grid(True,ls="--",alpha=.3,lw=.5)
    axes[0,0].set_ylabel("Power (kW)"); axes[0,1].set_ylabel("Net load (kW)")
    axes[1,0].set_ylabel("Power (kW)"); axes[1,1].set_ylabel("PB (EUR/kWh)")
    anchor_label = " + previous-net anchor" if USE_NET_ANCHOR else ""
    fig.suptitle(f"T5-perfect with 15% day-ahead SOC reference{anchor_label}: {duration_h/24:g} days",fontsize=13)
    fig.tight_layout(); fig.savefig(OUTPUT/"stackelberg_mpc_overview.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,3.3)); colors=("#0072B2","#E69F00","#009E73","#CC79A7")
    for bus,color in zip(BUS_ORDER,colors):
        col=f"soc_{bus.replace(' ','_').lower()}_kwh"
        ax.plot(x,frame[col],color=color,lw=1.1,label=f"{bus} MPC")
        ref=refs[(refs.bus==bus)&(refs.energy_step.between(1,SIM_STEPS))].sort_values("energy_step")
        ax.plot((ref.energy_step-1)*DT,ref.soc_reference_kwh,color=color,ls="--",lw=.8,alpha=.75)
    ax.set(xlabel="Time (h)",ylabel="Stored energy (kWh)",xlim=(0,duration_h),title="MPC SOC (solid) and day-ahead reference (dashed)")
    ax.grid(True,ls="--",alpha=.3,lw=.5); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(OUTPUT/"soc_reference_tracking.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def run():
    raise RuntimeError(
        "Obsolete experiment: the SOC-reference/midnight formulation was removed. "
        "Use the soft_soc_20_80_v1 experiment entry points instead."
    )
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print("[T5-perfect SOC reference] Loading data...",flush=True)
    actual,pa,timestamps=_load_data()
    refs=_reference(actual,pa,timestamps)
    lookup={(r.bus,int(r.energy_step)):float(r.soc_reference_kwh) for r in refs.itertuples()}
    partial_series=OUTPUT/"mpc_timeseries.partial.csv"; partial_solver=OUTPUT/"solver_log.partial.csv"
    if partial_series.exists():
        old=pd.read_csv(partial_series,parse_dates=["timestamp"])
        start=len(old)
        if start and np.array_equal(old.step.to_numpy(),np.arange(start)):
            series=old.to_dict("records")
            energy={b:float(old.iloc[-1][f"soc_{b.replace(' ','_').lower()}_kwh"]) for b in BUS_ORDER}
            previous_pb=float(old.iloc[-1].pb_eur_per_kwh)
            solver=pd.read_csv(partial_solver,parse_dates=["timestamp"]).to_dict("records") if partial_solver.exists() else []
            print(f"[T5-perfect SOC reference] Resuming from {start+1}/{SIM_STEPS}",flush=True)
        else:
            raise RuntimeError("Invalid partial checkpoint")
    else:
        start=0; series=[]; solver=[]; previous_pb=None
        energy={b:float(BUS_CONFIG[b]["bess_e_init"]) for b in BUS_ORDER}
    warm_pb=None
    for origin in range(start,SIM_STEPS):
        fc=_forecast(actual,"perfect",origin)
        end=origin+HORIZON; bands={}
        for bus in BUS_ORDER:
            width=SOC_BAND_FRAC*(BUS_CONFIG[bus]["bess_e_max"]-BUS_CONFIG[bus]["bess_e_min"])
            center=lookup[(bus,end)]
            bands[bus]=(center-width,center+width)
        limit=FIRST_STEP_LIMIT_S if origin<2 else 600.0
        print(f"\n[T5-perfect SOC reference] MPC {origin+1:02d}/{SIM_STEPS} | {timestamps[origin]} | limit={limit:.0f}s",flush=True)
        previous_net = None if not USE_NET_ANCHOR or not series else float(series[-1]["realized_net_kw"])
        sol=_solve_stackelberg_window(timestamps[origin],origin,fc,energy,pa[origin:origin+HORIZON],previous_pb,None,
            warm_pb=warm_pb,mip_gap=.05,time_limit_s=limit,enforce_daily_terminal=True,
            terminal_soc_bands=bands,previous_realized_net=previous_net)
        bess=sum(sol["actions"].values()); base=sum(actual["load"][b][origin]-actual["pv"][b][origin] for b in BUS_ORDER)
        for bus in BUS_ORDER: energy[bus]+=DT*sol["actions"][bus]
        series.append({"step":origin,"timestamp":timestamps[origin],"uncontrolled_net_kw":base,"forecast_net_kw":base,
            "realized_net_kw":base+bess,"bess_kw":bess,"pb_eur_per_kwh":sol["pb"][0],
            **{f"soc_{b.replace(' ','_').lower()}_kwh":energy[b] for b in BUS_ORDER}})
        solver.append({"step":origin,"timestamp":timestamps[origin],**{k:sol[k] for k in ("status","objective","best_bound","gap","runtime_s","nodes")}})
        previous_pb=float(sol["pb"][0]); warm_pb=np.r_[sol["pb"][1:],sol["pb"][-1]]
        print(f"[T5-perfect SOC reference] net={base+bess:.3f} kW, BESS={bess:+.3f}, gap={100*sol['gap']:.2f}%, runtime={sol['runtime_s']:.1f}s",flush=True)
        pd.DataFrame(series).to_csv(partial_series,index=False)
        pd.DataFrame(solver).to_csv(partial_solver,index=False)
    frame=pd.DataFrame(series); frame.to_csv(OUTPUT/"mpc_timeseries.csv",index=False)
    pd.DataFrame(solver).to_csv(OUTPUT/"solver_log.csv",index=False)
    metrics = _metrics("t5_perfect_soc_reference", frame)
    metrics.loc[metrics["period"] == "full_48h", "period"] = f"full_{SIM_STEPS*DT:g}h"
    metrics.to_csv(OUTPUT/"period_metrics.csv",index=False)
    _plot(frame,refs); print(f"Complete -> {OUTPUT}",flush=True)


if __name__=="__main__": run()
