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

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
PHASE4 = PROJECT_ROOT / "stackelberg_scenario_analysis" / "phase4"
if str(PHASE4) not in sys.path:
    sys.path.insert(0, str(PHASE4))
from p4_common import day_inputs, load_simbench_profiles, read_pa_year, read_pv_source

DAYS = 7
SIM_STEPS = DAYS * 96
OUTPUT = ROOT / "t5_iid_mpc_mpc_no_terminal" / "results_7days"
PV_THRESHOLD = 0.01


def _load_data():
    settings = deepcopy(PHASE4_SETTINGS)
    settings.update({"LOAD_MODE": "annual", "SOLVER_OUTPUT": 1})
    pa_year, _ = read_pa_year(settings)
    pv_source = read_pv_source(settings)
    profiles, full_days = load_simbench_profiles(settings)
    parts = []
    # Seven evaluated days plus one complete forecast-buffer day.
    for offset in range(DAYS + 1):
        date = START_DATE + pd.Timedelta(days=offset)
        parts.append(day_inputs(date, profiles, full_days, pa_year, pv_source, settings))
    actual = {v: {b: np.concatenate([p[0 if v == "load" else 1][b] for p in parts])
                  for b in BUS_ORDER} for v in ("load", "pv")}
    for bus in BUS_ORDER:
        actual["pv"][bus][actual["pv"][bus] < PV_THRESHOLD] = 0.0
    pa = np.concatenate([p[2] for p in parts])
    timestamps = pd.date_range(START_DATE, periods=len(pa), freq="15min")
    return actual, pa, timestamps


def _plot(frame):
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":9})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    x = frame.step * DT
    observed_hours = (float(frame.step.max()) + 1.0) * DT
    axes[0,0].plot(x,frame.uncontrolled_net_kw,color="#777777",ls=":",lw=.9,label="Uncontrolled")
    axes[0,0].plot(x,frame.realized_net_kw,color="#0072B2",lw=1,label="Controlled")
    axes[0,0].set_title("Grid exchange"); axes[0,0].legend(frameon=False)
    axes[0,1].plot(x,frame.forecast_net_kw,color="#E69F00",ls="--",lw=.9,label="Updated forecast")
    axes[0,1].plot(x,frame.uncontrolled_net_kw,color="#0072B2",lw=.9,label="Realization")
    axes[0,1].set_title("Current updated prediction"); axes[0,1].legend(frameon=False)
    axes[1,0].plot(x,frame.bess_kw,color="#009E73",lw=.9); axes[1,0].axhline(0,color=".4",lw=.5)
    axes[1,0].set_title("Aggregate BESS action")
    axes[1,1].step(x,frame.pb_eur_per_kwh,where="post",color="#CC79A7",lw=.9)
    axes[1,1].set_title("Leader tariff PB")
    for ax in axes.flat:
        ax.set_xlabel("Time (h)"); ax.set_xlim(0,observed_hours); ax.grid(True,ls="--",alpha=.3,lw=.5)
    axes[0,0].set_ylabel("Power (kW)"); axes[0,1].set_ylabel("Net load (kW)")
    axes[1,0].set_ylabel("Power (kW)"); axes[1,1].set_ylabel("PB (EUR/kWh)")
    fig.suptitle(f"T5 IID without terminal SOC: observed {observed_hours:.1f} h",fontsize=13)
    fig.tight_layout(); fig.savefig(OUTPUT/"stackelberg_mpc_overview_partial.png",dpi=300,bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2,3.2))
    colors=("#0072B2","#E69F00","#009E73","#CC79A7")
    for bus,color in zip(BUS_ORDER,colors):
        col=f"soc_{bus.replace(' ','_').lower()}_kwh"
        ax.plot(x,frame[col],color=color,lw=1,label=bus)
        ax.axhline(BUS_CONFIG[bus]["bess_e_min"],color=color,ls=":",lw=.6,alpha=.65)
    ax.set(xlabel="Time (h)",ylabel="Stored energy (kWh)",xlim=(0,observed_hours),title="SOC sustainability without daily terminal constraint")
    ax.grid(True,ls="--",alpha=.3,lw=.5); ax.legend(frameon=False,ncol=4)
    fig.tight_layout(); fig.savefig(OUTPUT/"soc_sustainability_partial.png",dpi=300,bbox_inches="tight"); plt.close(fig)


def finalize_partial():
    path = OUTPUT / "mpc_timeseries.partial.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        raise RuntimeError("The partial checkpoint contains no completed MPC steps.")
    observed_hours = len(frame) * DT
    frame.to_csv(OUTPUT / "mpc_timeseries_observed.csv", index=False)
    metrics = _metrics("t5_no_terminal_partial", frame)
    metrics.loc[metrics["period"] == "full_48h", "period"] = f"observed_{observed_hours:g}h"
    for bus in BUS_ORDER:
        col = f"soc_{bus.replace(' ', '_').lower()}_kwh"
        metrics[f"final_{col}"] = float(frame[col].iloc[-1])
    metrics.to_csv(OUTPUT / "period_metrics_observed.csv", index=False)
    _plot(frame)
    pd.DataFrame([{
        "requested_steps": SIM_STEPS,
        "completed_steps": len(frame),
        "completion_fraction": len(frame) / SIM_STEPS,
        "observed_hours": observed_hours,
        "last_timestamp": frame["timestamp"].iloc[-1],
        "status": "stopped_by_user_partial_result",
    }]).to_csv(OUTPUT / "run_status.csv", index=False)
    print(f"Partial outputs generated from {len(frame)}/{SIM_STEPS} steps ({observed_hours:.1f} h).", flush=True)


def run():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    print("[T5 no-terminal] Loading 7 days plus forecast buffer...",flush=True)
    actual,pa,timestamps=_load_data()
    series_path=OUTPUT/"mpc_timeseries.partial.csv"; solver_path=OUTPUT/"solver_log.partial.csv"
    if series_path.exists():
        old=pd.read_csv(series_path,parse_dates=["timestamp"]); series=old.to_dict("records"); start=len(old)
        if start and not np.array_equal(old.step.to_numpy(),np.arange(start)): raise RuntimeError("Invalid checkpoint")
        energy={b:float(old.iloc[-1][f"soc_{b.replace(' ','_').lower()}_kwh"]) for b in BUS_ORDER}
        previous_pb=float(old.iloc[-1].pb_eur_per_kwh) if start else None
        solver=pd.read_csv(solver_path,parse_dates=["timestamp"]).to_dict("records") if solver_path.exists() else []
        print(f"[T5 no-terminal] Resuming from {start+1}/{SIM_STEPS}",flush=True)
    else:
        series=[]; solver=[]; start=0; previous_pb=None
        energy={b:float(BUS_CONFIG[b]["bess_e_init"]) for b in BUS_ORDER}
    warm_pb=None; started=time.time()
    for origin in range(start,SIM_STEPS):
        fc=_forecast(actual,"iid",origin)
        limit=180.0 if origin<2 else 600.0
        print(f"\n[T5 no-terminal] MPC {origin+1:03d}/{SIM_STEPS} | {timestamps[origin]} | limit={limit:.0f}s",flush=True)
        sol=_solve_stackelberg_window(timestamps[origin],origin,fc,energy,pa[origin:origin+HORIZON],
            previous_pb,None,warm_pb=warm_pb,mip_gap=.05,time_limit_s=limit,enforce_daily_terminal=False)
        bess=sum(sol["actions"].values()); base=sum(actual["load"][b][origin]-actual["pv"][b][origin] for b in BUS_ORDER)
        for b in BUS_ORDER: energy[b]+=DT*sol["actions"][b]
        forecast_net=sum(fc["load"][b][0]-fc["pv"][b][0] for b in BUS_ORDER)
        series.append({"step":origin,"timestamp":timestamps[origin],"uncontrolled_net_kw":base,"forecast_net_kw":forecast_net,
            "realized_net_kw":base+bess,"bess_kw":bess,"pb_eur_per_kwh":sol["pb"][0],
            **{f"soc_{b.replace(' ','_').lower()}_kwh":energy[b] for b in BUS_ORDER}})
        solver.append({"step":origin,"timestamp":timestamps[origin],**{k:sol[k] for k in ("status","objective","best_bound","gap","runtime_s","nodes")}})
        previous_pb=float(sol["pb"][0]); warm_pb=np.r_[sol["pb"][1:],sol["pb"][-1]]
        print(f"[T5 no-terminal] executed net={base+bess:.3f} kW, BESS={bess:+.3f} kW, gap={100*sol['gap']:.2f}%, runtime={sol['runtime_s']:.1f}s",flush=True)
        pd.DataFrame(series).to_csv(series_path,index=False); pd.DataFrame(solver).to_csv(solver_path,index=False)
    frame=pd.DataFrame(series); frame.to_csv(OUTPUT/"mpc_timeseries.csv",index=False)
    pd.DataFrame(solver).to_csv(OUTPUT/"solver_log.csv",index=False)
    metrics=_metrics("t5_no_terminal",frame)
    for b in BUS_ORDER:
        col=f"soc_{b.replace(' ','_').lower()}_kwh"; metrics[f"final_{col}"]=float(frame[col].iloc[-1])
    metrics.to_csv(OUTPUT/"period_metrics.csv",index=False); _plot(frame)
    print(f"[T5 no-terminal] Complete in {(time.time()-started)/60:.1f} min -> {OUTPUT}",flush=True)


if __name__=="__main__": run()
