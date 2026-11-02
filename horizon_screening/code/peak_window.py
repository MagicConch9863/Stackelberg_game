"""Leader minimizes nonnegative import peak plus ramp; Follower KKT is unchanged."""
from pathlib import Path
import sys

CORE = Path(__file__).resolve().parents[1]/'source/forecast_mpc'
sys.path.insert(0, str(CORE))
from experiment_core import stackelberg_mpc as core
import numpy as np


def solve_window(date, origin, forecasts, energies, pa, previous_pb, current_hour_pb,
                 warm_pb=None, mip_gap=None, time_limit_s=None, previous_realized_net=None):
    gp, GRB, settings = core.gp, core.GRB, core.PHASE4_SETTINGS
    if settings.get('GRID_EXCHANGE_MODE') != 'grid_export':
        raise ValueError('This experiment requires signed grid exchange')
    horizon = len(pa)
    model = gp.Model(f'PeakRamp_{date:%Y%m%d}_{origin}')
    output = int(settings.get('SOLVER_OUTPUT',1))
    model.Params.OutputFlag = output
    model.Params.LogToConsole = output
    model.Params.DisplayInterval = int(settings.get('GUROBI_DISPLAY_INTERVAL_S',2))
    model.Params.MIPGap = float(settings['MIP_GAP'] if mip_gap is None else mip_gap)
    model.Params.TimeLimit = float(core.TIME_LIMIT_S_PER_UPDATE if time_limit_s is None else time_limit_s)
    for name, value in dict(NumericFocus=1,Heuristics=.05,NoRelHeurTime=0.0,PumpPasses=-1,
                            RINS=-1,MIPFocus=3,Cuts=3,Presolve=2,Symmetry=2,Seed=1,
                            Threads=int(settings.get('GUROBI_THREADS',1))).items():
        model.setParam(name,value)
    pb = model.addVars(horizon,lb=settings['PB_MIN'],ub=settings['PB_MAX'],name='PB')
    if warm_pb is not None:
        for h in range(min(horizon,len(warm_pb))):
            pb[h].Start = float(warm_pb[h])
    ramp_limit = float(settings['PB_RAMP'])
    for h in range(horizon):
        previous = previous_pb if h == 0 else pb[h-1]
        if previous is not None:
            model.addConstr(pb[h]-previous <= ramp_limit)
            model.addConstr(previous-pb[h] <= ramp_limit)
    x, bounds, duals, dual_bounds = {}, {}, {}, {}
    for bus in core.BUS_ORDER:
        baseline = forecasts['load'][bus]-forecasts['pv'][bus]
        x[bus],bounds[bus],duals[bus],dual_bounds[bus] = core._add_follower_kkt(
            model,bus,baseline,energies[bus],pa,pb,origin)
    # No valley variable, valley constraint or previous-grid peak/valley anchor.
    # The previous realized grid enters only the first ramp difference.
    peak = model.addVar(lb=0,name='import_peak')
    grid, ramp = {}, {}
    weight = float(settings.get('LEADER_RAMP_WEIGHT',0))
    for h in range(horizon):
        grid[h] = gp.quicksum(x[b][horizon+h]-x[b][2*horizon+h] for b in core.BUS_ORDER)
        model.addConstr(peak >= grid[h],name=f'peak_ge_grid_{h}')
        previous = previous_realized_net if h == 0 else grid[h-1]
        if previous is not None and weight > 0:
            ramp[h] = model.addVar(lb=0,name=f'ramp_abs_{h}')
            model.addConstr(ramp[h] >= grid[h]-previous)
            model.addConstr(ramp[h] >= previous-grid[h])
    model.setObjective(peak+weight*gp.quicksum(ramp.values()),GRB.MINIMIZE)
    model.optimize()
    if model.SolCount == 0:
        status = model.Status
        model.dispose()
        raise RuntimeError(f'{date.date()} step {origin}: no incumbent, status={status}')
    plans = {b:np.array([x[b][h].X for h in range(horizon)]) for b in core.BUS_ORDER}
    result = dict(actions={b:float(plans[b][0]) for b in core.BUS_ORDER}, bess_plans=plans,
        net_plans={b:np.array([x[b][horizon+h].X-x[b][2*horizon+h].X for h in range(horizon)]) for b in core.BUS_ORDER},
        pb=np.array([pb[h].X for h in range(horizon)]), status=int(model.Status),
        objective=float(model.ObjVal), best_bound=float(model.ObjBound), gap=float(model.MIPGap),
        runtime_s=float(model.Runtime), nodes=float(model.NodeCount), bounds=bounds,
        dual_big_m=max(dual_bounds.values()),
        max_dual_bound_fraction=max(d.X/dual_bounds[b] for b in core.BUS_ORDER for d in duals[b]),
        constraint_violation=float(model.ConstrVio), bound_violation=float(model.BoundVio),
        integrality_violation=float(model.IntVio), objective_kind='peak_ramp',
        peak_component=float(peak.X), ramp_component=sum(float(r.X) for r in ramp.values()),
        planned_grid=np.array([grid[h].getValue() for h in range(horizon)]))
    model.dispose()
    return result
