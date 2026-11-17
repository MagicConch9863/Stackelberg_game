"""OL-only solver initialization; leaves objective and constraints unchanged."""
import functools
import time
import numpy as np
from peak_window import core

gp, GRB = core.gp, core.GRB
Model = gp.Model


def follower_start(bus, baseline, energy, pa, prices, origin):
    aeq, beq, aub, bub, lb, ub = core._follower_lp(bus, baseline, energy, origin)[:6]
    h = len(pa)
    settings = core.PHASE4_SETTINGS
    epsilon = float(settings['FLOW_TIE_BREAK_EUR_PER_KWH'])
    costs = np.r_[np.zeros(h), core.DT*(pa+prices+epsilon),
                  core.DT*(-pa+epsilon),
                  np.full(2*h,settings['SOC_SOFT_PENALTY_EUR_PER_KWH']),
                  np.full(h,core.DT*settings['BESS_DEGRADATION_EUR_PER_KWH'])]
    lp = Model('OL_fixed_price_follower')
    try:
        lp.Params.OutputFlag = 0
        lp.Params.Threads = 1
        lp.Params.Method = 1
        x = lp.addVars(len(lb), lb=lb.tolist(), ub=ub.tolist())
        eq = [lp.addConstr(gp.quicksum(float(row[j])*x[j] for j in np.flatnonzero(row))==float(rhs))
              for row,rhs in zip(aeq,beq)]
        ine = [lp.addConstr(gp.quicksum(float(row[j])*x[j] for j in np.flatnonzero(row))<=float(rhs))
               for row,rhs in zip(aub,bub)]
        lp.setObjective(gp.quicksum(float(costs[j])*x[j] for j in range(len(lb))),GRB.MINIMIZE)
        lp.optimize()
        if lp.Status != GRB.OPTIMAL:
            raise RuntimeError(f'Fixed-price follower LP {bus}: status={lp.Status}')
        primal=np.array([x[j].X for j in range(len(lb))])
        reduced=np.array([x[j].RC for j in range(len(lb))])
        return dict(primal=primal, lam=-np.array([c.Pi for c in eq]),
                    mu=-np.array([c.Pi for c in ine]), lower=np.maximum(reduced,0),
                    upper=np.maximum(-reduced,0), aeq=aeq, beq=beq, aub=aub, bub=bub,
                    lb=lb, ub=ub, costs=costs, objective=float(lp.ObjVal))
    finally:
        lp.dispose()


def install_start(model, bus, start, x, dual_limit):
    tag=bus.replace(' ','_')
    model.update()
    for j,value in enumerate(start['primal']):x[j].Start=float(value)
    # LP multipliers use the opposite constraint sign to this KKT system.
    for i,value in enumerate(start['lam']):model.getVarByName(f'lam_{tag}[{i}]').Start=float(value)
    for i,value in enumerate(start['mu']):
        if -1e-7 <= value <= dual_limit+1e-7:
            model.getVarByName(f'mu_{tag}[{i}]').Start=float(np.clip(value,0,dual_limit))
        slack=max(0.,float(start['bub'][i]-start['aub'][i]@start['primal']))
        model.getVarByName(f'slack_ineq_{tag}_{i}').Start=slack
        model.getVarByName(f'z_ineq_{tag}_{i}').Start=1. if slack>1e-7 else 0.
    for j,value in enumerate(start['primal']):
        if start['ub'][j]-start['lb'][j] <= 1e-12:continue
        for prefix,slack,dual in [('lb',value-start['lb'][j],start['lower'][j]),
                                  ('ub',start['ub'][j]-value,start['upper'][j])]:
            model.getVarByName(f'slack_{prefix}_{tag}_{j}').Start=max(0.,float(slack))
            model.getVarByName(f'z_{prefix}_{tag}_{j}').Start=1. if slack>1e-7 else 0.
            if dual <= dual_limit+1e-7:
                model.getVarByName(f'{"dl" if prefix=="lb" else "du"}_{tag}_{j}').Start=float(min(dual,dual_limit))


class OLModel(Model):
    def optimize(self, *args, **kwargs):
        self.Params.MIPFocus=0
        self.Params.Heuristics=.50
        self.Params.NoRelHeurTime=60
        self.Params.PumpPasses=10
        self.Params.RINS=10
        print('[OL feasibility] MIPFocus=0 Heuristics=0.50 NoRelHeurTime=60 PumpPasses=10 RINS=10',flush=True)
        return super().optimize(*args,**kwargs)


def wrap(solve):
    @functools.wraps(solve)
    def run(date, origin, forecasts, energies, pa, previous_pb, current_hour_pb,
            warm_pb=None, **kwargs):
        prices=np.asarray(warm_pb if warm_pb is not None else
                          np.full(len(pa),previous_pb if previous_pb is not None else core.PHASE4_SETTINGS['PB_MIN']),float)
        tick=time.monotonic()
        starts={bus:follower_start(bus,forecasts['load'][bus]-forecasts['pv'][bus],
                                  energies[bus],np.asarray(pa),prices,origin) for bus in core.BUS_ORDER}
        original=core._add_follower_kkt
        def add(model,bus,baseline,energy,tariff,pb,step):
            result=original(model,bus,baseline,energy,tariff,pb,step)
            install_start(model,bus,starts[bus],result[0],result[3])
            return result
        old_model=gp.Model
        core._add_follower_kkt=add
        gp.Model=OLModel
        try:
            print(f'[OL feasibility] four fixed-price follower LPs solved in {time.monotonic()-tick:.3f}s',flush=True)
            return solve(date,origin,forecasts,energies,pa,previous_pb,current_hour_pb,
                         warm_pb=prices,**kwargs)
        finally:
            core._add_follower_kkt=original
            gp.Model=old_model
    return run
