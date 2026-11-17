"""Fixed-price KKT start and one model per window, including no-incumbent rescue."""
import functools
import math
import time
from pathlib import Path

import numpy as np
from ol_feasibility import core, follower_start, install_start
from io_utils import atomic_json
from deadline_policy import accept_mpc_stage
from review_case import matching_candidate

gp, GRB = core.gp, core.GRB
OriginalModel = gp.Model


def wrap(solve, method, policy, output):
    output = Path(output)

    @functools.wraps(solve)
    def run(date, origin, forecasts, energies, pa, previous_pb, current_hour_pb,
            warm_pb=None, **kwargs):
        started = time.monotonic()
        saved_plan = kwargs.pop('saved_plan', None)
        target = kwargs.get('mip_gap', policy['target_gap']) if policy.get('strict_gap_review') else policy['target_gap']
        limits = policy.get('mpc_time_limits_s') if method == 'MPC' and saved_plan is None else None
        initial = policy['ol_window_s'] if method == 'OL' else policy['mpc_window_s']
        if limits:
            initial = limits[0]
        elif method == 'MPC' and saved_plan is not None:
            initial = policy.get('certificate_window_s', initial)
        total = initial if method == 'OL' else policy['mpc_no_incumbent_total_s']
        if method == 'MPC' and saved_plan is not None:
            total = policy.get('certificate_no_incumbent_total_s', total)
        candidate=(matching_candidate(output,origin,energies,len(pa))
                   if policy.get('strict_gap_review') else None)
        start_path=output/'latest_incumbent.mst'
        reuse_start=candidate is not None and start_path.exists()
        prices = np.asarray(warm_pb if warm_pb is not None else
                            np.full(len(pa), previous_pb if previous_pb is not None else
                                    core.PHASE4_SETTINGS['PB_MIN']), float)
        starts = {bus: follower_start(bus, forecasts['load'][bus]-forecasts['pv'][bus],
                    energies[bus], np.asarray(pa), prices, origin) for bus in core.BUS_ORDER}
        if saved_plan:
            for bus, start in starts.items():
                action = np.asarray(saved_plan[bus], float)
                net = forecasts['load'][bus]-forecasts['pv'][bus]+action
                energy = energies[bus]+core.DT*np.cumsum(action)
                cap = core.BUS_CONFIG[bus]['bess_e_max']
                primal = np.r_[action, np.maximum(net, 0), np.maximum(-net, 0),
                               np.maximum(.2*cap-energy, 0), np.maximum(energy-.8*cap, 0),
                               np.abs(action)]
                residual = max(np.max(np.abs(start['aeq']@primal-start['beq'])),
                    np.max(np.maximum(start['aub']@primal-start['bub'], 0)),
                    np.max(np.maximum(start['lb']-primal, 0)),
                    np.max(np.maximum(primal-start['ub'], 0)))
                if residual <= 3e-5 and abs(start['costs']@primal-start['objective']) <= 1e-6*max(1., abs(start['objective'])):
                    start['primal'] = primal
        stages, models = [], []

        class BudgetModel(OriginalModel):
            def __init__(self, *args, **options):
                super().__init__(*args, **options)
                models.append(self)

            def optimize(self, *args, **options):
                # Restore the execution policy after the legacy builder sets parameters.
                self.Params.MIPFocus = 0
                self.Params.Heuristics = .20
                self.Params.NoRelHeurTime = 0
                self.Params.PumpPasses = -1
                self.Params.RINS = -1
                self.Params.Seed = int(policy.get('solver_seed',0))
                if reuse_start:
                    self.read(str(start_path))
                    print(f'[quality_resume] origin={origin} full_MIP_start=True solver_seed={self.Params.Seed}',flush=True)
                if limits:
                    for index, budget in enumerate(limits):
                        if index and self.Status in (GRB.INFEASIBLE, GRB.UNBOUNDED):
                            break
                        self.Params.TimeLimit = max(.1, budget-(time.monotonic()-started)) if index == 0 else budget
                        if index == 2 and self.SolCount == 0:
                            self.Params.MIPFocus = 1
                            self.Params.Heuristics = .50
                        print(f'[mpc_budget] origin={origin} stage={index+1} limit={budget}s same_model=True', flush=True)
                        tick = time.monotonic()
                        super().optimize(*args, **options)
                        if self.Status == GRB.INF_OR_UNBD:
                            # Disambiguate the solver status without changing the model.
                            self.Params.DualReductions = 0
                        valid = bool(self.SolCount) and all(math.isfinite(v) for v in
                            (self.ObjVal, self.ObjBound, self.MIPGap))
                        if valid:
                            valid = max(self.ConstrVio, self.BoundVio, self.IntVio) <= policy.get('validation_tolerance', 3e-5)
                            valid &= self.ObjBound <= self.ObjVal+1e-7*max(1.,abs(self.ObjVal))
                        stages.append(dict(stage=f'attempt_{index+1}', limit_s=budget,
                            elapsed_s=time.monotonic()-tick, status=int(self.Status),
                            solution_count=int(self.SolCount), valid_incumbent=valid))
                        if accept_mpc_stage(index, valid, self.MIPGap if valid else math.inf,
                                            target, strict=policy.get('strict_gap_review', False)):
                            break
                else:
                    self.Params.TimeLimit = max(.1, initial-(time.monotonic()-started))
                    print(f'[deadline48] origin={origin} initial={initial}s total_no_incumbent={total}s MIPFocus=0', flush=True)
                    tick = time.monotonic()
                    super().optimize(*args, **options)
                    stages.append(dict(stage='initial', elapsed_s=time.monotonic()-tick,
                                       status=int(self.Status), solution_count=int(self.SolCount)))
                    remaining = total-(time.monotonic()-started)
                    if self.SolCount == 0 and remaining > .1 and self.Status not in (GRB.INFEASIBLE, GRB.UNBOUNDED):
                        self.Params.MIPFocus = 1
                        self.Params.Heuristics = .50
                        self.Params.TimeLimit = remaining
                        tick = time.monotonic()
                        super().optimize(*args, **options)
                        stages.append(dict(stage='no_incumbent_rescue', elapsed_s=time.monotonic()-tick,
                                           status=int(self.Status), solution_count=int(self.SolCount)))
                if self.SolCount and all(math.isfinite(v) for v in (self.ObjVal, self.ObjBound, self.MIPGap)):
                    self.write(str(output/'latest_incumbent.mst'))

        original_add = core._add_follower_kkt

        def add(model, bus, baseline, energy, tariff, pb, step):
            result = original_add(model, bus, baseline, energy, tariff, pb, step)
            install_start(model, bus, starts[bus], result[0], result[3])
            return result

        core._add_follower_kkt = add
        old_model = gp.Model
        gp.Model = BudgetModel
        try:
            kwargs.update(time_limit_s=initial, mip_gap=target)
            result = solve(date, origin, forecasts, energies, pa, previous_pb,
                           current_hour_pb, warm_pb=prices, **kwargs)
            result['deadline_stages'] = stages
            result['runtime_s'] = time.monotonic()-started
            return result
        finally:
            core._add_follower_kkt = original_add
            gp.Model = old_model
            atomic_json(output/'solver_execution.json', dict(origin=origin, method=method,
                policy=policy['version'], same_model_rescue=True, full_follower_kkt_start=True,
                resumed_incumbent=reuse_start,solver_seed=int(policy.get('solver_seed',0)),
                elapsed_s=time.monotonic()-started, stages=stages))
            for model in models:
                try:
                    model.dispose()
                except gp.GurobiError:
                    pass

    return run
