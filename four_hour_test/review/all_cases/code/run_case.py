"""Independent fixed-state window audits; never overwrite simulated trajectories."""
from pathlib import Path
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import math
import site
import sys
import time

site.sethelper()
BASE=Path(__file__).resolve().parents[1]
SHARED=BASE.parents[1]/'code'
sys.path.insert(0,str(SHARED))
from io_utils import atomic_json,atomic_text


def load_source(source):
    return json.loads((source/'checkpoint.json').read_text()),json.loads((source/'manifest.json').read_text())


def review_target(old_gap,reasons):
    if old_gap>.20:return .10
    if old_gap>.100001:return .02
    return .01 if any('ramp' in r for r in reasons) else .10


def inspect(state,meta):
    """Select every official high-gap window plus control-amplified jumps and peaks."""
    rows=state['rows']; logs={s['origin']:s for s in state['solves']}
    reasons={}; errors=[]
    def flag(origin,reason):
        if origin>=16 and origin in logs:
            reasons.setdefault(origin,set()).add(reason)
    for s in state['solves']:
        if s['gap']>.100001:flag(s['origin'],'gap_above_10pct')
    official=[r for r in rows if r['step']>=0]
    if official:
        peak=max(official,key=lambda r:r['realized_net_kw'])
        flag(peak['origin'],'case_peak')
    for r in rows:
        k=r['execution_step']
        if abs(r['realized_net_kw']-r['uncontrolled_net_kw']-r['bess_kw'])>1e-5:
            errors.append(dict(step=k,kind='grid_balance'))
        for i,cap,power in [(1,8.,2.5),(2,5.,1.5),(3,8.,2.5),(4,8.,2.5)]:
            before=r[f'energy_before_bus_{i}']; after=r[f'soc_bus_{i}_kwh']; u=r[f'bess_bus_{i}_kw']
            if not -3e-5<=after<=cap+3e-5 or abs(u)>power+3e-5 or abs(after-before-.25*u)>3e-5:
                errors.append(dict(step=k,bus=i,kind='physical_or_state_update'))
    for a,b in zip(rows,rows[1:]):
        for i in range(1,5):
            if abs(a[f'soc_bus_{i}_kwh']-b[f'energy_before_bus_{i}'])>3e-5:
                errors.append(dict(step=b['execution_step'],bus=i,kind='state_continuity'))
    for a,b in zip(official,official[1:]):
        dp=b['realized_net_kw']-a['realized_net_kw']; dx=b['uncontrolled_net_kw']-a['uncontrolled_net_kw']
        if abs(dp)>=2 and abs(dp)-abs(dx)>=1:
            flag(a['origin'],'before_amplified_ramp');flag(b['origin'],'amplified_ramp')
    scenario=meta['scenario']; method=meta['method']; rho=meta['config']['rho']; max_error=0.
    for f in state['forecasts']:
        correction=rho**(f['lead']+1)*f['measured_residual'] if method=='MPC' and scenario=='ar1' else 0.
        expected=max(0.,f['nominal_load']+correction)
        max_error=max(max_error,abs(expected-f['predicted_load']))
    if max_error>1e-8:errors.append(dict(kind='forecast_formula',max_error=max_error))
    return reasons,dict(physical_errors=errors,forecast_formula_max_error_kw=max_error,
        source_steps=state['next_step'],official_points=len(official))


def plan_values(grid,anchor,label,weight):
    import numpy as np
    values=np.asarray(grid,float)
    ramp=float(np.abs(np.diff(np.r_[anchor,values])).sum()) if anchor is not None else float(np.abs(np.diff(values)).sum())
    if label=='peak_ramp':
        objective=max(0.,float(max(values)))+weight*ramp
    else:
        extended=np.r_[anchor,values] if anchor is not None else values
        objective=float(np.ptp(extended))+weight*ramp
    return dict(objective=objective,peak_kw=float(max(values)),valley_kw=float(min(values)),
                total_ramp_kw=ramp,first_grid_kw=float(values[0]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','season','family','scenario','method','seed','source-root','source-objective'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--gap-tiered',action='store_true')
    parser.add_argument('--retry-unresolved',action='store_true')
    parser.add_argument('--independent-certificate',action='store_true')
    parser.add_argument('--second-pass',action='store_true')
    parser.add_argument('--jump-only',action='store_true')
    parser.add_argument('--third-pass',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('--execute required')
    import numpy as np
    core_path=BASE.parents[2]/'hpc_seasonal_actual_load_noise/src/forecast_mpc'
    sys.path.insert(0,str(core_path))
    from experiment_core import stackelberg_mpc as core
    if args.source_objective=='peak_ramp':
        sys.path.insert(0,str(BASE.parents[2]/'four_hour_peak_ramp/code'))
        from solve_window import solve_window
    else:solve_window=core._solve_stackelberg_window
    source=Path(args.source_root)/'results'/args.family/args.season/args.scenario/f'seed_{args.seed}'/args.method
    out=BASE/'results'/args.source_objective/args.season/args.scenario/f'seed_{args.seed}'/args.method
    legacy_out=out
    if args.gap_tiered:
        out=BASE/'tiered_results'/args.source_objective/args.season/args.scenario/f'seed_{args.seed}'/args.method
    prior_out=out
    if args.retry_unresolved:
        out=BASE/'retry_results'/args.source_objective/args.season/args.scenario/f'seed_{args.seed}'/args.method
    if args.second_pass:
        prior_out=out
        label='jump_results' if args.jump_only else 'second_pass_results'
        out=BASE/label/args.source_objective/args.season/args.scenario/f'seed_{args.seed}'/args.method
    if args.third_pass:
        prior_out=out
        label='jump_followup_results' if args.jump_only else 'third_pass_results'
        out=BASE/label/args.source_objective/args.season/args.scenario/f'seed_{args.seed}'/args.method
    out.mkdir(parents=True,exist_ok=True)
    state,meta=load_source(source); cfg=meta['config']
    if hashlib.sha256(Path(solve_window.__code__.co_filename).read_bytes()).hexdigest()!=meta['solver_sha256']:
        raise ValueError('Solver source differs; cannot transfer objective bounds across formulations')
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest()!=meta['input_sha256']:
        raise ValueError('Source input hash differs')
    core.PHASE4_SETTINGS.update(meta['physical_settings'])
    core.PHASE4_SETTINGS.update(GRID_EXCHANGE_MODE='grid_export',GUROBI_THREADS=10,
        LEADER_RAMP_WEIGHT=cfg['leader_ramp_weight'],BESS_DEGRADATION_EUR_PER_KWH=cfg['degradation_eur_per_kwh'],SOLVER_OUTPUT=1)
    inputs=list(csv.DictReader(Path(args.input).open()))
    results=json.loads((out/'window_results.json').read_text()) if (out/'window_results.json').exists() else {}
    if (args.retry_unresolved or args.third_pass) and not results and (prior_out/'window_results.json').exists():
        results=json.loads((prior_out/'window_results.json').read_text())
    reasons,checks=inspect(state,meta)
    if args.jump_only:
        reasons={k:v for k,v in reasons.items() if any('ramp' in reason for reason in v)}
    if args.gap_tiered:
        gaps={s['origin']:s['gap'] for s in state['solves']}
        reasons={k:v for k,v in reasons.items() if gaps[k]>.100001}
    atomic_json(out/'source_audit.json',dict(source=str(source),source_manifest=meta,checks=checks,
        mode='independent fixed-state windows; not a replacement simulation',
        attempts_s=[600,1200,2400] if args.second_pass else ([120,600,2400] if args.retry_unresolved else [120,600,1200]),
        high_gap_target=.10,medium_gap_target=.02,already_low_gap_ramp_target=.01))
    if checks['physical_errors']:
        atomic_json(out/'validation_failure.json',checks)
        raise ValueError('Source physical/forecast validation failed; solver audit stopped')
    logs={s['origin']:s for s in state['solves']}
    forecasts={}
    for f in state['forecasts']:forecasts.setdefault(f['origin'],[]).append(f)
    ordered=sorted(reasons,key=lambda k:(logs[k]['gap']<=.20,-logs[k]['gap'],k))
    total=len(ordered)
    for origin in ordered:
        target=review_target(logs[origin]['gap'],reasons[origin])
        if args.gap_tiered and not args.independent_certificate and str(origin) not in results and (legacy_out/'window_results.json').exists():
            cached=json.loads((legacy_out/'window_results.json').read_text()).get(str(origin),{})
            if cached.get('combined_certificate_gap',math.inf)<=target+1e-7:
                results[str(origin)]=dict(cached,target_gap=target,reused_prior_certificate=True)
                atomic_json(out/'window_results.json',results)
        cached=results.get(str(origin),{})
        reusable=not args.independent_certificate or cached.get('certificate_scope','').startswith('new solve only')
        if reusable and cached.get('combined_certificate_gap',math.inf)<=target+1e-7 and cached.get('target_gap',1)<=target+1e-7:continue
        atomic_json(out/'progress.json',dict(status='solving',current_step=len(results),total=total,
            origin=origin,decision_time=state['rows'][origin]['timestamp']))
        old=logs[origin]; row=state['rows'][origin]; prev=state['rows'][origin-1]
        energies={b:float(row[f'energy_before_bus_{i}']) for i,b in enumerate(core.BUS_ORDER,1)}
        prediction={'load':{},'pv':{}}; original_bess={}; original_pb=None
        for i,b in enumerate(core.BUS_ORDER,1):
            fs=sorted([f for f in forecasts[origin] if f['bus']==i],key=lambda f:f['lead'])
            if len(fs)!=16:raise ValueError('Incomplete original prediction window')
            prediction['load'][b]=np.array([f['predicted_load'] for f in fs])
            prediction['pv'][b]=np.array([f['predicted_pv'] for f in fs])
            original_bess[b]=np.array([f['planned_bess'] for f in fs])
            original_pb=np.array([f['planned_pb'] for f in fs])
        baseline=sum(prediction['load'][b]-prediction['pv'][b] for b in core.BUS_ORDER)
        old_grid=baseline+sum(original_bess.values())
        old_plan=plan_values(old_grid,prev['realized_net_kw'],args.source_objective,cfg['leader_ramp_weight'])
        pa=np.array([float(r['pa_eur_per_kwh']) for r in inputs[origin+1:origin+17]])
        if abs(old['objective']-old_plan['objective'])>1e-4*max(1.,abs(old['objective'])):
            raise ValueError('Original objective is inconsistent with saved plan; no certificate transfer allowed')
        attempts=[]; best=None
        warm_start=results.get(str(origin),{}).get('new_pb',original_pb) if args.third_pass else original_pb
        independent=args.independent_certificate
        bound=-math.inf if independent else old.get('best_bound',max(0.,old['objective']*(1-old['gap'])))
        limits=(600,1200,2400) if args.second_pass else ((120,600,2400) if args.retry_unresolved else (120,600,1200))
        for limit in limits:
            tick=time.monotonic()
            try:
                result=solve_window(datetime.fromisoformat(row['timestamp']),origin,prediction,energies,pa,
                    prev['pb_eur_per_kwh'],None,warm_pb=warm_start,mip_gap=target,
                    time_limit_s=limit,previous_realized_net=prev['realized_net_kw'])
                valid=all(math.isfinite(result[k]) for k in ('objective','best_bound','gap')) and max(
                    result[k] for k in ('constraint_violation','bound_violation','integrality_violation'))<=3e-5
                attempts.append(dict(limit_s=limit,elapsed_s=time.monotonic()-tick,valid=valid,
                    **{k:result[k] if math.isfinite(result[k]) else None for k in
                       ('objective','gap','best_bound','constraint_violation','bound_violation','integrality_violation')}))
                if valid:
                    bound=max(bound,result['best_bound'])
                    if best is None or result['objective']<best['objective']:best=result
                    upper=best['objective'] if independent else min(old['objective'],best['objective'])
                    if bound>upper+1e-5*max(1.,abs(upper)):
                        raise ValueError('Inconsistent upper/lower bounds in fixed-state audit')
                    certificate=max(0.,upper-bound)/max(abs(upper),1e-10)
                    if certificate<=target+1e-7:break
            except RuntimeError as exc:
                if 'no incumbent' not in str(exc).lower():raise
                attempts.append(dict(limit_s=limit,elapsed_s=time.monotonic()-tick,error=str(exc)))
        entry=dict(origin=origin,time=row['timestamp'],reasons=sorted(reasons[origin]),target_gap=target,
            old_objective=old['objective'],old_gap=old['gap'],old_plan=old_plan,
            objective_identity_error=abs(old['objective']-old_plan['objective']),attempts=attempts,
            certificate_scope='new solve only; original certificate quarantined' if independent else 'fixed-state combined',
            prior_review=results.get(str(origin)))
        if best is not None:
            grid=baseline+sum(best['bess_plans'].values())
            upper=best['objective'] if independent else min(old['objective'],best['objective'])
            entry.update(new_objective=best['objective'],new_gap=best['gap'],best_bound=bound,
                combined_certificate_gap=max(0.,upper-bound)/max(abs(upper),1e-10),
                new_plan=plan_values(grid,prev['realized_net_kw'],args.source_objective,cfg['leader_ramp_weight']),
                new_pb=list(map(float,best['pb'])),
                new_bess={b:list(map(float,v)) for b,v in best['bess_plans'].items()})
            improvement=(old['objective']-best['objective'])/max(abs(old['objective']),1e-10)
            entry['relative_objective_improvement']=improvement
            if entry['combined_certificate_gap']>target+1e-7:entry['classification']='inconclusive_time_limit'
            elif improvement>.10:entry['classification']='substantially_better_window_solution'
            elif abs(old_plan['first_grid_kw']-entry['new_plan']['first_grid_kw'])>1:
                entry['classification']='action_sensitive_at_similar_objective'
            else:entry['classification']='no_large_first_action_change'
        else:entry['classification']='no_new_valid_incumbent'
        prior=results.get(str(origin),{})
        if args.third_pass and prior.get('combined_certificate_gap',math.inf)<entry.get('combined_certificate_gap',math.inf):
            entry=dict(prior,latest_attempts=attempts,retained_better_previous_certificate=True)
        results[str(origin)]=entry
        atomic_json(out/'window_results.json',results)
        table=io.StringIO(newline='')
        writer=csv.DictWriter(table,fieldnames=['origin','time','old_objective','new_objective','old_gap',
            'combined_certificate_gap','old_first_grid_kw','new_first_grid_kw','relative_objective_improvement','classification'])
        writer.writeheader()
        for item in sorted(results.values(),key=lambda r:r['origin']):
            record={k:item.get(k) for k in writer.fieldnames}
            record['old_first_grid_kw']=item['old_plan']['first_grid_kw']
            record['new_first_grid_kw']=item.get('new_plan',{}).get('first_grid_kw')
            writer.writerow(record)
        atomic_text(out/'window_comparison.csv',table.getvalue().replace('\r\n','\n'))
        atomic_json(out/'progress.json',dict(status='running',current_step=len(results),total=total,origin=origin))
        print(args.source_objective,args.season,args.scenario,args.method,len(results),'/',total,
              'origin',origin,entry['classification'],flush=True)
    # Source may have advanced while independent windows were being audited.
    latest,latest_meta=load_source(source); remaining,_=inspect(latest,latest_meta)
    if args.jump_only:
        remaining={k:v for k,v in remaining.items() if any('ramp' in reason for reason in v)}
    if args.gap_tiered:
        gaps={s['origin']:s['gap'] for s in latest['solves']}
        remaining={k:v for k,v in remaining.items() if gaps[k]>.100001}
    if latest['next_step']<304 or any(str(k) not in results for k in remaining):
        atomic_json(out/'progress.json',dict(status='waiting_for_source_or_new_windows',
            current_step=len(results),total=max(total,len(remaining))))
        return 75
    classifications={}
    for r in results.values():classifications[r['classification']]=classifications.get(r['classification'],0)+1
    unresolved=sum(r['classification'] in ('inconclusive_time_limit','no_new_valid_incumbent') for r in results.values())
    atomic_json(out/'metrics.json',dict(audit_only=True,windows_reviewed=len(results),classifications=classifications,
        source_complete=True,physical_errors=checks['physical_errors'],
        unresolved_windows=unresolved,all_targets_met=(unresolved==0)))
    atomic_json(out/'progress.json',dict(status='completed',current_step=len(results),total=len(results)))
    return 0


if __name__=='__main__':
    sys.exit(main())
