"""Plan tail-slot quality reviews and causally replay certified improved cases."""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys

from io_utils import atomic_json

FORMAL_OBJECTIVES = {'p2v_ramp', 'peak_ramp'}
POLICY = dict(high_gap_target=.10, low_gap_jump_target=.01,
              medium_gap_target=.02, medium_gap_original_range=[.10,.20],
              solve_attempts_s=[120,600,1200],
              jump_min_kw=2.0, jump_excess_over_raw_kw=1.0,
              mode='formal-first; reserve slots for all unfinished formal cases',
              never_splice_independent_window_actions=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_complete(task):
    source = Path(task['requires_completed_source'])
    try:
        return ((source/'metrics.json').exists()
                and read_json(source/'progress.json').get('status') == 'completed')
    except (OSError, ValueError):
        return False


def background_slots(tasks, limit):
    remaining = sum(t.get('objective') in FORMAL_OBJECTIVES
                    and t['status'] in ('pending','running') for t in tasks)
    return max(0, limit-remaining)


def certified_improvements(results):
    return [r for r in results.values()
        if r.get('new_objective') is not None and math.isfinite(r['new_objective'])
        and math.isfinite(r.get('combined_certificate_gap',math.inf))
        and r['combined_certificate_gap']<=r['target_gap']+1e-7
        and r['old_objective']-r['new_objective']>1e-5*max(1.,abs(r['old_objective']))]


def replay_prefix(state,first,warm_pb):
    if not 0<first<304 or first not in {s['origin'] for s in state['solves']}:
        raise ValueError('Replay must start at an original solved decision origin')
    row=state['rows'][first]
    previous=state['rows'][first-1]
    if row['execution_step']!=first or previous['execution_step']!=first-1:
        raise ValueError('Original execution rows are not contiguous')
    return copy.deepcopy(dict(next_step=first,
        energy={f'Bus {i}':row[f'energy_before_bus_{i}'] for i in range(1,5)},
        previous_pb=previous['pb_eur_per_kwh'],previous_grid=previous['realized_net_kw'],
        warm_pb=warm_pb,rows=[r for r in state['rows'] if r['execution_step']<first],
        forecasts=[r for r in state['forecasts'] if r['origin']<first],
        solves=[r for r in state['solves'] if r['origin']<first]))


def review_queue(tasks, active, limit):
    """Reserve capacity for formal retries rather than trapping them behind long audits."""
    primary = [t for t in tasks if not t.get('opportunistic')]
    background = [t for t in tasks if t.get('opportunistic')]
    active_background = sum(bool(item[2].get('opportunistic')) for item in active.values())
    available = max(0, background_slots(tasks,limit)-active_background)
    ready = [t for t in background if t['status']=='pending' and source_complete(t)]
    ready.sort(key=lambda t:(not t['name'].startswith('tiered_'),-t.get('high_gap_windows',0)))
    # Eligibility is enforced again before launch; the scheduler counts new
    # active background workers dynamically as it fills these tail slots.
    return primary+ready if available else primary


def add_audit_tasks(tasks, base):
    base = Path(base)
    audit_base = base/'review/all_cases'
    (audit_base/'logs').mkdir(parents=True,exist_ok=True)
    by_name = {t['name']:t for t in tasks}
    for source_task in list(tasks):
        if source_task.get('objective') not in FORMAL_OBJECTIVES:
            continue
        objective = source_task['objective']
        season, scenario, method = (source_task[k] for k in ('season','scenario','method'))
        seed = int(source_task.get('seed',20250903))
        family = source_task.get('family','matched')
        source_root = Path(source_task['suite_base'])
        # This already completed, causally replayed sensitivity branch is the
        # established source for the known first-seed summer Perfect anomaly.
        corrected = base/'review/summer_perfect'
        if (objective,season,scenario,method,seed)==('p2v_ramp','summer','perfect','MPC',20250903):
            if (corrected/'results'/family/season/scenario/f'seed_{seed}'/method/'metrics.json').exists():
                source_root = corrected
        suffix = '' if seed==20250903 else f'_seed_{seed}'
        name = f'audit_{objective}_{season}_{scenario}{suffix}_{method}'
        definition = dict(name=name,season=season,scenario=scenario,method=method,
            objective='audit_'+objective,family=family,seed=seed,suite_base=str(audit_base),
            output=str(audit_base/'results'/objective/season/scenario/f'seed_{seed}'/method),
            status='pending',isolated_startup=True,opportunistic=True,
            source_task_name=source_task['name'],
            requires_completed_source=str(source_root/'results'/family/season/scenario/f'seed_{seed}'/method),
            extra_args=['--source-root',str(source_root),'--source-objective',objective])
        if name in by_name:
            # Keep the existing independent-window checkpoint and live PID.
            existing = by_name[name]
            for key in ('opportunistic','source_task_name','requires_completed_source','family','seed'):
                existing[key] = definition[key]
        else:
            tasks.append(definition)
            by_name[name] = definition
        state_path=Path(definition['requires_completed_source'])/'checkpoint.json'
        if state_path.exists():
            state=read_json(state_path)
            gaps=[s['gap'] for s in state['solves'] if s['origin']>=16 and s['gap']>.100001]
            tier_name='tiered_'+name.removeprefix('audit_')
            if gaps and tier_name not in by_name:
                tier=dict(definition,name=tier_name,objective='tiered_'+objective,
                    script=str(audit_base/'code/run_case.py'),
                    output=str(audit_base/'tiered_results'/objective/season/scenario/f'seed_{seed}'/method),
                    extra_args=definition['extra_args']+['--gap-tiered'],
                    high_gap_windows=sum(g>.20 for g in gaps),
                    medium_gap_windows=sum(g<=.20 for g in gaps))
                tasks.append(tier);by_name[tier_name]=tier
            tier_output=audit_base/'tiered_results'/objective/season/scenario/f'seed_{seed}'/method
            evidence_path=tier_output/'window_results.json'
            if evidence_path.exists():
                evidence=read_json(evidence_path)
                unresolved=sum(r.get('combined_certificate_gap',math.inf)>r['target_gap']+1e-7 for r in evidence.values())
                missing=len(gaps)-len(evidence)
                retry_name='retry_'+tier_name
                if (unresolved or missing) and retry_name not in by_name:
                    extra=definition['extra_args']+['--gap-tiered','--retry-unresolved']
                    if (objective,season,scenario,seed,method)==('peak_ramp','spring','iid',20250903,'MPC'):
                        extra+=['--independent-certificate']
                    retry=dict(definition,name=retry_name,objective='retry_tiered_'+objective,
                        script=str(audit_base/'code/run_case.py'),
                        output=str(audit_base/'retry_results'/objective/season/scenario/f'seed_{seed}'/method),extra_args=extra)
                    tasks.append(retry);by_name[retry_name]=retry
    atomic_json(audit_base/'policy.json',POLICY)
    add_second_pass_tasks(tasks,base)


def add_second_pass_tasks(tasks,base):
    """Explicit second pass: unresolved certificates and jumps on completed replays."""
    import csv
    saved=base/'status/suite_status.json'
    if not saved.exists():return
    old=read_json(saved)['cases'];known={t['name'] for t in tasks}
    audit_base=base/'review/all_cases'
    for t in old:
        high=t['name'].startswith('retry_tiered_') and t['status']=='completed'
        jump=t['name'].startswith('replay_') and t['status']=='completed' and t['method']=='MPC'
        if not (high or jump):continue
        if high:
            results=read_json(Path(t['output'])/'window_results.json')
            count=sum(r.get('combined_certificate_gap',math.inf)>r['target_gap']+1e-7 for r in results.values())
            if not count:continue
            objective=t['objective'].removeprefix('retry_tiered_')
            extra=t['extra_args']+['--second-pass']
            label='second_pass_results';name='second_'+t['name']
        else:
            rows=list(csv.DictReader((Path(t['output'])/'official_timeseries.csv').open()))
            origins=set()
            for a,b in zip(rows,rows[1:]):
                dp=abs(float(b['realized_net_kw'])-float(a['realized_net_kw']))
                dx=abs(float(b['uncontrolled_net_kw'])-float(a['uncontrolled_net_kw']))
                if dp>=2 and dp-dx>=1:origins.update((int(a['origin']),int(b['origin'])))
            if not origins:continue
            count=len(origins);objective=t['objective'].removeprefix('replay_')
            extra=['--source-root',str(Path(t['output']).parents[5]),'--source-objective',objective,
                '--second-pass','--jump-only','--independent-certificate']
            label='jump_results';name='jump_'+t['name']
        if name in known:continue
        tasks.append(dict(name=name,objective='diagnostic_'+objective,
            season=t['season'],scenario=t['scenario'],method=t['method'],seed=t['seed'],
            family=t.get('family','matched'),suite_base=str(audit_base),
            script=str(audit_base/'code/run_case.py'),
            output=str(audit_base/label/objective/t['season']/t['scenario']/f'seed_{t["seed"]}'/t['method']),
            status='pending',opportunistic=True,isolated_startup=True,
            requires_completed_source=t['requires_completed_source'] if high else t['output'],
            extra_args=extra,selected_windows=count))
        known.add(name)


def discover_replays(tasks, base):
    """Inspect at most one newly completed audit per minute; retain originals."""
    base = Path(base)
    for task in list(tasks):
        if not task.get('objective','').startswith('audit_') or task['status']!='completed':
            continue
        if task.get('replay_decision'):
            continue
        try:
            out=Path(task['output'])
            evidence=read_json(out/'source_audit.json')
            results=read_json(out/'window_results.json')
            if evidence['checks']['physical_errors']:
                task['replay_decision']='blocked_by_source_validation'
                return
            improved=certified_improvements(results)
            if not improved:
                task['replay_decision']='no_certified_objective_improvement'
                return
            objective=task['objective'].removeprefix('audit_')
            destination=base/'review/replayed'/objective
            (destination/'logs').mkdir(parents=True,exist_ok=True)
            seed=int(task.get('seed',20250903));family=task.get('family','matched')
            case_base=destination/'jobs'/f'{task["season"]}_{task["scenario"]}_{seed}_{task["method"]}'
            name='replay_'+task['name'].removeprefix('audit_')
            if not any(t['name']==name for t in tasks):
                tasks.append(dict(name=name,objective='replay_'+objective,
                    season=task['season'],scenario=task['scenario'],method=task['method'],
                    seed=seed,family=family,suite_base=str(destination),
                    script=str(base/'code/postrun_review.py'),
                    output=str(case_base/'results'/family/task['season']/task['scenario']/f'seed_{seed}'/task['method']),
                    status='pending',opportunistic=True,isolated_startup=True,
                    source_task_name=task['name'],requires_completed_source=str(out),
                    extra_args=['--replay-audit',str(out),'--destination',str(case_base),
                                '--source-objective',objective]))
            task['replay_decision']='causal_replay_queued'
            task['replay_task_name']=name
            return
        except (OSError,ValueError,KeyError) as error:
            # No audit/source data is silently treated as valid; retry discovery
            # after a transient read or an audit still committing its files.
            task['replay_discovery_errors']=task.get('replay_discovery_errors',0)+1
            task['replay_discovery_error']=str(error)
            if task['replay_discovery_errors']>=3:
                task['replay_decision']='blocked_by_audit_read_error'
            return


def add_followup_tasks(tasks,base):
    """Fill free slots only after the same case's preceding review has finished."""
    known={t['name'] for t in tasks}
    for t in list(tasks):
        if t['status']!='completed' or not (t['name'].startswith('second_retry_') or t['name'].startswith('jump_replay_')):
            continue
        name='followup_'+t['name']
        if name in known:continue
        path=Path(t['output'])/'window_results.json'
        if not path.exists():continue
        results=read_json(path)
        unresolved=sum(r.get('combined_certificate_gap',math.inf)>r['target_gap']+1e-7 for r in results.values())
        if not unresolved:continue
        label='jump_followup_results' if t['name'].startswith('jump_') else 'third_pass_results'
        previous_label='jump_results' if t['name'].startswith('jump_') else 'second_pass_results'
        task=copy.deepcopy(t)
        task.update(name=name,objective='followup_diagnostic',status='pending',
            output=str(Path(t['output'].replace(previous_label,label))),
            extra_args=t['extra_args']+['--third-pass'],
            requires_completed_source=t['output'],source_task_name=t['name'],selected_windows=unresolved,
            retries=0,retry_after=0)
        for field in ('pid','started_at','finished_at','exit_code','resumed_completed','retry_reason','startup_deferrals'):
            task.pop(field,None)
        tasks.append(task);known.add(name)


def run_replay():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('input','season','family','scenario','method','seed','replay-audit','destination','source-objective'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:parser.error('--execute required')
    import site
    site.sethelper()
    audit=Path(args.replay_audit); destination=Path(args.destination)
    evidence=read_json(audit/'source_audit.json')
    source=Path(evidence['source']); meta=evidence['source_manifest']
    if hashlib.sha256(Path(args.input).read_bytes()).hexdigest()!=meta['input_sha256']:
        raise ValueError('Replay input differs from source input')
    if evidence['checks']['physical_errors']:
        raise ValueError('Source physical/forecast validation failed; no replay allowed')
    windows=read_json(audit/'window_results.json')
    improved=certified_improvements(windows)
    if not improved:raise ValueError('No certified improved window to replay')
    first=min(r['origin'] for r in improved)
    state=read_json(source/'checkpoint.json')
    if state['next_step']!=304:raise ValueError('Require completed source trajectory')
    identity=(args.season,args.scenario,args.method,int(args.seed),args.family)
    if identity!=(meta['season'],meta['scenario'],meta['method'],int(meta['seed']),meta['family']):
        raise ValueError('Replay identity differs from source')
    objective=args.source_objective
    if objective not in FORMAL_OBJECTIVES:raise ValueError('Invalid replay objective')
    cfg=copy.deepcopy(meta['config'])
    stride=16 if args.method=='OL' else 1
    cfg['solver_review']=dict(origins=list(range(first,304,stride)),time_limits_s=[120,600,1200])
    cfg['target_gap']=.10
    cfg['review_provenance']=dict(source=str(source),audit=str(audit),replay_from=first,
        low_gap_jump_origins=[r['origin'] for r in windows.values() if r['target_gap']<=.010001],
        prefix_solver_logs_retained=True,independent_windows_never_spliced=True,
        source_checkpoint_sha256=hashlib.sha256((source/'checkpoint.json').read_bytes()).hexdigest(),
        audit_results_sha256=hashlib.sha256((audit/'window_results.json').read_bytes()).hexdigest(),
        review_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    case_base=destination
    (case_base/'configs').mkdir(parents=True,exist_ok=True)
    config_path=case_base/'configs/experiment.json'
    if config_path.exists() and read_json(config_path)!=cfg:
        previous_cfg=read_json(config_path)
        comparison=copy.deepcopy(previous_cfg)
        comparison['review_provenance']['review_runner_sha256']=cfg['review_provenance']['review_runner_sha256']
        if comparison!=cfg:
            raise ValueError('Existing replay uses another source or review policy')
        # A wrapper-only update is not a change of source/model/review policy.
        cfg=previous_cfg
    atomic_json(case_base/'configs/experiment.json',cfg)
    output=case_base/'results'/args.family/args.season/args.scenario/f'seed_{args.seed}'/args.method
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'checkpoint.json').exists():
        prefix=replay_prefix(state,first,next(r['new_pb'] for r in improved if r['origin']==first))
        atomic_json(output/'checkpoint.json',prefix)
        atomic_json(output/'review_provenance.json',cfg['review_provenance'])
    shared=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('causal_review_runner',shared/'run_case.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    core_path=shared.parents[1]/'hpc_seasonal_actual_load_noise/src/forecast_mpc'
    sys.path.insert(0,str(core_path))
    from experiment_core import stackelberg_mpc as core
    solver_source=Path(core.__file__)
    solve=core._solve_stackelberg_window
    if objective=='peak_ramp':
        peak_path=shared.parents[1]/'four_hour_peak_ramp/code'
        sys.path.insert(0,str(peak_path))
        from solve_window import solve_window
        solve=solve_window;solver_source=peak_path/'solve_window.py'
    if hashlib.sha256(solver_source.read_bytes()).hexdigest()!=meta['solver_sha256']:
        raise ValueError('Replay solver differs from the source formulation')
    if core.BUS_CONFIG!=meta['bus_config']:
        raise ValueError('Replay device configuration differs from source')
    core.PHASE4_SETTINGS.update(meta['physical_settings'])
    strict_low=set(cfg['review_provenance']['low_gap_jump_origins'])
    def strict_solve(*positional,**keywords):
        origin=positional[1]
        target=.01 if origin in strict_low else .10
        keywords['mip_gap']=target
        result=solve(*positional,**keywords)
        if not math.isfinite(result['gap']) or result['gap']>target+1e-7:
            raise RuntimeError('no incumbent accepted by strict review policy '
                f'(feasible incumbent exists; gap={result["gap"]}; target={target})')
        return result
    sys.argv=[str(shared/'run_case.py'),'--input',args.input,'--season',args.season,
        '--family',args.family,'--scenario',args.scenario,'--method',args.method,
        '--seed',args.seed,'--execute']
    runner.main(experiment_base=case_base,solve_window=strict_solve,solver_source=solver_source)
    atomic_json(output/'metrics.json',dict(**read_json(output/'metrics.json'),
        review_provenance=cfg['review_provenance']))


if __name__=='__main__':
    run_replay()
