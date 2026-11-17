"""Validate or start an exported review package; does not import a checkout's model."""
from pathlib import Path
import ast
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import site
import subprocess
import sys

BASE=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate():
    hs=BASE/'horizon_screening'
    with (BASE/'files.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            relative=row['file']
            # Checkpoints and execution-only settings legitimately change on resume.
            if (relative.startswith('cases/') and '/resume/' in relative) or relative in (
                    'study_queue/configs/plan.json','horizon_screening/configs/experiment.json'):
                continue
            path=BASE/relative
            assert path.is_file() and sha(path)==row['sha256'], f'File checksum mismatch: {relative}'
    catalog=read(BASE/'catalog.json')
    runtime=read(hs/'configs/experiment.json')
    tree=ast.parse((hs/'code/run_case.py').read_text(encoding='utf-8-sig'))
    allowed=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(target,ast.Name) and target.id=='IO_COMPATIBLE_RUNNERS' for target in node.targets))
    allowed.add(sha(hs/'code/run_case.py'))
    checked=0
    for task in catalog['cases']:
        folder=BASE/'cases'/task['name']/'resume'
        state=read(folder/'checkpoint.json');manifest=read(folder/'manifest.json');plan=read(folder/'review_plan.json')
        assert 16<=state['next_step']<=304
        assert [r['execution_step'] for r in state['rows']]==list(range(state['next_step']))
        assert all(r['origin']<state['next_step'] for r in state['solves'])
        assert len(state['warm_pb'])==task['horizon']
        expected_config=dict(runtime,horizon_steps=task['horizon'],objective=task['objective'],rho=task['rho'])
        if task['method']=='OL':expected_config['time_limits_s']=[3600]
        old_config=dict(manifest['config'])
        for key in ('threads_per_case','parallel_cases','time_limits_s'):old_config[key]=expected_config[key]
        assert old_config==expected_config, f'Configuration mismatch: {task["name"]}'
        assert manifest['input_sha256']==sha(hs/'inputs'/f'{task["season"]}.csv')
        assert manifest['runner_sha256'] in allowed, f'Unsupported checkpoint runner: {task["name"]}'
        assert manifest['solver_sha256']==sha(hs/'source/forecast_mpc/experiment_core/stackelberg_mpc.py')
        assert manifest['prediction_sha256']==sha(hs/'code/prediction.py')
        assert manifest['peak_solver_sha256']==sha(hs/'code/peak_window.py')
        for solve in state['solves']:
            if solve['origin']>=16:
                assert solve['gap']<=plan['targets'].get(str(solve['origin']),.1)+1e-7, f'Unqualified saved prefix: {task["name"]}'
        last=state['rows'][-1]
        for i,(bus,energy) in enumerate(state['energy'].items(),1):
            assert abs(energy-last[f'soc_bus_{i}_kwh'])<1e-8
            cap=manifest['bus_config'][bus]['bess_e_max']
            limit=manifest['bus_config'][bus]['bess_kw']
            for row in state['rows']:
                assert -3e-5<=row[f'soc_bus_{i}_kwh']<=cap+3e-5
                assert abs(row[f'bess_bus_{i}_kw'])<=limit+3e-5
        assert abs(state['previous_pb']-last['pb_eur_per_kwh'])<1e-8
        assert abs(state['previous_grid']-last['realized_net_kw'])<1e-8
        for row in state['rows']:
            assert abs(row['realized_net_kw']-row['uncontrolled_net_kw']-row['bess_kw'])<3e-5
        with (BASE/task['shared_warmup']).open(encoding='utf-8-sig',newline='') as stream:warm=list(csv.DictReader(stream))
        assert len(warm)==16
        for row,saved in zip(warm,state['rows'][:16]):
            assert row['timestamp']==saved['timestamp']
            assert abs(float(row['uncontrolled_net_kw'])-saved['uncontrolled_net_kw'])<1e-8
            assert abs(float(row['pb_eur_per_kwh'])-saved['pb_eur_per_kwh'])<1e-8
        checked+=1
    assert checked==len(catalog['cases']) and checked>0
    print(f'Pruefung bestanden: {checked} Faelle; Daten, Modellcode, Zustaende, Praefixe und Ziel-gaps konsistent.',flush=True)
    return catalog


def alive(saved):
    if not saved:return False
    sys.path.insert(0,str(BASE/'horizon_screening/code'))
    from run_suite import process_info
    code,created=process_info(saved['pid'])
    return code==259 and created==saved.get('process_created')


def start(catalog,args):
    if os.name!='nt':raise RuntimeError('The saved coordinator uses Windows process reattachment.')
    status=BASE/'study_queue/status'
    coordinator=read(status/'coordinator.json')
    if alive(coordinator):
        print('Koordinator bereits aktiv. Fortschritt: '+str(status/'progress_brief.log'))
        return
    if not args.prepare:
        missing=[name for name in ('numpy','pandas','matplotlib','scipy','gurobipy','simbench') if importlib.util.find_spec(name) is None]
        if missing:raise RuntimeError('Fehlende Pakete: '+', '.join(missing)+'; pip install -r requirements.txt')
    selected=[t for t in catalog['cases'] if args.include_new or t['bundle_kind']=='existing_strict_review']
    previous=read(status/'suite_status.json',dict(cases=[]))
    saved={t['name']:t for t in previous['cases']}
    tasks=[]
    for original in selected:
        task=dict(original)
        if task['name'] in saved:task.update(saved[task['name']])
        task['output']=str((BASE/'cases'/task['name']/'resume').resolve())
        task['review_source']=str((BASE/'cases'/task['name']/'reference').resolve())
        task['shared_warmup']=str((BASE/original['shared_warmup']).resolve())
        folder=Path(task['output'])
        manifest=read(folder/'manifest.json')
        manifest['shared_warmup']=dict(file=task['shared_warmup'],
            interpretation='Shared causal MPC4h prehistory from 50 percent initial energy',seed=task['seed'],rho=task['rho'])
        write(folder/'manifest.json',manifest)
        plan=read(folder/'review_plan.json');plan['source']=task['review_source'];plan['quality_review']=True
        write(folder/'review_plan.json',plan)
        tasks.append(task)
    cfg=dict(catalog,cases=tasks,parallel_cases=args.parallel,threads_per_case=args.threads)
    write(BASE/'study_queue/configs/plan.json',cfg)
    write(status/'suite_status.json',dict(cases=tasks))
    settings_path=BASE/'horizon_screening/configs/experiment.json'
    settings=read(settings_path);settings.update(threads_per_case=args.threads,parallel_cases=args.parallel)
    write(settings_path,settings)
    if args.prepare:
        print(f'Vorbereitet: {len(tasks)} Reviews; lokale Pfade gesetzt, keine Optimierung gestartet.')
        return
    env=os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')})
    paths=list(sys.path)+site.getsitepackages()
    env['PYTHONPATH']=os.pathsep.join(dict.fromkeys(p for p in paths if p and Path(p).is_dir()))
    env['PYTHONIOENCODING']='utf-8'
    bootstrap="import pathlib,runpy,site,sys;site.sethelper();entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));sys.argv[0]=entry;runpy.run_path(entry,run_name='__main__')"
    log=(status/'launcher.log').open('ab',buffering=0)
    try:
        child=subprocess.Popen([sys.executable,'-S','-u','-c',bootstrap,str(BASE/'study_queue/code/run_queue.py')],
            cwd=BASE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    finally:log.close()
    print(f'Gestartet: {len(tasks)} Reviews, parallel {args.parallel}, Gurobi-Threads je Fall {args.threads}; PID {child.pid}.')
    print('Fortschritt: '+str(status/'progress_brief.log'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true')
    action.add_argument('--run',action='store_true')
    action.add_argument('--prepare',action='store_true')
    parser.add_argument('--parallel',type=int,default=4)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--include-new',action='store_true')
    args=parser.parse_args()
    if not 1<=args.parallel<=30 or not 1<=args.threads<=192:parser.error('Invalid parallel/thread count')
    catalog=validate()
    if args.run or args.prepare:start(catalog,args)


if __name__=='__main__':main()
