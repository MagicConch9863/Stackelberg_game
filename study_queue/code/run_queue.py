"""Persistent, hidden priority scheduler for the requested AR and seed studies."""
from pathlib import Path
import csv
import io
import json
import os
import subprocess
import sys
import time
import traceback

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parent
HS=ROOT/'horizon_screening'
STATUS=BASE/'status'
sys.path.insert(0,str(HS/'code'))
from io_utils import atomic_json,atomic_text,exclusive_lock
from run_suite import Attached,process_info,complete
from review_case import targets_for

BOOTSTRAP="import pathlib,runpy,site,sys;site.sethelper();entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));sys.argv[0]=entry;runpy.run_path(entry,run_name='__main__')"


def read(path,default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except FileNotFoundError:
        return default


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def command(task):
    script=HS/'code'/('review_case.py' if task.get('quality_review') else 'run_case.py')
    args=['--input',str(HS/'inputs'/f'{task["season"]}.csv'),'--season',task['season'],
        '--family','matched','--scenario',task['scenario'],'--seed',str(task['seed']),
        '--method',task['method'],'--horizon',str(task['horizon']),'--objective',task['objective'],
        '--execute','--review-output',task['output']]
    if 'rho' in task:args+=['--rho',str(task['rho'])]
    if task.get('shared_warmup'):args+=['--shared-warmup',task['shared_warmup']]
    if task.get('generate_warmup'):args+=['--generate-warmup']
    if task.get('quality_review'):
        args+=['--review-source',task['review_source'],'--quality-review']
        if task.get('review_failed_origin') is not None:args+=['--failed-origin',str(task['review_failed_origin'])]
    return [sys.executable,'-S','-u','-c',BOOTSTRAP,str(script)]+args


def is_ready(task):
    return task.get('generate_warmup') or not task.get('shared_warmup') or Path(task['shared_warmup']).exists()


def state_step(folder):
    import re
    path=Path(folder)/'checkpoint.json'
    if not path.exists():return 0
    with path.open(encoding='utf-8') as stream:head=stream.read(1500)
    match=re.search(r'"next_step"\s*:\s*(\d+)',head)
    return int(match.group(1)) if match else 0


def finish(task,code):
    task['last_exit_code']=code
    task['last_finished_at']=stamp()
    if complete(task):
        task.update(status='completed',finished_at=stamp())
        return
    logfile=BASE/'logs'/f'{task["name"]}.log'
    if logfile.exists() and logfile.stat().st_size==task.get('launch_log_start_bytes',-1):
        task['startup_deferrals']=task.get('startup_deferrals',0)+1
        task.update(status='pending',retry_after=time.time()+min(300,10*task['startup_deferrals']),
            last_failure='Interpreter nicht gestartet; automatischer Neustart')
        return
    task['failures']=task.get('failures',0)+1
    failure=read(Path(task['output'])/'failure.json',{})
    task['last_failure']=failure.get('reason','Prozess beendet; Checkpoint bleibt erhalten')
    tail=logfile.read_text(encoding='utf-8',errors='replace')[-8000:] if logfile.exists() else ''
    configuration_error=any(v in tail for v in ('Existing results use different','AssertionError','ModuleNotFoundError','ValueError:'))
    if configuration_error and task['failures']>=3:
        task.update(status='attention',retry_after=0)
        return
    task.update(status='pending',retry_after=time.time()+min(300,15*task['failures']))
    # A failed task must not starve the other requested experiments.
    task.setdefault('original_stage',task['stage'])
    task['stage']=4
    if task.get('quality_review'):
        path=Path(task['output'])/'review_plan.json'
        plan=read(path,{})
        if plan:
            plan['automatic_round']=int(plan.get('automatic_round',0))+1
            atomic_json(path,plan)


def queue_quality(tasks):
    for task in list(tasks):
        if task['status']!='completed' or task.get('quality_review') or task.get('quality_scanned'):continue
        solves=read(Path(task['output'])/'solver_log.json',[])
        targets=targets_for([v for v in solves if v['origin']>=16])
        task['quality_scanned']=True
        if not targets:continue
        new=dict(task)
        new.update(name=task['name']+'_quality',stage=4,status='pending',attempt=0,failures=0,retry_after=0,
            output=str(BASE/'review'/task['name']),quality_review=True,review_source=task['output'],
            review_target_windows=len(targets),parent_case=task['name'])
        for key in ('pid','process_created','started_at','started_timestamp','finished_at','quality_scanned'):
            new.pop(key,None)
        tasks.append(new)


def publish(tasks,active,cfg,phase='running'):
    counts={s:sum(t['status']==s for t in tasks) for s in ('pending','running','completed','attention')}
    lines=[f'Aktualisiert: {stamp()}',f'Aktive Faelle: {len(active)}/{cfg["parallel_cases"]}; Gurobi-Threads je Fall: {cfg["threads_per_case"]}',
        'Prioritaet: 1 rho=0.5; 2 Seed-Paare; 3 Monte-Carlo-Fortsetzung; 4 Qualitaetspruefung.',
        'MPC: 60/120/600 s; OL: 3600 s. Strenge Nachpruefung: MPC 120/600/3600 s; OL 3600 s.',
        '8h MPC bleibt pausiert. Freie Plaetze werden alle 10 s nachbesetzt.']
    rows=[]
    for stage,label in cfg['stages'].items():
        group=[t for t in tasks if t['stage']==int(stage)]
        values={s:sum(t['status']==s for t in group) for s in counts}
        lines.append(f'{stage}. {label}: fertig {values["completed"]}/{len(group)}, aktiv {values["running"]}, offen {values["pending"]}, zu klaeren {values["attention"]}')
    for task in tasks:
        step=state_step(task['output'])
        progress=read(Path(task['output'])/'progress.json',{})
        row=dict(name=task['name'],stage=task['stage'],status=task['status'],method=task['method'],horizon_hours=task['horizon']/4,
            season=task['season'],scenario=task['scenario'],seed=task['seed'],rho=task.get('rho',.9),
            checkpoint_step=step,total_steps=304,pid=task.get('pid',''),output=task['output'])
        rows.append(row)
        if task['status']=='running':
            elapsed=(time.time()-task.get('started_timestamp',time.time()))/60
            lines.append(f'AKTIV {task["name"]} | Schritt {step}/304 | aktueller Start vor {elapsed:.1f} min | Zeitpunkt {progress.get("decision_time", "Start")}')
        elif task['status']=='attention':
            lines.append(f'PRUEFEN {task["name"]} | {task.get("last_failure", "Startfehler")} | logs/{task["name"]}.log')
    blocked=[t for t in tasks if t['status']=='pending' and not is_ready(t)]
    if blocked:lines.append(f'Abhaengige OL-Faelle: {len(blocked)} warten auf identische 4h-Vorgeschichte; andere bereite Aufgaben fuellen freie Plaetze.')
    payload=dict(updated_at=stamp(),counts=counts,cases=tasks)
    atomic_json(STATUS/'suite_status.json',payload)
    atomic_json(STATUS/'coordinator.json',dict(pid=os.getpid(),process_created=process_info(os.getpid())[1],
        status=phase,updated_at=stamp(),live_processes=len(active),parallel_cases=cfg['parallel_cases'],counts=counts))
    content='\n'.join(lines)+'\n'
    atomic_text(STATUS/'progress_brief.log',content)
    atomic_text(HS/'status/progress_brief.log',content)
    atomic_text(HS/'status/terminal_status.log',content)
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    atomic_text(STATUS/'cases.csv',stream.getvalue())
    last=read(STATUS/'two_hour_snapshot.json',{})
    if time.time()-last.get('timestamp',0)>=7200:
        atomic_json(STATUS/'two_hour_snapshot.json',dict(timestamp=time.time(),updated_at=stamp(),counts=counts,active=len(active)))
        atomic_text(STATUS/'two_hour_check.log',content)
        with (STATUS/'two_hour_history.log').open('a',encoding='utf-8') as log:log.write(content+'\n')


def main():
    cfg=read(BASE/'configs/plan.json')
    saved=read(STATUS/'suite_status.json')
    tasks=saved['cases'] if saved else cfg['cases']
    if not saved:
        old=read(HS/'status/suite_status.json')['cases']
        for item in old:
            if item.get('review_mode')=='quality_review' and not complete(item):
                new=dict(item)
                new.update(stage=4,status='pending',attempt=0,quality_review=True,retry_after=0)
                for key in ('pid','process_created'):new.pop(key,None)
                tasks.append(new)
    active={}
    for task in tasks:
        if task['status']=='completed':continue
        if complete(task):task['status']='completed';continue
        if task['status']=='running':
            process=Attached(task['pid'],task.get('process_created'))
            if process.poll() is None:active[task['name']]=(process,None,task)
            else:task.update(status='pending',retry_after=0)
    env=os.environ.copy()
    env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')})
    env['PYTHONPATH']=str(ROOT/'venv/Lib/site-packages')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONIOENCODING']='utf-8'
    while True:
        for name,(worker,log,task) in list(active.items()):
            code=worker.poll()
            if code is not None:
                if log:log.close()
                finish(task,code);del active[name]
        queue_quality(tasks)
        if (STATUS/'STOP').exists():
            publish(tasks,active,cfg,'dispatch_stopped');return
        for task in sorted(tasks,key=lambda t:(t['stage'],t.get('attempt',0),-float(t.get('review_last_gap') or 0),t['name'])):
            if len(active)>=cfg['parallel_cases']:break
            if task['status']!='pending' or time.time()<task.get('retry_after',0) or not is_ready(task):continue
            if complete(task):task['status']='completed';continue
            path=BASE/'logs'/f'{task["name"]}.log'
            log=path.open('ab',buffering=0)
            log.write(f'\n[{stamp()}] Start/Fortsetzung: {task["name"]}\n'.encode('utf-8'))
            task['launch_log_start_bytes']=path.stat().st_size
            try:
                worker=subprocess.Popen(command(task),cwd=str(ROOT),env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                task.update(status='running',pid=worker.pid,process_created=process_info(worker.pid)[1],
                    started_at=stamp(),started_timestamp=time.time(),attempt=task.get('attempt',0)+1)
                active[task['name']]=(worker,log,task)
                # Persist each launch so a coordinator restart cannot duplicate it.
                atomic_json(STATUS/'suite_status.json',dict(updated_at=stamp(),cases=tasks))
            except OSError as exc:
                log.close();task.update(retry_after=time.time()+30,last_failure=str(exc))
        publish(tasks,active,cfg)
        if not active and all(t['status'] in ('completed','attention') for t in tasks):
            publish(tasks,active,cfg,'completed' if all(t['status']=='completed' for t in tasks) else 'attention');return
        time.sleep(cfg['refill_seconds'])


if __name__=='__main__':
    sys.stdout=(STATUS/'coordinator_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(STATUS/'coordinator_stderr.log').open('a',encoding='utf-8',buffering=1)
    with exclusive_lock(STATUS/'coordinator.lock'):
        try:main()
        except Exception:
            with (STATUS/'coordinator_error.log').open('a',encoding='utf-8') as log:traceback.print_exc(file=log)
            raise
