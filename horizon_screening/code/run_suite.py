"""Headless 48-case scheduler with PID reattachment and durable checkpoints."""
from pathlib import Path
import argparse
import csv
import ctypes
from ctypes import wintypes
import io
import json
import os
import subprocess
import sys
import time
from io_utils import atomic_json,atomic_text,exclusive_lock

BASE=Path(__file__).resolve().parents[1]
STATUS=BASE/'status'
KERNEL=ctypes.WinDLL('kernel32',use_last_error=True)
KERNEL.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
KERNEL.OpenProcess.restype=wintypes.HANDLE
KERNEL.CloseHandle.argtypes=[wintypes.HANDLE]
KERNEL.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
KERNEL.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
KERNEL.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT]


def process_info(pid):
    handle=KERNEL.OpenProcess(0x1000,False,pid)
    if not handle:return None,None
    code=wintypes.DWORD();times=[wintypes.FILETIME() for _ in range(4)]
    try:
        if not KERNEL.GetExitCodeProcess(handle,ctypes.byref(code)):return None,None
        if not KERNEL.GetProcessTimes(handle,*[ctypes.byref(v) for v in times]):return None,None
        return int(code.value),times[0].dwHighDateTime<<32|times[0].dwLowDateTime
    finally:
        KERNEL.CloseHandle(handle)


class Attached:
    def __init__(self,pid,created):
        self.pid,self.created=pid,created
    def poll(self):
        code,created=process_info(self.pid)
        if created!=self.created or code is None:return 1
        return None if code==259 else code


def plan(cfg):
    tasks=[]
    seed=20250903
    for scenario in ('perfect','iid','ar1'):
        for season in cfg['dates']:
            for objective in ('p2v_ramp','peak_ramp'):
                for method,h in [('MPC',v) for v in cfg['horizons_mpc']]+[('OL',96)]:
                    label='S1';variant=f'{method}_H{h:02d}'
                    name=f'{objective}_{season}_{scenario}_{variant}'
                    output=BASE/'results'/objective/variant/season/scenario/f'seed_{seed}'
                    tasks.append(dict(name=name,seed=seed,seed_label=label,season=season,
                        objective=objective,horizon=h,scenario=scenario,method=method,
                        status='pending',output=str(output),
                        retries=0,startup_deferrals=0,retry_after=0))
    assert len(tasks)==120
    return tasks


def external_workers():
    path=BASE.parent/'monte_carlo_3seeds/status/suite_status.json'
    if not path.exists():return 0
    state=json.loads(path.read_text())
    return sum(process_info(t['pid'])[0]==259 and
        process_info(t['pid'])[1]==t.get('process_created')
        for t in state['cases'] if t['status']=='running' and t.get('pid'))


def complete(task):
    folder=Path(task['output'])
    if not all((folder/name).exists() for name in ('metrics.json','checkpoint.json','official_timeseries.csv')):
        return False
    if json.loads((folder/'checkpoint.json').read_text())['next_step']!=304:return False
    with (folder/'official_timeseries.csv').open(encoding='utf-8-sig',newline='') as stream:
        return sum(1 for _ in csv.DictReader(stream))==288


def finish(task,code):
    if code==0 and complete(task):
        task.update(status='completed',exit_code=0,finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        return
    code=code or 1
    path=BASE/'logs'/f'{task["name"]}.log'
    empty=path.stat().st_size==task.get('launch_log_start_bytes',-1) if path.exists() else True
    task['exit_history']=(task.get('exit_history',[])+[code])[-30:]
    if empty:
        task['startup_deferrals']+=1
        task.update(status='pending',retry_after=time.time()+min(300,15*task['startup_deferrals']),
                    retry_reason='interpreter_startup')
    elif task['retries']<2:
        task['retries']+=1
        task.update(status='pending',retry_after=time.time()+60,retry_reason='case_error')
    else:
        task.update(status='failed',exit_code=code,retry_reason='case_error_requires_attention',
                    finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))


def reap(active):
    for name,(process,stream,task) in list(active.items()):
        code=process.poll()
        if code is not None:
            if stream:stream.close()
            finish(task,code);del active[name]


def publish(tasks,active,cfg,phase='running'):
    now=time.time();rows=[]
    counts={s:sum(t['status']==s for t in tasks) for s in ('completed','running','pending','failed','paused')}
    external=external_workers()
    lines=[f'updated: {time.strftime("%Y-%m-%d %H:%M:%S")}',
        'Horizon screening: MPC 2/4/6/8h + OL24h; two objectives; S1; 120 cases',
        f'completed: {counts["completed"]}/120 | failed: {counts["failed"]} | paused: {counts["paused"]} | own running: {len(active)} | pending: {counts["pending"]}',
        'Priority: finish MPC2h, then interleave MPC4h and OL24h. MPC6h/8h paused; checkpoints retained.',
        f'old batch running: {external} | combined running: {external+len(active)}/{cfg["parallel_cases"]}',
        f'per case: {cfg["threads_per_case"]} Gurobi threads | budgets: 60 / 120 / 300 s | no-incumbent rescue: 600 s',
        'Progress includes 16 shared warmup points (304 total). process_min = current process age, excluding earlier startup retries.']
    for task in tasks:
        progress={};path=Path(task['output'])/'progress.json'
        if path.exists():
            try:progress=json.loads(path.read_text())
            except (ValueError,OSError):pass
        step=304 if task['status']=='completed' else progress.get('current_step',0)
        elapsed=(now-task['first_started_timestamp']) if 'first_started_timestamp' in task else 0.
        created=task.get('process_created')
        born=(created/10000000-11644473600) if created else None
        process_minutes=max(0.,(now-born)/60) if task['status']=='running' and born else None
        fresh_progress=path.exists() and born is not None and path.stat().st_mtime >= born
        display_status=('SOLVING' if fresh_progress else 'STARTING') if task['status']=='running' else task['status'].upper()
        rows.append(dict(name=task['name'],seed_label=task['seed_label'],seed=task['seed'],
            season=task['season'],scenario=task['scenario'],method=task['method'],
            status=task['status'],display_status=display_status,current_step=step,total_steps=304,
            first_attempt_age_minutes=round(elapsed/60,2),
            current_process_minutes=round(process_minutes,2) if process_minutes is not None else '',
            startup_retries=task['startup_deferrals'],pid=task.get('pid',''),
            retries=task['retries'],last_attempt_limit_s=progress.get('last_attempt_limit_s',''),
            best_gap=progress.get('best_gap',''),output=task['output']))
        if task['status']=='running':
            lines.append(f'{display_status} {task["name"]} | step {step}/304 | process_min {process_minutes:.1f} | startup_retries {task["startup_deferrals"]}')
        elif task['status']=='failed':
            lines.append(f'FAILED {task["name"]} | checkpoint retained | see logs/{task["name"]}.log')
    if len(active)+external<cfg['parallel_cases'] and counts['pending']:
        lines.append('Free slots will be filled; startup retries may be cooling down.')
    atomic_json(STATUS/'suite_status.json',dict(counts=counts,cases=tasks))
    atomic_text(STATUS/'terminal_status.log','\n'.join(lines)+'\n')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_text(STATUS/'cases.csv',stream.getvalue())
    atomic_json(STATUS/'coordinator.json',dict(pid=os.getpid(),process_created=process_info(os.getpid())[1],
        status=phase,checked_timestamp=now,checked_at=time.strftime('%Y-%m-%d %H:%M:%S'),
        live_processes=len(active),parallel_cases=cfg['parallel_cases'],counts=counts))
    return counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((BASE/'configs/experiment.json').read_text())
    assert 1 <= cfg['parallel_cases'] <= 30 and cfg['threads_per_case'] in (8,12)
    STATUS.mkdir(exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
    tasks=plan(cfg);previous={}
    if (STATUS/'suite_status.json').exists():
        previous={t['name']:t for t in json.loads((STATUS/'suite_status.json').read_text())['cases']}
    active={}
    for task in tasks:
        if task['name'] in previous:task.update(previous[task['name']])
        if complete(task):task['status']='completed'
        elif task['method']=='MPC' and task['horizon'] in (24,32) and task['status']!='failed':
            if task['status']=='running' and task.get('pid'):
                code,created=process_info(task['pid'])
                if code==259 and created==task.get('process_created'):
                    handle=KERNEL.OpenProcess(1,False,task['pid'])
                    if not handle:raise OSError('Cannot pause worker '+task['name'])
                    try:
                        if not KERNEL.TerminateProcess(handle,0):raise OSError('Cannot stop worker '+task['name'])
                    finally:KERNEL.CloseHandle(handle)
            task.update(status='paused',paused_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                        retry_reason='user_paused_long_horizon_checkpoint_retained')
        elif task['status']=='running':
            process=Attached(task['pid'],task.get('process_created'))
            if process.poll() is None:active[task['name']]=(process,None,task)
            else:task.update(status='pending',retry_after=0,retry_reason='resume_after_restart')
        elif task['status']=='failed' and args.retry_failed and not (
                task['method']=='MPC' and task['horizon'] in (24,32)):
            task.update(status='pending',retries=0,retry_after=0)
    env=os.environ.copy()
    env.update({v:'1' for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['PYTHONPATH']=str(BASE.parent/'venv/Lib/site-packages')+os.pathsep+env.get('PYTHONPATH','')
    env['PYTHONIOENCODING']='utf-8'
    publish(tasks,active,cfg)
    while True:
        tick=time.monotonic()
        reap(active)
        if (STATUS/'STOP').exists():
            publish(tasks,active,cfg,'dispatch_stopped');break
        for task in sorted(tasks,key=lambda t:0 if t['method']=='MPC' and t['horizon']==8 else 1):
            if len(active)+external_workers()>=cfg['parallel_cases']:break
            if task['status']!='pending' or time.time()<task.get('retry_after',0):continue
            if complete(task):task['status']='completed';continue
            script=BASE/'code/run_case.py'
            cmd=[sys.executable,'-u',str(script),'--input',str(BASE/'inputs'/f'{task["season"]}.csv'),
                '--season',task['season'],'--family','matched','--scenario',task['scenario'],
                '--method',task['method'],'--seed',str(task['seed']),
                '--horizon',str(task['horizon']),'--objective',task['objective'],'--execute']
            if task['startup_deferrals']>=2:
                bootstrap=('import pathlib,runpy,site,sys;site.sethelper();'
                    'entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));'
                    'sys.argv[0]=entry;runpy.run_path(entry,run_name="__main__")')
                cmd=[sys.executable,'-S','-u','-c',bootstrap]+cmd[2:]
            stream=(BASE/'logs'/f'{task["name"]}.log').open('a',encoding='utf-8')
            task['launch_log_start_bytes']=stream.tell()
            try:
                startup=subprocess.STARTUPINFO()
                startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startup.wShowWindow = 0
                process=subprocess.Popen(cmd,cwd=BASE.parent,stdout=stream,stderr=subprocess.STDOUT,
                    env=env,stdin=subprocess.DEVNULL,startupinfo=startup,
                    creationflags=subprocess.CREATE_NO_WINDOW,close_fds=True)
            except OSError as exc:
                stream.close();task.update(retry_after=time.time()+60,retry_reason=str(exc));continue
            task.setdefault('first_started_timestamp',time.time())
            task.update(status='running',pid=process.pid,process_created=process_info(process.pid)[1],
                        started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            active[task['name']]=(process,stream,task)
            publish(tasks,active,cfg)
            time.sleep(2);reap(active)
        counts=publish(tasks,active,cfg)
        if not active and counts['pending']==0:
            publish(tasks,active,cfg,'completed' if counts['failed']==0 else 'completed_with_failures')
            break
        time.sleep(max(1.,10-(time.monotonic()-tick)))


if __name__=='__main__':
    STATUS.mkdir(parents=True,exist_ok=True)
    sys.stdout=(STATUS/'coordinator_stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(STATUS/'coordinator_stderr.log').open('a',encoding='utf-8',buffering=1)
    with exclusive_lock(STATUS/'coordinator.lock'):
        main()
