"""Resume the pilot and additional per-seed cases under one shared concurrency limit."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from io_utils import atomic_json, atomic_text, exclusive_lock
from postrun_review import (FORMAL_OBJECTIVES, add_audit_tasks, background_slots,
                            discover_replays, review_queue, source_complete, add_followup_tasks)

BASE = Path(__file__).resolve().parents[1]
CHECK_INTERVAL_S = 60


def finish_task(task, returncode):
    """Retry startup failures, but retain bounded retries for actual solver errors."""
    task.update(status='completed' if returncode == 0 else 'failed', exit_code=returncode,
                finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
    case_log = Path(task['suite_base'])/'logs'/f'{task["name"]}.log'
    empty_attempt = case_log.exists() and case_log.stat().st_size == task.get('launch_log_start_bytes', 0)
    if returncode == 75:
        task.update(status='pending', retry_after=time.time()+300, retry_reason='waiting_for_source')
    elif returncode != 0 and empty_attempt:
        # An interpreter that died before producing any output is not evidence
        # of a mathematical infeasibility. Do not permanently abandon this case.
        task['previous_exit_codes'] = (task.get('previous_exit_codes', [])+[returncode])[-50:]
        task['startup_deferrals'] = task.get('startup_deferrals', 0)+1
        task.update(status='pending', retry_after=time.time()+min(300,60*task['startup_deferrals']),
                    retry_reason='interpreter_startup')
    elif returncode != 0 and task.get('retries', 0) < 2:
        task['previous_exit_codes'] = (task.get('previous_exit_codes', [])+[returncode])[-50:]
        task.update(status='pending', retries=task.get('retries', 0)+1,
                    retry_after=time.time()+60, retry_reason='case_error')
    elif returncode != 0:
        task['retry_reason'] = 'case_error_requires_attention'


def reap_processes(active):
    for name, (process, stream, task) in list(active.items()):
        returncode = process.poll()
        if returncode is not None:
            if stream is not None:
                stream.close()
            finish_task(task, returncode)
            del active[name]


def save_tasks(folder, tasks):
    counts = {s:sum(t['status']==s for t in tasks)
              for s in ('completed','running','pending','failed','blocked')}
    atomic_json(folder/'suite_status.json', dict(counts=counts, cases=tasks))
    return counts


def worker_command(python, task, family, seed):
    script = task.get('script',str(Path(task['suite_base'])/'code/run_case.py'))
    arguments = ['--input', str(BASE/'inputs'/f'{task["season"]}.csv'),
                 '--season', task['season'], '--family', task.get('family',family),
                 '--scenario', task['scenario'], '--method', task['method'],
                 '--seed', str(task.get('seed',seed)), '--execute']+task.get('extra_args',[])
    if task.get('startup_deferrals',0) >= 2 and not task.get('isolated_startup'):
        # Avoid site startup hooks after repeated silent interpreter exits.
        # PYTHONPATH still supplies the identical scientific packages; restore
        # builtin help expected by Gurobi without running site.main().
        bootstrap = ('import pathlib,runpy,site,sys;site.sethelper();'
                     'entry=sys.argv.pop(1);sys.path.insert(0,str(pathlib.Path(entry).parent));'
                     'sys.argv[0]=entry;runpy.run_path(entry,run_name="__main__")')
        task['startup_mode'] = 'without_site_hooks'
        return [python,'-S','-u','-c',bootstrap,script]+arguments
    task['startup_mode'] = 'isolated' if task.get('isolated_startup') else 'normal'
    return [python]+(['-S'] if task.get('isolated_startup') else [])+['-u',script]+arguments


class AttachedProcess:
    """Monitor a saved child PID when the coordinator alone is restarted on Windows."""
    def __init__(self, pid):
        import ctypes
        from ctypes import wintypes
        self.pid = pid
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.OpenProcess(0x1000, False, pid)

    def poll(self):
        import ctypes
        from ctypes import wintypes
        if not self.handle:
            return 1
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            return 1
        if code.value == 259:
            return None
        self.kernel.CloseHandle(self.handle)
        self.handle = None
        return code.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--retry-replays', action='store_true')
    args = parser.parse_args()
    cfg = json.loads((BASE/'configs/experiment.json').read_text())
    folder = BASE/'status'
    folder.mkdir(exist_ok=True)
    logs = BASE/'logs'
    logs.mkdir(exist_ok=True)
    family = cfg['pilot_family']
    seed = cfg['seeds'][family][0]
    suites = [('p2v_ramp', BASE)]
    peak_base = BASE.parent/'four_hour_peak_ramp'
    if (peak_base/'configs/experiment.json').exists():
        suites.append(('peak_ramp',peak_base))
    grouped = {}
    for objective, suite_base in suites:
        (suite_base/'logs').mkdir(exist_ok=True)
        grouped[objective] = []
        for scenario in ('perfect', 'iid', 'ar1'):
            for season in cfg['dates']:
                for method in ('OL', 'MPC'):
                    case = f'{season}_{scenario}_{method}'
                    name = case if objective == 'p2v_ramp' else f'peak_ramp_{case}'
                    output = suite_base/'results'/family/season/scenario/f'seed_{seed}'/method
                    grouped[objective].append(dict(name=name, season=season, scenario=scenario, method=method,
                        objective=objective, suite_base=str(suite_base), output=str(output),
                        family=family, seed=seed, status='pending'))
    # Put one new case into the next free slot, then alternate pending suites.
    tasks = []
    for i in range(len(grouped['p2v_ramp'])):
        if 'peak_ramp' in grouped:
            tasks.append(grouped['peak_ramp'][i])
        tasks.append(grouped['p2v_ramp'][i])
    additional_path = BASE/'configs/additional_tasks.json'
    if additional_path.exists():
        tasks = json.loads(additional_path.read_text())+tasks
    add_audit_tasks(tasks,BASE)
    # Replay jobs are discovered after their independent audits finish, so they
    # may exist only in the persisted queue rather than additional_tasks.json.
    saved_path = folder/'suite_status.json'
    if saved_path.exists():
        known={t['name'] for t in tasks}
        for old in json.loads(saved_path.read_text())['cases']:
            if old['name'] not in known and (old.get('objective','').startswith('replay_') or old['name'].startswith('followup_')):
                followup=old['name'].startswith('followup_')
                expected_script=BASE/'review/all_cases/code/run_case.py' if followup else BASE/'code/postrun_review.py'
                expected_root=BASE/'review/all_cases' if followup else BASE/'review/replayed'
                if (Path(old.get('script','')).resolve()!=expected_script.resolve()
                        or not Path(old['output']).resolve().is_relative_to(expected_root.resolve())):
                    raise ValueError('Unexpected replay job paths in saved queue')
                tasks.append(old)
                known.add(old['name'])
    atomic_json(folder/'task_plan.json', tasks)
    active = {}
    if saved_path.exists():
        saved = {t['name']:t for t in json.loads(saved_path.read_text())['cases']}
        for task in tasks:
            old = saved.get(task['name'], {})
            retry_output=task['output'] if task['name'].startswith('retry_tiered_') else None
            task.update(old)
            if retry_output and task['output']!=retry_output:
                task['output']=retry_output
                if task['status']=='completed':task.update(status='pending',retries=0,retry_after=0)
            if task['status'] == 'failed' and (args.retry_failed or (args.retry_replays and task['name'].startswith('replay_'))):
                task.setdefault('recovery_history', []).append(dict(exit_code=task.get('exit_code'),
                    retries=task.get('retries',0), recovered_at=time.strftime('%Y-%m-%d %H:%M:%S')))
                task.update(status='pending', retries=0, retry_after=0)
            if task['status'] == 'running' and sys.platform == 'win32':
                process = AttachedProcess(task['pid'])
                if process.poll() is None:
                    active[task['name']] = (process, None, task)
                else:
                    task['status'] = 'pending'
            elif task['status'] == 'failed' and task.get('retries',0) < 2:
                task['status'] = 'pending'
    child_env = os.environ.copy()
    child_env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    child_env['PYTHONPATH'] = str(BASE.parent/'venv/Lib/site-packages')+os.pathsep+child_env.get('PYTHONPATH','')
    python = str(getattr(sys, '_base_executable', sys.executable))
    while True:
        cycle_start = time.monotonic()
        atomic_json(folder/'coordinator.json', dict(pid=os.getpid(), status='checking',
            checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), checked_timestamp=time.time(),
            check_interval_s=CHECK_INTERVAL_S))
        runtime_path = BASE/'configs/runtime.json'
        if runtime_path.exists():
            cfg['parallel_cases'] = int(json.loads(runtime_path.read_text())['parallel_cases'])
            if cfg['parallel_cases'] < 1:
                raise ValueError('parallel_cases must be positive')
        priority_review = any(t.get('objective')=='p2v_ramp_review' and
            t['status'] in ('pending','running') and t.get('current_step',107)<=108 for t in tasks)
        effective_parallel = min(cfg['parallel_cases'],14) if priority_review else cfg['parallel_cases']
        reap_processes(active)
        task_count_before=len(tasks)
        discover_replays(tasks,BASE)
        add_followup_tasks(tasks,BASE)
        if len(tasks)!=task_count_before:
            atomic_json(folder/'task_plan.json',tasks)
        by_name={t['name']:t for t in tasks}
        for task in tasks:
            if task.get('opportunistic') and task['status'] in ('pending','blocked'):
                upstream=by_name.get(task.get('source_task_name'),{})
                if upstream.get('status') in ('failed','blocked') and not source_complete(task):
                    task.update(status='blocked',retry_reason='source_requires_attention')
                elif task['status']=='blocked':
                    task.update(status='pending',retry_after=0)
        for task in review_queue(tasks,active,effective_parallel):
            if time.monotonic()-cycle_start >= 40:
                # Leave time for liveness reconciliation and the minute heartbeat
                # even when many interpreters fail to start in the same cycle.
                break
            if len(active) >= effective_parallel:
                break
            if task['status'] != 'pending':
                continue
            if task.get('opportunistic'):
                active_background=sum(bool(item[2].get('opportunistic')) for item in active.values())
                if active_background>=background_slots(tasks,effective_parallel) or not source_complete(task):
                    continue
            if priority_review and task['name']=='peak_ramp_summer_perfect_MPC':
                continue
            if priority_review and not any(t.get('objective')=='p2v_ramp_review' and t['status']=='running' for t in tasks) and task.get('objective')!='p2v_ramp_review':
                continue
            if time.time() < task.get('retry_after',0):
                continue
            output = Path(task['output'])
            if (output/'metrics.json').exists():
                task.update(status='completed', resumed_completed=True)
                continue
            suite_base = Path(task['suite_base'])
            command = worker_command(python, task, family, seed)
            stream = (suite_base/'logs'/f'{task["name"]}.log').open('a', encoding='utf-8')
            task['launch_log_start_bytes'] = stream.tell()
            try:
                process = subprocess.Popen(command, cwd=BASE.parent, stdout=stream, stderr=subprocess.STDOUT, env=child_env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
            except OSError as exc:
                stream.close()
                task.setdefault('launch_errors',[]).append(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),
                    error=str(exc),winerror=getattr(exc,'winerror',None)))
                task.update(status='pending',retry_after=time.time()+60)
                continue
            task.update(status='running', pid=process.pid, started_at=time.strftime('%Y-%m-%d %H:%M:%S'))
            active[task['name']] = (process, stream, task)
            # Persist the PID before any stagger delay, so a coordinator-only
            # restart can reattach instead of launching a duplicate worker.
            save_tasks(folder, tasks)
            # Stagger heavy scientific imports instead of starting a burst of interpreters.
            time.sleep(2)
            reap_processes(active)
        # In particular, do not report an interpreter that died during startup
        # as a running solver occupying one of the sixteen slots.
        reap_processes(active)
        counts = save_tasks(folder, tasks)
        waiting_retry = sum(t['status']=='pending' and t.get('retry_after',0)>time.time() for t in tasks)
        formal_remaining=sum(t.get('objective') in FORMAL_OBJECTIVES and t['status'] in ('pending','running') for t in tasks)
        review_budget=background_slots(tasks,effective_parallel)
        lines = [f'updated: {time.strftime("%Y-%m-%d %H:%M:%S")}', f'counts: {counts}',
                 f'family: {family}; seeds: {sorted({t.get("seed", seed) for t in tasks})}; '
                 f'parallel requested: {cfg["parallel_cases"]}; effective: {effective_parallel}',
                 f'live processes verified: {len(active)}; free slots: {max(0,effective_parallel-len(active))}; '
                 f'pending in retry cooldown: {waiting_retry}; check interval: {CHECK_INTERVAL_S}s',
                 f'formal cases remaining: {formal_remaining}; background review slot budget: {review_budget}; '
                 'formal simulations have priority']
        for task in tasks:
            if task['status'] == 'running':
                progress = Path(task['output'])/'progress.json'
                try:
                    data = json.loads(progress.read_text())
                    task['current_step'] = data['current_step']
                    task['total_steps'] = data.get('total',304)
                except (OSError, ValueError, KeyError):
                    pass
                lines.append(f'{task["name"]}: {task.get("current_step", 0)}/{task.get("total_steps",304)}')
            elif task['status'] == 'failed':
                lines.append(f'FAILED {task["name"]}: exit={task["exit_code"]}')
            elif task['status']=='blocked':
                lines.append(f'BLOCKED {task["name"]}: {task.get("retry_reason")}')
        save_tasks(folder, tasks)
        atomic_text(folder/'terminal_status.log', '\n'.join(lines)+'\n')
        atomic_json(folder/'review_queue.json',dict(
            formal_remaining=formal_remaining,background_slot_budget=review_budget,
            policy='reserve formal capacity; run completed-source reviews in tail slots',
            jobs=[dict(name=t['name'],status=t['status'],source=t.get('requires_completed_source'),
                       output=t['output'],replay_decision=t.get('replay_decision'))
                  for t in tasks if t.get('opportunistic')]))
        for objective, suite_base in suites:
            selected = [t for t in tasks if t['objective'] == objective]
            suite_counts = {s:sum(t['status']==s for t in selected) for s in counts}
            if objective == 'peak_ramp':
                status_folder = suite_base/'status'
                status_folder.mkdir(exist_ok=True)
                atomic_json(status_folder/'suite_status.json',dict(counts=suite_counts,cases=selected))
                suite_lines = [lines[0], f'objective: {objective}', f'counts: {suite_counts}',
                    'Shared total concurrency limit: '+str(cfg['parallel_cases'])]
                suite_lines.extend(f'{t["name"]}: {t["status"]} {t.get("current_step",0)}/304' for t in selected)
                atomic_text(status_folder/'terminal_status.log','\n'.join(suite_lines)+'\n')
        undecided_reviews=any(t.get('objective','').startswith('audit_') and t['status']=='completed'
                              and not t.get('replay_decision') for t in tasks)
        finished = not active and not counts['pending'] and not undecided_reviews
        atomic_json(folder/'coordinator.json', dict(pid=os.getpid(),
            status=('needs_attention' if counts['failed'] or counts['blocked'] else 'completed') if finished else 'running',
            checked_at=time.strftime('%Y-%m-%d %H:%M:%S'), checked_timestamp=time.time(),
            check_interval_s=CHECK_INTERVAL_S, live_processes=len(active), counts=counts,
            free_slots=max(0,effective_parallel-len(active)), waiting_retry=waiting_retry))
        if finished:
            break
        time.sleep(max(1, CHECK_INTERVAL_S-(time.monotonic()-cycle_start)))


if __name__ == '__main__':
    with exclusive_lock(BASE/'status/coordinator.lock'):
        main()
