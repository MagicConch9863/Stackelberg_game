"""Check the coordinator each minute; restart only an exited coordinator, not solvers."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from io_utils import atomic_json, atomic_text, exclusive_lock
from run_pilot import AttachedProcess

BASE = Path(__file__).resolve().parents[1]
INTERVAL = 60


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def main():
    status = BASE/'status'
    logs = BASE/'logs'
    logs.mkdir(exist_ok=True)
    process = None
    stream = None
    restarts = 0
    last_error = None
    while True:
        tick = time.monotonic()
        info = read_json(status/'coordinator.json')
        # A manual stop flag disables the watcher without killing any worker.
        if (status/'monitor.stop').exists():
            atomic_text(status/'monitor.log', 'Monitor stopped by monitor.stop; workers not terminated.\n')
            break
        if process is None and info.get('pid'):
            candidate = AttachedProcess(info['pid'])
            if candidate.poll() is None:
                process = candidate
        alive = process is not None and process.poll() is None
        if not alive:
            if stream is not None:
                stream.close()
                stream = None
            process = None
            info = read_json(status/'coordinator.json')
            if info.get('status') in ('completed', 'needs_attention'):
                summary = read_json(status/'suite_status.json').get('counts', {})
                if not summary.get('running',0) and not summary.get('pending',0):
                    atomic_text(status/'monitor.log',
                        f'Finished: {info["status"]}; counts={summary}; workers not terminated.\n')
                    break
            try:
                # The coordinator holds this lock throughout its lifetime.
                # Never spawn a second coordinator just because a heartbeat is old.
                with exclusive_lock(status/'coordinator.lock'):
                    pass
                env = os.environ.copy()
                env.update({key:'1' for key in
                    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
                stream = (logs/'coordinator_watch.log').open('a', encoding='utf-8')
                stream.write(f'\nMonitor launch {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
                stream.flush()
                process = subprocess.Popen(
                    [sys.executable,'-S','-u',str(BASE/'code/run_pilot.py')], cwd=BASE.parent,
                    stdout=stream, stderr=subprocess.STDOUT, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                restarts += 1
                alive = process.poll() is None
                last_error = None
            except (OSError, RuntimeError) as error:
                last_error = str(error)
                if stream is not None:
                    stream.close()
                    stream = None
        info = read_json(status/'coordinator.json')
        age = max(0, time.time()-info.get('checked_timestamp',0)) if info else None
        report = dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'), monitor_pid=os.getpid(),
            check_interval_s=INTERVAL, coordinator_pid=getattr(process,'pid',None),
            coordinator_alive=alive, coordinator_restarts=restarts,
            coordinator_heartbeat_age_s=age, last_error=last_error,
            note='Old heartbeat is a warning only; healthy solvers are never terminated.')
        atomic_json(status/'monitor.json', report)
        atomic_text(status/'monitor.log', '\n'.join(f'{key}: {value}' for key,value in report.items())+'\n')
        time.sleep(max(1,INTERVAL-(time.monotonic()-tick)))


if __name__ == '__main__':
    with exclusive_lock(BASE/'status/monitor.lock'):
        main()
