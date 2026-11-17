"""Verify budget migration to 3600 s without running a solver."""
from pathlib import Path
from tempfile import TemporaryDirectory
import run_suite as suite

with TemporaryDirectory() as folder:
    suite.BASE=Path(folder)
    (suite.BASE/'logs').mkdir()
    (suite.BASE/'logs/case.log').write_text('No incumbent after 1800 s')
    task=dict(name='case',method='OL',output=str(suite.BASE/'output'),
              retries=1,startup_deferrals=0,launch_log_start_bytes=0,ol_time_limit_s=2400)
    suite.finish(task,1)
    assert task['status']=='pending' and task['ol_time_limit_s']==3600
    assert task['retries']==2
    suite.finish(task,1)
    assert task['status']=='failed', '3600 s failure must not retry forever'
    empty=dict(name='empty',method='OL',output=str(suite.BASE/'empty'),
               retries=0,startup_deferrals=0)
    suite.finish(empty,9)
    assert empty['status']=='pending' and 'ol_time_limit_s' not in empty
print('OL migration to 3600 s; terminal failure and interpreter-startup handling verified.')
