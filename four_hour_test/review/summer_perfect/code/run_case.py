"""Resume the original P2V summer Perfect case before origin 107."""
from pathlib import Path
import site
import sys

if not hasattr(__builtins__, 'help'):
    site.sethelper()
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE.parents[1]/'code'))
import run_case as shared

if __name__ == '__main__':
    shared.main(experiment_base=BASE)
