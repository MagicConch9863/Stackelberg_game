"""Small solver check: objective identity, signed export and nonnegative peak."""
import numpy as np
import pandas as pd
from solve_window import solve_window, core


def main():
    settings=core.PHASE4_SETTINGS
    settings.update(GRID_EXCHANGE_MODE='grid_export',GUROBI_THREADS=1,SOLVER_OUTPUT=0,
                    LEADER_RAMP_WEIGHT=.05,BESS_DEGRADATION_EUR_PER_KWH=.05)
    for name,load,pv,anchor in [('import',2.,0.,8.),('export',0.,3.,-12.)]:
        forecasts={'load':{b:np.full(2,load) for b in core.BUS_ORDER},
                   'pv':{b:np.full(2,pv) for b in core.BUS_ORDER}}
        energies={b:float(core.BUS_CONFIG[b]['bess_e_init']) for b in core.BUS_ORDER}
        result=solve_window(pd.Timestamp('2025-01-15'),0,forecasts,energies,np.full(2,.1),None,None,
                            mip_gap=.1,time_limit_s=30,previous_realized_net=anchor)
        grid=result['planned_grid']
        expected=max(0,float(max(grid)))+.05*float(np.abs(np.diff(np.r_[anchor,grid])).sum())
        assert abs(result['objective']-expected)<1e-4, (result['objective'],expected)
        assert abs(result['peak_component']-max(0,float(max(grid))))<1e-4
        if name=='export':
            assert max(grid)<0
            assert abs(result['peak_component'])<1e-5
        print(name,'passed; objective=',round(result['objective'],6),'gap=',result['gap'],flush=True)


if __name__=='__main__':
    main()
