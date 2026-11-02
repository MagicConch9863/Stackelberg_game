"""Causal forecasts: start denotes the first interval that has not yet occurred."""


def forecast_window(actual, method, scenario, start, horizon, rho):
    import numpy as np
    if start < 1 or method not in ('OL', 'MPC') or scenario not in ('perfect', 'iid', 'ar1'):
        raise ValueError('Require one completed history interval and a valid method/scenario')
    result = {'load': {}, 'pv': {}}
    for bus, nominal in actual['nominal_load'].items():
        if start+horizon > len(nominal):
            raise ValueError('Incomplete prediction horizon')
        # Only the completed interval is observed. No actual[start:] is accessed.
        residual = float(actual['load'][bus][start-1]-nominal[start-1])
        correction = rho**np.arange(1, horizon+1)*residual if method == 'MPC' and scenario == 'ar1' else 0.0
        result['load'][bus] = np.maximum(0, nominal[start:start+horizon]+correction).copy()
        result['pv'][bus] = actual['pv'][bus][start:start+horizon].copy()
    return result
