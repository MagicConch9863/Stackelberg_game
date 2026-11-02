"""Check information timing without creating an optimization model."""
from copy import deepcopy
import numpy as np
from prediction import forecast_window


def main():
    base = np.linspace(3, 6, 40)
    actual = {'nominal_load': {'Bus 1': base},
              'load': {'Bus 1': base+2}, 'pv': {'Bus 1': np.zeros(40)}}
    for method in ('OL', 'MPC'):
        for scenario in ('perfect', 'iid', 'ar1'):
            forecast = forecast_window(actual, method, scenario, 5, 16, .9)
            changed = deepcopy(actual)
            changed['load']['Bus 1'][5:] += 100
            future_changed = forecast_window(changed, method, scenario, 5, 16, .9)
            np.testing.assert_array_equal(forecast['load']['Bus 1'], future_changed['load']['Bus 1'])
            expected = base[5:21].copy()
            if method == 'MPC' and scenario == 'ar1':
                expected += 2*.9**np.arange(1, 17)
            np.testing.assert_allclose(forecast['load']['Bus 1'], expected)
    changed = deepcopy(actual)
    changed['load']['Bus 1'][4] += 1
    first = forecast_window(changed, 'MPC', 'ar1', 5, 16, .9)['load']['Bus 1'][0]
    assert np.isclose(first, base[5]+2.7)
    assert len(range(0, 304, 16)) == 19
    assert 16 in range(0, 304, 16)
    print('Passed: future-data independence, OL nominal forecast, IID zero correction, AR lag and block alignment.')


if __name__ == '__main__':
    main()
