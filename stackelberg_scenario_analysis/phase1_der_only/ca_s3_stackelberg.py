"""Compatibility entry point for the complete 96-PB S3 model.

The complete Gurobi/Big-M/KKT implementation is kept in
ca_s3_standalone_96pb.py.  This filename is the normal S3 filename used by the
project and avoids importing helpers from another scenario script.
"""

from ca_s3_standalone_96pb import main


if __name__ == "__main__":
    SETTINGS = {
        "ENABLE_PV": False,
        "PV_FILE": "pv_2019-07-01_local.csv",
        "PV_COLUMN": "electricity",
        "PRICE_FILE": "price_data.csv",
        "PRICE_COLUMN": "price_eur_per_mwh_1",
        "PRICE_UNIT": "EUR/MWH",
        "PB_MIN": 0.0,
        "PB_MAX": 0.30,
        "PB_RAMP": 0.15,
        "PB_AVERAGE": 0.10,
        "PB_AVERAGE_TOL": 0.01,
        "GRID_BOUND_KW": 20.0,
        "DUAL_BIG_M": 1000.0,
        "MIP_GAP": 0.001,
        "TIME_LIMIT_S": 1200,
        "SOLVER_OUTPUT": 1,
    }
    main(SETTINGS)
