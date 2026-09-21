"""Shared SOC feasibility and soft-comfort-band formulation.

Physical feasibility is hard: 0 <= E <= capacity.  The preferred operating
band is soft: 20% <= E <= 80%.  Violations are represented by nonnegative
slacks and penalized linearly.  There is deliberately no time-dependent SOC
reference and no midnight-specific condition.
"""

from __future__ import annotations

import gurobipy as gp


def soc_limits(cfg, settings):
    capacity = float(cfg["bess_kwh"])
    physical_low = 0.0
    physical_high = capacity
    soft_low = float(settings.get("SOC_SOFT_MIN_FRAC", 0.20)) * capacity
    soft_high = float(settings.get("SOC_SOFT_MAX_FRAC", 0.80)) * capacity
    if not (physical_low <= soft_low < soft_high <= physical_high):
        raise ValueError("SOC fractions must satisfy 0 <= soft_min < soft_max <= 1")
    return physical_low, physical_high, soft_low, soft_high


def add_soft_soc_band(model, energy, horizon, cfg, settings, prefix):
    """Add soft [20%, 80%] constraints for E[1], ..., E[horizon]."""
    _, _, soft_low, soft_high = soc_limits(cfg, settings)
    violation_low = model.addVars(horizon, lb=0.0, name=f"soc_under_{prefix}")
    violation_high = model.addVars(horizon, lb=0.0, name=f"soc_over_{prefix}")
    for t in range(horizon):
        model.addConstr(energy[t + 1] + violation_low[t] >= soft_low)
        model.addConstr(energy[t + 1] - violation_high[t] <= soft_high)
    weight = float(settings.get("SOC_SOFT_PENALTY_EUR_PER_KWH", 1.0))
    penalty = weight * gp.quicksum(
        violation_low[t] + violation_high[t] for t in range(horizon)
    )
    return penalty, violation_low, violation_high

