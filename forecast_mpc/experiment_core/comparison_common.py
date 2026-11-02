"""Shared result calculations for the soft-SOC experiment family."""

from __future__ import annotations

from experiment_core.stackelberg_mpc import BUS_CONFIG, PHASE4_SETTINGS


def soft_soc_values(bus: str, energy_kwh: float) -> tuple[float, float, float]:
    """Return lower slack, upper slack and linear soft-SOC penalty."""
    capacity = float(BUS_CONFIG[bus]["bess_kwh"])
    lower_slack = max(0.20 * capacity - energy_kwh, 0.0)
    upper_slack = max(energy_kwh - 0.80 * capacity, 0.0)
    penalty = float(PHASE4_SETTINGS["SOC_SOFT_PENALTY_EUR_PER_KWH"]) * (
        lower_slack + upper_slack
    )
    return lower_slack, upper_slack, penalty
