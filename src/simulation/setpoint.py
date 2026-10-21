"""Synthetic coolant-flow recommendation.

The loss is a typed stand-in for a plant model. It is not a fab process model.
"""

from __future__ import annotations

from scipy.optimize import minimize_scalar


def recommend_flow(temperature_c: float, target_temperature_c: float) -> dict:
    excess = float(temperature_c) - float(target_temperature_c)

    def loss(flow: float) -> float:
        residual = excess * (1.0 - float(flow))
        return residual * residual + 0.01 * float(flow) * float(flow)

    found = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    if not found.success:
        raise RuntimeError("setpoint optimization failed")
    return {
        "tool": "scipy.optimize.minimize_scalar",
        "flow_fraction": round(float(found.x), 6),
        "loss": round(float(found.fun), 8),
        "excess_c": round(excess, 6),
        "applied": False,
    }
