"""Physics-informed heat-diffusion benchmark."""

from thermal_pinn.model import HeatPINN, fit_pinn, pde_residual
from thermal_pinn.solver import (
    ReferenceSolution,
    analytical_temperature,
    solve_reference,
)

__all__ = [
    "HeatPINN",
    "ReferenceSolution",
    "analytical_temperature",
    "fit_pinn",
    "pde_residual",
    "solve_reference",
]
