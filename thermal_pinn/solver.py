"""Method-of-lines reference solution and analytical validation for 1-D diffusion."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ReferenceSolution:
    """Dense SciPy solution sampled on the reference solver's spatial grid."""

    x_nodes: NDArray[np.float64]
    _dense_solution: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    max_time: float

    def evaluate(
        self, x: ArrayLike, times: ArrayLike
    ) -> NDArray[np.float64]:
        """Return temperatures with shape ``(len(times), len(x))``."""
        positions = np.asarray(x, dtype=float).reshape(-1)
        time_values = np.asarray(times, dtype=float).reshape(-1)
        if not np.all(np.isfinite(positions)) or np.any(
            (positions < 0.0) | (positions > 1.0)
        ):
            raise ValueError("x coordinates must be finite and in [0, 1].")
        if not np.all(np.isfinite(time_values)) or np.any(
            (time_values < 0.0) | (time_values > self.max_time)
        ):
            raise ValueError(
                f"times must be finite and in [0, {self.max_time}]."
            )

        states = self._dense_solution(time_values)
        temperatures = np.empty((time_values.size, positions.size), dtype=float)
        reference_x = np.concatenate(([0.0], self.x_nodes, [1.0]))
        for index, state in enumerate(states.T):
            reference_temperature = np.concatenate(([0.0], state, [0.0]))
            temperatures[index] = np.interp(
                positions, reference_x, reference_temperature
            )
        return temperatures


def solve_reference(
    diffusivity: float = 0.1,
    max_time: float = 0.5,
    spatial_points: int = 65,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> ReferenceSolution:
    """Solve the heat equation with a central-difference spatial discretization.

    The boundary temperatures are fixed at zero, and the interior initial
    temperature is ``sin(pi*x)``. SciPy's adaptive DOP853 integrator advances
    the resulting system of ordinary differential equations in time.
    """
    if not np.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError("diffusivity must be finite and positive.")
    if not np.isfinite(max_time) or max_time <= 0.0:
        raise ValueError("max_time must be finite and positive.")
    if spatial_points < 3:
        raise ValueError("spatial_points must be at least 3.")
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError("rtol must be finite and positive.")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be finite and positive.")

    full_grid = np.linspace(0.0, 1.0, spatial_points)
    x_nodes = full_grid[1:-1]
    dx = full_grid[1] - full_grid[0]
    initial_temperature = np.sin(np.pi * x_nodes)

    def heat_equation_rhs(
        _time: float, temperature: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        padded_temperature = np.pad(temperature, (1, 1), mode="constant")
        second_derivative = (
            padded_temperature[2:]
            - 2.0 * padded_temperature[1:-1]
            + padded_temperature[:-2]
        ) / dx**2
        return diffusivity * second_derivative

    solution = solve_ivp(
        heat_equation_rhs,
        (0.0, max_time),
        initial_temperature,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"SciPy reference solve failed: {solution.message}")
    return ReferenceSolution(
        x_nodes=x_nodes,
        _dense_solution=solution.sol,
        max_time=max_time,
    )


def analytical_temperature(
    x: ArrayLike, times: ArrayLike, diffusivity: float
) -> NDArray[np.float64]:
    """Return the exact single-mode heat-equation solution on a unit slab."""
    positions = np.asarray(x, dtype=float).reshape(-1)
    time_values = np.asarray(times, dtype=float).reshape(-1)
    if not np.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError("diffusivity must be finite and positive.")
    if not np.all(np.isfinite(positions)) or np.any(
        (positions < 0.0) | (positions > 1.0)
    ):
        raise ValueError("x coordinates must be finite and in [0, 1].")
    if not np.all(np.isfinite(time_values)) or np.any(time_values < 0.0):
        raise ValueError("times must be finite and nonnegative.")

    decay = np.exp(
        -diffusivity * np.pi**2 * time_values[:, np.newaxis]
    )
    return decay * np.sin(np.pi * positions[np.newaxis, :])
