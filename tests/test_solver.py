import numpy as np
import pytest

from thermal_pinn.solver import analytical_temperature, solve_reference


def test_method_of_lines_matches_analytical_heat_solution() -> None:
    x = np.linspace(0.0, 1.0, 65)
    times = np.linspace(0.0, 0.5, 21)
    numerical = solve_reference(spatial_points=x.size).evaluate(x, times)
    analytical = analytical_temperature(x, times, diffusivity=0.1)

    assert numerical.shape == (times.size, x.size)
    assert np.max(np.abs(numerical[:, [0, -1]])) == 0.0
    assert np.sqrt(np.mean((numerical - analytical) ** 2)) < 1e-4


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diffusivity": 0.0}, "diffusivity"),
        ({"max_time": 0.0}, "max_time"),
        ({"spatial_points": 2}, "spatial_points"),
    ],
)
def test_reference_solver_rejects_invalid_inputs(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solve_reference(**kwargs)


def test_reference_evaluation_rejects_out_of_domain_points() -> None:
    reference = solve_reference()
    with pytest.raises(ValueError, match=r"x coordinates"):
        reference.evaluate([-0.1], [0.0])
    with pytest.raises(ValueError, match="times"):
        reference.evaluate([0.5], [0.6])
