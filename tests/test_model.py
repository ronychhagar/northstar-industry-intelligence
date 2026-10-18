import numpy as np
import pytest
import torch

from thermal_pinn.model import HeatPINN, fit_pinn, pde_residual


def test_pinn_exactly_enforces_initial_and_boundary_conditions() -> None:
    model = HeatPINN()
    x = torch.linspace(0.0, 1.0, 21)
    initial_coordinates = torch.stack((x, torch.zeros_like(x)), dim=1)
    initial_temperature = model(initial_coordinates).detach().numpy().ravel()

    torch.testing.assert_close(
        torch.tensor(initial_temperature),
        torch.sin(torch.pi * x),
    )
    boundary_coordinates = torch.tensor(
        [[0.0, 0.0], [0.0, 0.5], [1.0, 0.0], [1.0, 0.5]]
    )
    boundary_temperature = model(boundary_coordinates)
    torch.testing.assert_close(
        boundary_temperature, torch.zeros_like(boundary_temperature)
    )


def test_pde_residual_is_finite_and_supports_backpropagation() -> None:
    model = HeatPINN()
    coordinates = torch.tensor(
        [[0.2, 0.1], [0.5, 0.25], [0.8, 0.4]]
    )

    residual = pde_residual(model, coordinates)
    residual.square().mean().backward()

    assert residual.shape == (3, 1)
    assert torch.isfinite(residual).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_fit_pinn_reduces_sampled_physics_residual() -> None:
    _, metrics = fit_pinn(
        steps=1200,
        collocation_points=256,
        hidden_width=32,
        seed=42,
    )

    assert metrics.steps == 1200
    assert np.isfinite(metrics.final_residual_rmse)
    assert metrics.final_residual_rmse < metrics.initial_residual_rmse


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steps": 0}, "steps"),
        ({"collocation_points": 0}, "collocation_points"),
        ({"max_time": 0.0}, "max_time"),
    ],
)
def test_fit_pinn_rejects_invalid_inputs(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_pinn(**kwargs)
