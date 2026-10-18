"""A fully connected PINN for transient heat conduction in a unit slab."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


class HeatPINN(nn.Module):
    """Approximate temperature while exactly enforcing the initial and boundary data."""

    def __init__(self, diffusivity: float = 0.1, hidden_width: int = 48) -> None:
        super().__init__()
        if not np.isfinite(diffusivity) or diffusivity <= 0.0:
            raise ValueError("diffusivity must be finite and positive.")
        if hidden_width < 1:
            raise ValueError("hidden_width must be positive.")
        self.diffusivity = float(diffusivity)

        layers: list[nn.Module] = []
        input_width = 2
        for _ in range(3):
            layers.extend((nn.Linear(input_width, hidden_width), nn.Tanh()))
            input_width = hidden_width
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)
        self.network.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        x = coordinates[:, 0:1]
        time = coordinates[:, 1:2]
        correction = time * x * (1.0 - x) * self.network(coordinates)
        return torch.sin(torch.pi * x) + correction


def pde_residual(model: HeatPINN, coordinates: Tensor) -> Tensor:
    """Evaluate ``T_t - alpha*T_xx`` at differentiable (x, t) coordinates."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (n, 2) in (x, t) order.")
    differentiable_coordinates = coordinates.detach().clone().requires_grad_(True)
    temperature = model(differentiable_coordinates)
    derivatives = torch.autograd.grad(
        temperature,
        differentiable_coordinates,
        grad_outputs=torch.ones_like(temperature),
        create_graph=True,
    )[0]
    spatial_gradient = derivatives[:, 0:1]
    spatial_second_derivative = torch.autograd.grad(
        spatial_gradient,
        differentiable_coordinates,
        grad_outputs=torch.ones_like(spatial_gradient),
        create_graph=True,
    )[0][:, 0:1]
    time_derivative = derivatives[:, 1:2]
    return time_derivative - model.diffusivity * spatial_second_derivative


@dataclass(frozen=True)
class TrainingMetrics:
    """Training residual diagnostics for the final sampled collocation batch."""

    steps: int
    initial_residual_rmse: float
    final_residual_rmse: float


def fit_pinn(
    diffusivity: float = 0.1,
    max_time: float = 0.5,
    steps: int = 2500,
    collocation_points: int = 512,
    hidden_width: int = 48,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> tuple[HeatPINN, TrainingMetrics]:
    """Train on the heat-equation residual without using reference-solver data."""
    if not np.isfinite(max_time) or max_time <= 0.0:
        raise ValueError("max_time must be finite and positive.")
    if steps < 1:
        raise ValueError("steps must be positive.")
    if collocation_points < 1:
        raise ValueError("collocation_points must be positive.")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")

    torch.manual_seed(seed)
    model = HeatPINN(diffusivity=diffusivity, hidden_width=hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    initial_residual_rmse: float | None = None
    final_residual_rmse = float("nan")

    model.train()
    for step in range(steps):
        coordinates = torch.rand(
            collocation_points, 2, generator=generator, dtype=torch.float32
        )
        coordinates[:, 1] *= max_time
        residual = pde_residual(model, coordinates)
        loss = torch.mean(residual.square())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        residual_rmse = float(torch.sqrt(loss.detach()).item())
        if step == 0:
            initial_residual_rmse = residual_rmse
        final_residual_rmse = residual_rmse

    if initial_residual_rmse is None:
        raise RuntimeError("PINN training did not execute any optimization steps.")
    model.eval()
    return model, TrainingMetrics(
        steps=steps,
        initial_residual_rmse=initial_residual_rmse,
        final_residual_rmse=final_residual_rmse,
    )
