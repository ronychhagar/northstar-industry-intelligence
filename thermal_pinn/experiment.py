"""Train, compare, and plot the heat-diffusion PINN benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from thermal_pinn.model import fit_pinn
from thermal_pinn.solver import (
    analytical_temperature,
    solve_reference,
)


def run_experiment(
    steps: int = 2500,
    collocation_points: int = 512,
    spatial_points: int = 65,
    seed: int = 42,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    diffusivity = 0.1
    max_time = 0.5
    x = np.linspace(0.0, 1.0, spatial_points)
    times = np.linspace(0.0, max_time, 41)

    reference_solver = solve_reference(
        diffusivity=diffusivity,
        max_time=max_time,
        spatial_points=spatial_points,
    )
    reference = reference_solver.evaluate(x, times)
    analytical = analytical_temperature(x, times, diffusivity)
    model, training = fit_pinn(
        diffusivity=diffusivity,
        max_time=max_time,
        steps=steps,
        collocation_points=collocation_points,
        seed=seed,
    )
    coordinates = torch.tensor(
        np.stack(
            np.meshgrid(x, times, indexing="xy"), axis=-1
        ).reshape(-1, 2),
        dtype=torch.float32,
    )
    with torch.no_grad():
        pinn = model(coordinates).cpu().numpy().reshape(times.size, x.size)

    error = pinn - reference
    metrics: dict[str, float | int] = {
        "diffusivity": diffusivity,
        "max_time": max_time,
        "spatial_points": spatial_points,
        "training_steps": training.steps,
        "initial_pde_residual_rmse": training.initial_residual_rmse,
        "final_pde_residual_rmse": training.final_residual_rmse,
        "pinn_rmse_vs_scipy": float(np.sqrt(np.mean(error**2))),
        "pinn_max_abs_error_vs_scipy": float(np.max(np.abs(error))),
        "scipy_rmse_vs_analytical": float(
            np.sqrt(np.mean((reference - analytical) ** 2))
        ),
    }
    fields = {
        "x": x,
        "times": times,
        "pinn": pinn,
        "reference": reference,
        "analytical": analytical,
    }
    return metrics, fields


def save_figure(fields: dict[str, np.ndarray], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = fields["x"]
    times = fields["times"]
    pinn = fields["pinn"]
    reference = fields["reference"]
    error = pinn - reference

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure.suptitle("PINN vs. SciPy reference: 1-D transient heat conduction")
    axes[0, 0].plot(x, reference[-1], label="SciPy method of lines", linewidth=2)
    axes[0, 0].plot(
        x, fields["analytical"][-1], "--", label="Analytical", linewidth=1.5
    )
    axes[0, 0].plot(x, pinn[-1], ":", label="PINN", linewidth=2)
    axes[0, 0].set(
        title=f"Temperature at t = {times[-1]:.2f}",
        xlabel="Normalized position x",
        ylabel="Normalized temperature T",
    )
    axes[0, 0].legend()

    image_options = {
        "origin": "lower",
        "aspect": "auto",
        "extent": (x[0], x[-1], times[0], times[-1]),
    }
    pinn_image = axes[0, 1].imshow(pinn, **image_options)
    axes[0, 1].set(
        title="PINN temperature field",
        xlabel="Normalized position x",
        ylabel="Time t",
    )
    figure.colorbar(pinn_image, ax=axes[0, 1], label="T")

    reference_image = axes[1, 0].imshow(reference, **image_options)
    axes[1, 0].set(
        title="SciPy reference temperature field",
        xlabel="Normalized position x",
        ylabel="Time t",
    )
    figure.colorbar(reference_image, ax=axes[1, 0], label="T")

    error_image = axes[1, 1].imshow(
        np.abs(error), cmap="magma", **image_options
    )
    axes[1, 1].set(
        title="Absolute error |PINN - SciPy|",
        xlabel="Normalized position x",
        ylabel="Time t",
    )
    figure.colorbar(error_image, ax=axes[1, 1], label="|error|")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a heat-equation PINN and compare it with SciPy's "
            "method-of-lines solution."
        )
    )
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--collocation-points", type=int, default=512)
    parser.add_argument("--spatial-points", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("reports/thermal_pinn_vs_scipy.png"),
        help="path for the benchmark figure",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=0.03,
        help="maximum PINN-vs-SciPy RMSE required for a successful run",
    )
    args = parser.parse_args()
    if not np.isfinite(args.max_rmse) or args.max_rmse <= 0.0:
        parser.error("--max-rmse must be finite and positive.")

    metrics, fields = run_experiment(
        steps=args.steps,
        collocation_points=args.collocation_points,
        spatial_points=args.spatial_points,
        seed=args.seed,
    )
    save_figure(fields, args.plot)
    print(json.dumps(metrics, indent=2))
    print(f"Benchmark figure: {args.plot}")
    if metrics["pinn_rmse_vs_scipy"] > args.max_rmse:
        print(
            f"PINN RMSE exceeded the required threshold ({args.max_rmse:g})."
        )
        return 1
    return 0
