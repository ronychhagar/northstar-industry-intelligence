"""Calibrate and simulate a differential-drive robot from encoder and IMU logs."""

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

_CSV_COLUMNS = (
    "time_s",
    "left_encoder_rad",
    "right_encoder_rad",
    "gyro_z_rad_s",
)


@dataclass(frozen=True)
class DriveLog:
    """Timestamped wheel angles and measured vertical-axis IMU angular rate."""

    time_s: ArrayLike
    left_encoder_rad: ArrayLike
    right_encoder_rad: ArrayLike
    gyro_z_rad_s: ArrayLike

    def __post_init__(self) -> None:
        validated = {
            name: np.asarray(getattr(self, name), dtype=float).reshape(-1)
            for name in _CSV_COLUMNS
        }
        lengths = {values.size for values in validated.values()}
        if len(lengths) != 1:
            raise ValueError("All sensor columns must contain the same number of rows.")
        if not lengths or next(iter(lengths)) < 12:
            raise ValueError("The sensor log must contain at least 12 samples.")
        if any(not np.all(np.isfinite(values)) for values in validated.values()):
            raise ValueError("Sensor values must all be finite; remove missing or infinite readings.")
        if np.any(np.diff(validated["time_s"]) <= 0.0):
            raise ValueError("time_s must be strictly increasing, without duplicate timestamps.")

        for name, values in validated.items():
            values = values.copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)


def load_sensor_csv(source: str | bytes | Path) -> DriveLog:
    """Parse a UTF-8 CSV with the four documented sensor columns."""
    if isinstance(source, bytes):
        try:
            text = source.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("The sensor CSV must be UTF-8 encoded.") from error
    elif isinstance(source, Path):
        try:
            text = source.read_text(encoding="utf-8-sig")
        except OSError as error:
            raise ValueError(f"Could not read sensor CSV: {error}") from error
    else:
        text = source

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ValueError("The sensor CSV must include a header row.")
    normalized_headers = [header.strip().lower() for header in reader.fieldnames]
    if len(normalized_headers) != len(set(normalized_headers)):
        raise ValueError("Sensor CSV column names must be unique.")
    missing_columns = sorted(set(_CSV_COLUMNS) - set(normalized_headers))
    if missing_columns:
        raise ValueError(
            "Sensor CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    values: dict[str, list[float]] = {column: [] for column in _CSV_COLUMNS}
    for row_number, row in enumerate(reader, start=2):
        normalized_row = {
            (key or "").strip().lower(): value for key, value in row.items()
        }
        for column in _CSV_COLUMNS:
            try:
                values[column].append(float(normalized_row[column] or ""))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Row {row_number} has a missing or invalid {column} reading."
                ) from error
    return DriveLog(**values)


@dataclass(frozen=True)
class DriveModel:
    """Calibrated differential-drive geometry and held-out fit metrics."""

    wheel_radius_m: float
    track_width_m: float
    gyro_bias_rad_s: float
    training_gyro_rmse_rad_s: float
    validation_gyro_rmse_rad_s: float
    training_intervals: int
    validation_intervals: int

    @property
    def radius_over_track_width(self) -> float:
        return self.wheel_radius_m / self.track_width_m


@dataclass(frozen=True)
class DriveSimulation:
    """Pose and wheel-derived yaw-rate predictions at each input sample."""

    time_s: NDArray[np.float64]
    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    yaw_rad: NDArray[np.float64]
    predicted_gyro_z_rad_s: NDArray[np.float64]
    measured_yaw_rad: NDArray[np.float64]


def fit_drive_model(
    drive_log: DriveLog,
    wheel_radius_m: float,
    train_fraction: float = 0.7,
) -> DriveModel:
    """Estimate effective track width from wheel encoders and gyro yaw rate.

    Fits the differential-drive equation
    ``gyro_z = wheel_radius / track_width * (right_rate - left_rate) + bias``
    using a robust training-period regression, then reports gyro RMSE on a
    separate, later validation period. The wheel radius must be supplied.
    """
    if not np.isfinite(wheel_radius_m) or wheel_radius_m <= 0.0:
        raise ValueError("wheel_radius_m must be finite and positive.")
    if (
        not np.isfinite(train_fraction)
        or not 0.5 <= train_fraction <= 0.9
    ):
        raise ValueError("train_fraction must be finite and between 0.5 and 0.9.")

    split_index = int(drive_log.time_s.size * train_fraction)
    if split_index < 6 or drive_log.time_s.size - split_index < 5:
        raise ValueError(
            "The log is too short for the selected training fraction; "
            "provide at least 12 samples."
        )

    elapsed = np.diff(drive_log.time_s)
    wheel_difference_rate = (
        np.diff(drive_log.right_encoder_rad)
        - np.diff(drive_log.left_encoder_rad)
    ) / elapsed
    measured_gyro_rate = 0.5 * (
        drive_log.gyro_z_rad_s[:-1] + drive_log.gyro_z_rad_s[1:]
    )
    training_end = split_index - 1
    training_x = wheel_difference_rate[:training_end]
    training_y = measured_gyro_rate[:training_end]
    validation_x = wheel_difference_rate[split_index:]
    validation_y = measured_gyro_rate[split_index:]

    if training_x.size < 5 or validation_x.size < 4:
        raise ValueError(
            "More sensor readings are needed for training and held-out validation."
        )
    if np.std(training_x) < 1e-6:
        raise ValueError(
            "The training drive has too little turning motion to estimate "
            "wheelbase. Include left and right turns."
        )

    design_matrix = np.column_stack((training_x, np.ones_like(training_x)))
    initial_estimate, *_ = np.linalg.lstsq(design_matrix, training_y, rcond=None)
    initial_residual = training_y - design_matrix @ initial_estimate
    robust_scale = max(
        float(
            1.4826
            * np.median(
                np.abs(initial_residual - np.median(initial_residual))
            )
        ),
        1e-4,
    )
    fit = least_squares(
        lambda parameters: (
            parameters[0] * training_x + parameters[1] - training_y
        ),
        initial_estimate,
        loss="soft_l1",
        f_scale=robust_scale,
    )
    scale, gyro_bias = (float(value) for value in fit.x)
    if not fit.success or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "Could not infer a positive wheelbase from these sensor logs. "
            "Check wheel direction, wheel radius, and turning maneuvers."
        )

    track_width_m = wheel_radius_m / scale
    if not np.isfinite(track_width_m) or track_width_m <= 0.0:
        raise ValueError("Estimated track width is not finite and positive.")

    training_prediction = scale * training_x + gyro_bias
    validation_prediction = scale * validation_x + gyro_bias
    return DriveModel(
        wheel_radius_m=float(wheel_radius_m),
        track_width_m=track_width_m,
        gyro_bias_rad_s=gyro_bias,
        training_gyro_rmse_rad_s=float(
            np.sqrt(np.mean((training_prediction - training_y) ** 2))
        ),
        validation_gyro_rmse_rad_s=float(
            np.sqrt(np.mean((validation_prediction - validation_y) ** 2))
        ),
        training_intervals=int(training_x.size),
        validation_intervals=int(validation_x.size),
    )


def simulate_drive(drive_log: DriveLog, model: DriveModel) -> DriveSimulation:
    """Reconstruct the robot's planar path by integrating differential odometry."""
    elapsed = np.diff(drive_log.time_s)
    left_distance = model.wheel_radius_m * np.diff(drive_log.left_encoder_rad)
    right_distance = model.wheel_radius_m * np.diff(drive_log.right_encoder_rad)
    forward_distance = 0.5 * (left_distance + right_distance)
    heading_change = (right_distance - left_distance) / model.track_width_m

    heading_midpoint = np.cumsum(heading_change) - 0.5 * heading_change
    x_m = np.concatenate(
        ([0.0], np.cumsum(forward_distance * np.cos(heading_midpoint)))
    )
    y_m = np.concatenate(
        ([0.0], np.cumsum(forward_distance * np.sin(heading_midpoint)))
    )
    yaw_rad = np.concatenate(([0.0], np.cumsum(heading_change)))
    measured_yaw_rad = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5
                * (
                    drive_log.gyro_z_rad_s[:-1]
                    + drive_log.gyro_z_rad_s[1:]
                    - 2.0 * model.gyro_bias_rad_s
                )
                * elapsed
            ),
        )
    )
    predicted_gyro_z_rad_s = (
        model.radius_over_track_width
        * (
            np.diff(drive_log.right_encoder_rad)
            - np.diff(drive_log.left_encoder_rad)
        )
        / elapsed
        + model.gyro_bias_rad_s
    )
    return DriveSimulation(
        time_s=drive_log.time_s.copy(),
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        predicted_gyro_z_rad_s=predicted_gyro_z_rad_s,
        measured_yaw_rad=measured_yaw_rad,
    )
