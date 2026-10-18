"""Reproducible encoder and IMU data for the product demo."""

import csv
import io
from dataclasses import dataclass

import numpy as np

from robotics_twin.odometry import DriveLog


@dataclass(frozen=True)
class DemoDrive:
    log: DriveLog
    csv_bytes: bytes
    wheel_radius_m: float
    true_track_width_m: float


def make_demo_drive() -> DemoDrive:
    """Create a noisy, maneuver-rich, simulated wheel-encoder and IMU log."""
    time_s = np.linspace(0.0, 30.0, 601)
    wheel_radius_m = 0.05
    true_track_width_m = 0.32
    gyro_bias_rad_s = 0.012
    generator = np.random.default_rng(7)

    forward_speed = (
        4.0
        + 0.7 * np.sin(2.0 * np.pi * time_s / 11.0)
        + 0.35 * np.sin(2.0 * np.pi * time_s / 4.1)
    )
    differential_speed = (
        0.8 * np.sin(2.0 * np.pi * time_s / 6.0)
        + 0.25 * np.sin(2.0 * np.pi * time_s / 2.7)
        + 0.12
    )
    left_speed = forward_speed - differential_speed / 2.0
    right_speed = forward_speed + differential_speed / 2.0
    left_angle = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5 * (left_speed[1:] + left_speed[:-1]) * np.diff(time_s)
            ),
        )
    )
    right_angle = np.concatenate(
        (
            [0.0],
            np.cumsum(
                0.5 * (right_speed[1:] + right_speed[:-1]) * np.diff(time_s)
            ),
        )
    )
    gyro_rate = (
        wheel_radius_m
        * differential_speed
        / true_track_width_m
        + gyro_bias_rad_s
        + generator.normal(0.0, 0.006, size=time_s.size)
    )
    drive_log = DriveLog(
        time_s=time_s,
        left_encoder_rad=left_angle,
        right_encoder_rad=right_angle,
        gyro_z_rad_s=gyro_rate,
    )

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ("time_s", "left_encoder_rad", "right_encoder_rad", "gyro_z_rad_s")
    )
    writer.writerows(
        zip(time_s, left_angle, right_angle, gyro_rate, strict=True)
    )
    return DemoDrive(
        log=drive_log,
        csv_bytes=output.getvalue().encode("utf-8"),
        wheel_radius_m=wheel_radius_m,
        true_track_width_m=true_track_width_m,
    )
