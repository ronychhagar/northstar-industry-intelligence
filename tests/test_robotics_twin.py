from pathlib import Path

import numpy as np
import pytest

from robotics_twin.demo import make_demo_drive
from robotics_twin.odometry import (
    DriveLog,
    fit_drive_model,
    load_sensor_csv,
    simulate_drive,
)


def test_bundled_example_csv_matches_the_synthetic_demo() -> None:
    example_csv = Path(__file__).parent.parent / "examples" / "example_robot_drive.csv"

    assert example_csv.read_bytes() == make_demo_drive().csv_bytes


def test_demo_log_recovers_track_width_and_validates_unseen_turns() -> None:
    demo = make_demo_drive()
    model = fit_drive_model(demo.log, wheel_radius_m=demo.wheel_radius_m)

    assert model.track_width_m == pytest.approx(
        demo.true_track_width_m, abs=0.006
    )
    assert 0.0 < model.training_gyro_rmse_rad_s < 0.015
    assert 0.0 < model.validation_gyro_rmse_rad_s < 0.015
    assert model.validation_intervals > 100


def test_demo_csv_import_round_trip() -> None:
    demo = make_demo_drive()
    parsed = load_sensor_csv(demo.csv_bytes)

    np.testing.assert_allclose(parsed.time_s, demo.log.time_s)
    np.testing.assert_allclose(parsed.left_encoder_rad, demo.log.left_encoder_rad)
    np.testing.assert_allclose(
        parsed.right_encoder_rad, demo.log.right_encoder_rad
    )
    np.testing.assert_allclose(parsed.gyro_z_rad_s, demo.log.gyro_z_rad_s)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "header row"),
        ("time_s,gyro_z_rad_s\n0,0\n", "missing required columns"),
        (
            "time_s,left_encoder_rad,right_encoder_rad,gyro_z_rad_s\n"
            "0,0,0,0\n"
            "0.1,1,,0.1\n",
            "Row 3",
        ),
    ],
)
def test_sensor_csv_rejects_invalid_inputs(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_sensor_csv(source)


@pytest.mark.parametrize(
    ("time_s", "message"),
    [
        (np.array([0.0, 0.0] + list(np.arange(0.2, 1.2, 0.1))), "strictly increasing"),
        (np.arange(11, dtype=float), "at least 12"),
    ],
)
def test_sensor_log_rejects_invalid_timestamps_and_lengths(
    time_s: np.ndarray, message: str
) -> None:
    count = time_s.size
    readings = np.linspace(0.0, 1.0, count)
    with pytest.raises(ValueError, match=message):
        DriveLog(
            time_s=time_s,
            left_encoder_rad=readings,
            right_encoder_rad=readings + 0.1,
            gyro_z_rad_s=readings * 0.0,
        )


@pytest.mark.parametrize(
    ("wheel_radius_m", "train_fraction", "message"),
    [
        (0.0, 0.7, "wheel_radius_m"),
        (0.05, 0.2, "train_fraction"),
    ],
)
def test_model_fit_rejects_invalid_options(
    wheel_radius_m: float,
    train_fraction: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_drive_model(
            make_demo_drive().log,
            wheel_radius_m=wheel_radius_m,
            train_fraction=train_fraction,
        )


def test_insufficient_turning_motion_cannot_identify_track_width() -> None:
    time_s = np.linspace(0.0, 2.0, 51)
    same_wheel_motion = 4.0 * time_s
    straight_drive = DriveLog(
        time_s=time_s,
        left_encoder_rad=same_wheel_motion,
        right_encoder_rad=same_wheel_motion,
        gyro_z_rad_s=np.zeros_like(time_s),
    )

    with pytest.raises(ValueError, match="turning motion"):
        fit_drive_model(straight_drive, wheel_radius_m=0.05)


def test_simulation_starts_at_origin_and_returns_one_pose_per_sample() -> None:
    demo = make_demo_drive()
    model = fit_drive_model(demo.log, wheel_radius_m=demo.wheel_radius_m)
    simulation = simulate_drive(demo.log, model)

    assert simulation.time_s.shape == demo.log.time_s.shape
    assert simulation.x_m.shape == demo.log.time_s.shape
    assert simulation.y_m.shape == demo.log.time_s.shape
    assert simulation.yaw_rad.shape == demo.log.time_s.shape
    assert np.isfinite(simulation.x_m).all()
    assert np.isfinite(simulation.y_m).all()
    assert simulation.x_m[0] == 0.0
    assert simulation.y_m[0] == 0.0
