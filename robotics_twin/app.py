"""Browser-based sensor calibration and simulation product prototype."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from robotics_twin.demo import make_demo_drive
from robotics_twin.odometry import (
    DriveLog,
    DriveSimulation,
    fit_drive_model,
    load_sensor_csv,
    simulate_drive,
)


def _plot_results(
    drive_log: DriveLog,
    simulation: DriveSimulation,
    train_fraction: float,
) -> None:
    split_index = int(drive_log.time_s.size * train_fraction)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(simulation.x_m, simulation.y_m, color="#2878b5", linewidth=2)
    axes[0].scatter([0.0], [0.0], color="#24874a", s=40, label="Start")
    axes[0].scatter(
        [simulation.x_m[-1]],
        [simulation.y_m[-1]],
        color="#cf493d",
        s=40,
        label="Finish",
    )
    axes[0].set(
        title="Encoder-reconstructed path",
        xlabel="Distance east (m)",
        ylabel="Distance north (m)",
        aspect="equal",
    )
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        drive_log.time_s,
        simulation.yaw_rad,
        label="Wheel odometry",
        color="#2878b5",
    )
    axes[1].plot(
        drive_log.time_s,
        simulation.measured_yaw_rad,
        label="Bias-corrected IMU",
        color="#d58c32",
        alpha=0.8,
    )
    axes[1].axvline(
        drive_log.time_s[split_index],
        color="#666666",
        linestyle="--",
        label="Validation starts",
    )
    axes[1].set(title="Heading cross-check", xlabel="Time (s)", ylabel="Yaw (rad)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        drive_log.time_s[1:],
        drive_log.gyro_z_rad_s[1:],
        label="Measured IMU",
        linewidth=1.1,
        color="#d58c32",
    )
    axes[2].plot(
        drive_log.time_s[1:],
        simulation.predicted_gyro_z_rad_s,
        label="Wheel-model prediction",
        linewidth=1.3,
        color="#2878b5",
    )
    axes[2].axvline(
        drive_log.time_s[split_index],
        color="#666666",
        linestyle="--",
        label="Validation starts",
    )
    axes[2].set(
        title="Gyroscope: measured vs. predicted",
        xlabel="Time (s)",
        ylabel="Yaw rate (rad/s)",
    )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def main() -> None:
    st.set_page_config(
        page_title="MotionLab | Robot digital twin",
        page_icon="🧭",
        layout="wide",
    )
    st.title("MotionLab")
    st.caption("From robot sensor logs to an explainable motion model.")
    st.markdown(
        "Upload wheel encoders and a yaw gyroscope. MotionLab estimates your "
        "robot's **effective track width**, checks predictions on unseen "
        "readings, and reconstructs its path."
    )

    with st.sidebar:
        st.header("Robot setup")
        wheel_radius = st.number_input(
            "Measured wheel radius (m)",
            min_value=0.005,
            max_value=1.0,
            value=0.05,
            step=0.005,
            format="%.3f",
            help="Wheel radius is measured or specified; the gyroscope and "
            "encoders together cannot determine wheel radius and track width "
            "independently without another distance reference.",
        )
        train_fraction = st.slider(
            "Training data (%)",
            min_value=50,
            max_value=90,
            value=70,
            step=5,
            help="Earlier readings estimate the model. Later readings provide "
            "a separate, held-out validation.",
        )
        st.session_state.train_fraction = train_fraction / 100.0

        demo = make_demo_drive()
        st.download_button(
            "Download example sensor log",
            demo.csv_bytes,
            file_name="example_robot_drive.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Example log is synthetic; it is not real robot telemetry.")

    upload = st.file_uploader(
        "Robot sensor log (CSV)",
        type=("csv",),
        help=(
            "Required columns: time_s, left_encoder_rad, "
            "right_encoder_rad, gyro_z_rad_s. Both encoders must increase "
            "when the robot moves forward."
        ),
    )
    if upload is None:
        st.info(
            "Showing a 30-second example drive with forward, left-turn, and "
            "right-turn maneuvers. Upload your own log to calibrate a robot."
        )
        drive_log = demo.log
        if not np.isclose(wheel_radius, demo.wheel_radius_m):
            st.info(
                "The example was simulated with 0.05 m wheels. Change the "
                "wheel radius above to 0.05 m to reproduce the example "
                "calibration."
            )
    else:
        try:
            drive_log = load_sensor_csv(upload.getvalue())
        except ValueError as error:
            st.error(str(error))
            st.stop()
        st.success(f"Loaded {drive_log.time_s.size:,} sensor samples.")

    try:
        model = fit_drive_model(
            drive_log,
            wheel_radius_m=wheel_radius,
            train_fraction=st.session_state.train_fraction,
        )
        simulation = simulate_drive(drive_log, model)
    except ValueError as error:
        st.error(f"Could not calibrate this drive: {error}")
        st.info(
            "Use synchronized encoder/IMU logs with clear left and right turns, "
            "a known wheel radius, and both encoder counts increasing on "
            "forward motion."
        )
        st.stop()

    st.subheader("Calibrated model")
    width_col, bias_col = st.columns(2)
    width_col.metric("Estimated track width", f"{model.track_width_m:.3f} m")
    bias_col.metric("Estimated gyro bias", f"{model.gyro_bias_rad_s:.4f} rad/s")
    train_col, validation_col = st.columns(2)
    train_col.metric(
        "Training yaw-rate RMSE",
        f"{model.training_gyro_rmse_rad_s:.4f} rad/s",
    )
    validation_col.metric(
        "Unseen-data yaw-rate RMSE",
        f"{model.validation_gyro_rmse_rad_s:.4f} rad/s",
        help=f"Computed on {model.validation_intervals} held-out intervals, "
        "not used to fit the model.",
    )

    _plot_results(drive_log, simulation, st.session_state.train_fraction)

    with st.expander("Sensor data preview"):
        preview = pd.DataFrame(
            {
                "time_s": drive_log.time_s,
                "left_encoder_rad": drive_log.left_encoder_rad,
                "right_encoder_rad": drive_log.right_encoder_rad,
                "gyro_z_rad_s": drive_log.gyro_z_rad_s,
            }
        )
        st.dataframe(preview.head(12), use_container_width=True, hide_index=True)

    st.caption(
        "Model: differential-drive wheel odometry. Wheel radius is supplied; "
        "effective track width and constant gyro bias are fitted to training "
        "data. Assumes synchronized sensors, planar motion, and no wheel slip. "
        "The plotted path is an odometry reconstruction—not independently "
        "measured ground truth. This kinematics prototype does not estimate "
        "wheel radius, slip, acceleration, or full rigid-body dynamics."
    )


if __name__ == "__main__":
    main()
