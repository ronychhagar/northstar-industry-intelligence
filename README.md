# Sensor Physics Studio

**A physics-modeling product prototype: connect real sensors to interpretable
models, inspect a simulation, and validate predictions on data the model never
saw.**

The first vertical slice, **MotionLab**, calibrates and visualizes the
differential-drive motion of a wheeled robot from wheel encoders and an
inertial measurement unit (IMU). A second, independent example in this
repository demonstrates a physics-informed neural network (PINN) for heat
diffusion.

## Try MotionLab

Python 3.10 or later:

```powershell
python -m pip install -e ".[app,test]"
python -m streamlit run robotics_twin/app.py
```

Open the local URL printed by Streamlit. MotionLab starts immediately with a
reproducible, synthetic 30-second robot drive. You can also explore the
bundled [`example_robot_drive.csv`](./examples/example_robot_drive.csv) or
click **Download example sensor log** to obtain it from the app. Change the
wheel radius or training split to explore the calibration. Training uses the
earlier data; held-out yaw-rate prediction error uses the later data.

### CSV sensor format

Provide one synchronized observation per row with these required,
case-insensitive column names:

| Column | Unit | Meaning |
| --- | --- | --- |
| `time_s` | seconds | Strictly increasing sensor timestamp |
| `left_encoder_rad` | radians | Accumulated left wheel angle |
| `right_encoder_rad` | radians | Accumulated right wheel angle |
| `gyro_z_rad_s` | radians/second | IMU angular rate around the vertical axis |

Both encoders must increase when their wheels move the robot forward. Sensor
timestamps must be synchronized and strictly increasing; at least 12 samples
are required. The uploaded log remains in the local Streamlit session.

### What the robot model estimates

Differential-drive kinematics predict yaw rate as
\[
\omega_z =
\frac{r}{b}(\dot{\phi}_{right}-\dot{\phi}_{left}) + \beta,
\]
where \(r\) is wheel radius, \(b\) is effective track width, and \(\beta\) is
constant gyroscope bias. MotionLab uses the supplied wheel radius, robustly
fits \(r/b\) and \(\beta\) on the training portion, then reports prediction
RMSE for gyroscope readings in the held-out portion. It integrates wheel
encoder motion into an estimated planar trajectory and compares encoder
heading with bias-corrected IMU heading.

**Identifiability matters:** yaw-rate measurements determine the ratio
\(r/b\), not wheel radius and track width independently. The user supplies a
measured wheel radius; the app estimates effective track width. A log with
straight-line motion alone cannot identify track width: include both left and
right turns. Different sensors, another measured distance, or a known
calibration maneuver can constrain other parameters.

The reconstructed path is dead-reckoned odometry, not independently measured
ground truth. This first prototype assumes planar motion, synchronized
sensors, constant wheel radius, and no wheel slip. It does not infer geometry
from camera images, calibrate wheel radius, estimate slip or acceleration, or
reconstruct arbitrary robot dynamics. These are deliberate, explicit scope
boundaries—not guarantees of unattended, general-purpose physics discovery.

## Heat-transfer PINN benchmark

The independent `thermal_pinn` example learns the one-dimensional heat
equation

\[
T_t = \alpha T_{xx},\qquad T(0,t)=T(1,t)=0,\qquad
T(x,0)=\sin(\pi x),
\]

with \(\alpha=0.1\). A neural network enforces the conditions exactly and
minimizes the governing-equation residual without reference-solver training
data. It is compared with a SciPy method-of-lines solution and an exact
analytical solution.

```powershell
python -m thermal_pinn
```

## Tests

```powershell
python -m pytest
```

The tests cover the analytical-versus-numerical heat solver, PINN conditions
and training, robot CSV import and validation, physics-based wheelbase
recovery, identifiability checks, and forward trajectory reconstruction.

## Roadmap: grow into a useful product

1. Collect sensor logs from a real mobile robot and verify units,
   synchronization, and encoder direction.
2. Compare the odometry trajectory with an independent reference, such as
   motion capture, visual odometry, or measured course landmarks.
3. Add additional sensor-model templates only when each can be calibrated and
   tested against held-out real measurements.
4. Explore camera-assisted geometry as a human-confirmed input, not a claim
   that a single image reveals hidden physical properties.
5. Keep estimated parameters, uncertainty, model assumptions, and validation
   results visible so users can tell when a model is not supported by its data.
