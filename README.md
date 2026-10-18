# Northstar Industry Intelligence

**A multi-industry research-as-a-service workspace.** Northstar helps teams in
automotive, robotics, semiconductors, and embedded systems turn strategic
questions into scoped research briefs, track engagements, and structure
market, competitor, technology, and customer research in one client-facing
desk. Research dimensions and signal watchlists adapt to the selected industry.
MotionLab remains available as a robotics engineering module.

## Start the workspace

Python 3.10 or later:

```powershell
python -m pip install -e ".[app,test]"
python -m streamlit run robotics_twin/app.py
```

Open the local URL printed by Streamlit. Select an **Industry workspace** in
the sidebar, then open **Research requests** to submit a decision-led brief.
Each request stores its industry, research focus, decision, geography,
competitors or technologies, deliverable, target date, and context. Track its
status, add research-team notes, download an individual brief as JSON, or
export the engagement list as CSV.

The SQLite database is `automotive_research.db` at the project root. Existing
databases are migrated automatically, with existing requests assigned to
Automotive. Keep this file backed up if it contains client information. This is
a local workspace; it does not include hosted accounts, multi-user access
controls, or shared cloud storage.

## Industry workspaces

- **Automotive:** EVs, charging, software-defined vehicles, ADAS, fleets,
  battery supply chains, and aftermarket.
- **Robotics:** industrial automation, collaborative robots, mobile robots,
  humanoids, deployment economics, safety, and systems integration.
- **Semiconductors:** AI accelerators, automotive chips, power devices,
  process and packaging technology, capacity, design wins, and supply risk.
- **Embedded systems:** compute platforms, RTOS and software, connectivity,
  edge AI, safety, security, product lifecycle, and component availability.

The shared **Market monitor**, **Competitor landscape**, and **Technology
radar** adapt their lenses, signal watchlists, comparison dimensions, and
assessment factors to the selected industry. Market figures and competitor
findings are intentionally not pre-populated. Add dated, attributable evidence
and methodology before using a finding in a business decision. Every estimate
should state its definition, geography, time period, source, and confidence.

## MotionLab sensor format

Upload a synchronized CSV with one observation per row and these required,
case-insensitive columns:

| Column | Unit | Meaning |
| --- | --- | --- |
| `time_s` | seconds | Strictly increasing timestamp |
| `left_encoder_rad` | radians | Accumulated left wheel angle |
| `right_encoder_rad` | radians | Accumulated right wheel angle |
| `gyro_z_rad_s` | radians/second | IMU angular rate around the vertical axis |

Both encoders must increase when their wheels move forward. At least 12 samples
are required. The uploaded log remains in the local Streamlit session.

The model uses the supplied wheel radius and fits effective track width and
constant gyroscope bias. The yaw-rate model is checked on later held-out
readings. Track width cannot be identified from straight-line motion alone;
include both left and right turns. The reconstructed path is dead-reckoned
odometry, not independently measured ground truth, and assumes planar motion,
synchronized sensors, and no wheel slip.

## Heat-transfer PINN benchmark

The independent `thermal_pinn` example learns the one-dimensional heat
equation and compares its physics-informed neural network with numerical and
analytical solutions:

```powershell
python -m thermal_pinn
```

## Tests

```powershell
python -m pytest
```
