# Model cards

## PaceModel

**Purpose:** estimate the next clean lap time conditional on the state known after the prior lap.

**Baseline:** CPU-friendly histogram gradient boosting quantile regression, with tyre age, race progress, and rolling pre-decision lap median.

**Metrics:** rolling-season MAE/RMSE, per-circuit error, and interval coverage after training p10/p50/p90 variants.

**Limitations:** fuel mass, setup, component state, wind, red flags, and unobserved traffic are not fully identified from public timing data.

## SafetyCarModel

**Purpose:** calibrated probability of a VSC/safety-car-related neutralisation in the next five laps.

**Baseline:** calibrated histogram-gradient-boosted classifier. Report Brier score and calibration curve; do not report raw accuracy as the principal metric for rare events.

**Limitations:** public race-control labels are sparse, track-specific samples are limited, and causality is not implied.
