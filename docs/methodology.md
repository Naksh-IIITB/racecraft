# Methodology

RaceCraft represents a strategy recommendation as a conditional simulation: *under declared pace, pit-loss, traffic, and safety-car assumptions*, which candidate has the best risk-adjusted outcome? It does not assert that the resulting counterfactual occurred or was objectively optimal.

The current baseline is a transparent simulator with a compact all-car digital twin. The next learned components are: (1) quantile gradient-boosted clean-air lap-time predictions, (2) a calibrated safety-car/VSC survival model, (3) an empirical pit-loss model, and (4) an opponent pit-window classifier. Every feature must be computed from pre-decision information only. Evaluation must use rolling season splits and report calibration alongside point error.

The twin makes no assertion that a simulated alternate finish is a factual result. Its purpose is to compare policies under identical explicit assumptions, then measure whether its probability distributions are calibrated on held-out races.

## Data handling

Raw downloads remain in `data/raw/` and are excluded from version control. Derived training data must retain source, retrieval time, session key, and transformation version. The repository includes no redistributed race data.
