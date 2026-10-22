"""Calibrated probability model for a neutralisation within a fixed horizon."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_FEATURES = ("race_progress", "lap_number", "prior_neutralisations")


@dataclass
class SafetyCarModel:
    features: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        self.model = CalibratedClassifierCV(
            HistGradientBoostingClassifier(
                max_iter=150, max_leaf_nodes=12, l2_regularization=2.0
            ),
            method="sigmoid",
            cv=3,
        )

    def fit(self, frame: pd.DataFrame) -> SafetyCarModel:
        self._validate(frame, "neutralisation_next_5")
        self.model.fit(
            frame.loc[:, self.features], frame.neutralisation_next_5.astype(int)
        )
        return self

    def predict_probability(self, frame: pd.DataFrame) -> pd.Series:
        self._validate(frame)
        return pd.Series(
            self.model.predict_proba(frame.loc[:, self.features])[:, 1],
            index=frame.index,
            name="neutralisation_p_next_5",
        )

    def _validate(self, frame: pd.DataFrame, label: str | None = None) -> None:
        missing = set(self.features) | ({label} if label else set())
        missing -= set(frame.columns)
        if missing:
            raise ValueError(f"missing model columns: {sorted(missing)}")
