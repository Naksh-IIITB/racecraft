"""CPU-friendly quantile pace model with explicit train/predict contract."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DEFAULT_FEATURES = ("tyre_age", "race_progress", "clean_lap_median")


@dataclass
class PaceModel:
    quantile: float = .5
    features: tuple[str, ...] = DEFAULT_FEATURES

    def __post_init__(self) -> None:
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=self.quantile, max_iter=250,
                                                   learning_rate=.06, max_leaf_nodes=20, l2_regularization=1.0)

    def fit(self, frame: pd.DataFrame) -> "PaceModel":
        self._validate(frame, label="next_lap_duration")
        self.model.fit(frame.loc[:, self.features], frame.next_lap_duration)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        self._validate(frame)
        return pd.Series(self.model.predict(frame.loc[:, self.features]), index=frame.index, name=f"pace_p{int(self.quantile * 100)}")

    def _validate(self, frame: pd.DataFrame, label: str | None = None) -> None:
        needed = set(self.features) | ({label} if label else set())
        missing = needed - set(frame.columns)
        if missing:
            raise ValueError(f"missing model columns: {sorted(missing)}")
