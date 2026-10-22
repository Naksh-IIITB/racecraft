"""Time-ordered evaluation; intentionally never uses random row splits."""
from __future__ import annotations

from typing import Callable

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss


def rolling_backtest(frame: pd.DataFrame, model_factory: Callable[[], object], *, season_column: str = "year",
                     target: str = "next_lap_duration") -> pd.DataFrame:
    """Train on preceding seasons and evaluate the next one.

    Models must provide `fit(frame)` and either `predict(frame)` for pace or
    `predict_probability(frame)` for probability targets.
    """
    if season_column not in frame:
        raise ValueError(f"{season_column} is required for time-ordered validation")
    results = []
    seasons = sorted(frame[season_column].dropna().unique())
    for test_season in seasons[1:]:
        train, test = frame[frame[season_column] < test_season], frame[frame[season_column] == test_season]
        if train.empty or test.empty:
            continue
        model = model_factory().fit(train)
        if target == "neutralisation_next_5":
            prediction = model.predict_probability(test)
            results.append({"test_season": int(test_season), "n": len(test), "brier": brier_score_loss(test[target], prediction)})
        else:
            prediction = model.predict(test)
            results.append({"test_season": int(test_season), "n": len(test), "mae": mean_absolute_error(test[target], prediction),
                            "rmse": mean_squared_error(test[target], prediction, squared=False)})
    return pd.DataFrame(results)
