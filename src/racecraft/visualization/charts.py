from __future__ import annotations

import pandas as pd
import plotly.express as px


def strategy_chart(results: list) -> object:
    rows = [{"Strategy": r.strategy.label, "Expected finish": r.expected_finish,
             "Points probability": r.p_points, "Best-decile finish": r.p10_finish,
             "Worst-decile finish": r.p90_finish} for r in results]
    frame = pd.DataFrame(rows).sort_values("Expected finish")
    return px.bar(frame, x="Strategy", y="Expected finish", error_y=frame["Worst-decile finish"] - frame["Expected finish"],
                  error_y_minus=frame["Expected finish"] - frame["Best-decile finish"], color="Points probability",
                  color_continuous_scale="Tealgrn", title="Strategy comparison (lower expected finish is better)")
