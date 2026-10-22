from __future__ import annotations

import pandas as pd
import plotly.express as px


def strategy_chart(results: list) -> object:
    rows = [
        {
            "Strategy": r.strategy.label,
            "Expected finish": r.expected_finish,
            "Points probability": r.p_points,
            "Delta vs stay out (s)": r.delta_to_baseline_seconds,
            "Beat stay-out probability": r.p_beat_baseline,
            "Best-decile finish": r.p10_finish,
            "Worst-decile finish": r.p90_finish,
        }
        for r in results
    ]
    frame = pd.DataFrame(rows).sort_values("Delta vs stay out (s)", ascending=False)
    return px.bar(
        frame,
        x="Strategy",
        y="Delta vs stay out (s)",
        color="Beat stay-out probability",
        color_continuous_scale="Tealgrn",
        title="Decision value vs staying out (positive means faster)",
        hover_data=["Expected finish", "Points probability"],
    )
