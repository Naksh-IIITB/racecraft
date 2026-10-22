from __future__ import annotations

import pandas as pd
import plotly.express as px


def strategy_chart(results: list) -> object:
    rows = [
        {
            "Strategy": r.strategy.label,
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
        y="Strategy",
        x="Delta vs stay out (s)",
        color="Beat stay-out probability",
        color_continuous_scale="Tealgrn",
        title="Best options — time gained versus staying out",
        hover_data=["Beat stay-out probability"],
    ).update_layout(
        height=max(320, 48 * len(frame)),
        coloraxis_colorbar_title="Chance\nto beat\nstay-out",
        xaxis_title="Seconds gained (+) or lost (−)",
        yaxis_title="",
        margin={"l": 12, "r": 12, "t": 55, "b": 50},
    )
