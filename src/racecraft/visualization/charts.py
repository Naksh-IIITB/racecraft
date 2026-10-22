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
        color_continuous_scale=["#ff4d61", "#ffbe3d", "#53dfc6"],
        title="STRATEGY BOARD · time gained versus staying out",
        hover_data=["Beat stay-out probability"],
    ).update_layout(
        height=max(340, 56 * len(frame)),
        coloraxis_colorbar_title="Win chance",
        xaxis_title="Seconds gained (+) or lost (−)",
        yaxis_title="",
        margin={"l": 12, "r": 22, "t": 65, "b": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,29,.68)",
        font={"color": "#e7edf7", "family": "Inter, Arial, sans-serif"},
        title_font={"size": 16, "color": "#65e5d1"},
        coloraxis_colorbar={"tickformat": ".0%"},
        xaxis={"gridcolor": "rgba(148,163,184,.18)", "zerolinecolor": "#ffbe3d"},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
