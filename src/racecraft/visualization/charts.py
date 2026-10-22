from __future__ import annotations

import pandas as pd
import plotly.express as px


def strategy_chart(results: list) -> object:
    rows = [
        {
            "Strategy": r.strategy.label,
            "Delta vs stay out (s)": r.delta_to_baseline_seconds,
            "Direction": "Faster" if r.delta_to_baseline_seconds >= 0 else "Slower",
        }
        for r in results
    ]
    frame = pd.DataFrame(rows).sort_values("Delta vs stay out (s)", ascending=False)
    return px.bar(
        frame,
        y="Strategy",
        x="Delta vs stay out (s)",
        color="Direction",
        color_discrete_map={"Faster": "#53dfc6", "Slower": "#ff4d61"},
        title="STRATEGY BOARD · time gained versus staying out",
        hover_data={"Direction": False},
    ).update_layout(
        height=max(340, 56 * len(frame)),
        xaxis_title="Seconds gained (+) or lost (−)",
        yaxis_title="",
        margin={"l": 12, "r": 22, "t": 65, "b": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,29,.68)",
        font={"color": "#e7edf7", "family": "Inter, Arial, sans-serif"},
        title_font={"size": 16, "color": "#65e5d1"},
        xaxis={"gridcolor": "rgba(148,163,184,.18)", "zerolinecolor": "#ffbe3d"},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
    )


def position_progression_chart(
    progression: pd.DataFrame, drivers: list[str], target_driver: str
) -> object:
    """Show expected race order over time for a target and nearby rivals."""
    frame = progression.loc[progression.driver_id.isin(drivers)].copy()
    return px.line(
        frame,
        x="lap",
        y="expected_position",
        color="driver_id",
        markers=True,
        title="PROJECTED RACE ORDER · target versus nearby rivals",
        labels={
            "lap": "Race lap",
            "expected_position": "Expected position",
            "driver_id": "Car",
        },
        category_orders={"driver_id": [target_driver, *[d for d in drivers if d != target_driver]]},
        color_discrete_sequence=["#ff4655", "#65e5d1", "#ffbe3d", "#a78bfa", "#60a5fa"],
    ).update_layout(
        height=390,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12,18,29,.68)",
        font={"color": "#e7edf7", "family": "Inter, Arial, sans-serif"},
        title_font={"size": 16, "color": "#65e5d1"},
        legend_title_text="",
        margin={"l": 12, "r": 22, "t": 65, "b": 50},
        xaxis={"gridcolor": "rgba(148,163,184,.18)"},
        yaxis={"autorange": "reversed", "dtick": 1, "gridcolor": "rgba(148,163,184,.18)"},
    )
