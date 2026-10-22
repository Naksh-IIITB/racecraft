"""Leak-safe conversion of source tables into per-driver-lap decision records."""

from __future__ import annotations

import pandas as pd

REQUIRED_LAP_COLUMNS = {"session_key", "driver_number", "lap_number", "lap_duration"}


def validate_laps(laps: pd.DataFrame) -> None:
    missing = REQUIRED_LAP_COLUMNS - set(laps.columns)
    if missing:
        raise ValueError(f"laps is missing required columns: {sorted(missing)}")
    if laps.duplicated(["session_key", "driver_number", "lap_number"]).any():
        raise ValueError("laps must be unique per session, driver, and lap")


def _attach_stints(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    if stints.empty:
        return laps.assign(compound="UNKNOWN", tyre_age=0)
    required = {
        "session_key",
        "driver_number",
        "lap_start",
        "lap_end",
        "compound",
        "tyre_age_at_start",
    }
    if missing := required - set(stints.columns):
        raise ValueError(f"stints is missing required columns: {sorted(missing)}")
    pieces = []
    for (_, driver), driver_laps in laps.groupby(
        ["session_key", "driver_number"], sort=False
    ):
        session, number = driver_laps.iloc[0][["session_key", "driver_number"]]
        stint_rows = stints[
            (stints.session_key == session) & (stints.driver_number == number)
        ]
        frame = driver_laps.copy()
        frame["compound"], frame["tyre_age"] = "UNKNOWN", 0
        for stint in stint_rows.itertuples():
            mask = frame.lap_number.between(stint.lap_start, stint.lap_end)
            frame.loc[mask, "compound"] = stint.compound
            frame.loc[mask, "tyre_age"] = (
                frame.loc[mask, "lap_number"]
                - stint.lap_start
                + stint.tyre_age_at_start
            )
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def build_decision_dataset(
    laps: pd.DataFrame, stints: pd.DataFrame, race_control: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Produce only features available before the next lap begins.

    `next_lap_duration` and `neutralisation_next_5` are labels, never features.
    """
    validate_laps(laps)
    frame = laps.copy().sort_values(["session_key", "driver_number", "lap_number"])
    frame = frame[frame.lap_duration.notna() & (frame.lap_duration > 0)]
    frame = _attach_stints(frame, stints)
    group = frame.groupby(["session_key", "driver_number"], sort=False)
    frame["clean_lap_median"] = group.lap_duration.transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).median()
    )
    frame["next_lap_duration"] = group.lap_duration.shift(-1)
    max_lap = frame.groupby("session_key").lap_number.transform("max")
    frame["race_progress"] = frame.lap_number / max_lap
    frame["prior_neutralisations"] = 0
    frame["neutralisation_next_5"] = 0
    if (
        race_control is not None
        and not race_control.empty
        and {"lap_number", "category", "session_key"}.issubset(race_control.columns)
    ):
        neutral = race_control[race_control.category.isin(["SafetyCar", "Flag"])]
        for session, events in neutral.groupby("session_key"):
            event_laps = events.lap_number.dropna().astype(int).tolist()
            mask = frame.session_key == session
            frame.loc[mask, "prior_neutralisations"] = frame.loc[
                mask, "lap_number"
            ].map(
                lambda x, events_before=event_laps: sum(
                    event < x for event in events_before
                )
            )
            frame.loc[mask, "neutralisation_next_5"] = frame.loc[
                mask, "lap_number"
            ].map(
                lambda x, future_events=event_laps: int(
                    any(x < event <= x + 5 for event in future_events)
                )
            )
    # Exclude pit-out laps where raw lap time is not a clean pace target.
    if "is_pit_out_lap" in frame:
        frame = frame[~frame.is_pit_out_lap.fillna(False)]
    return frame.dropna(subset=["clean_lap_median", "next_lap_duration"]).reset_index(
        drop=True
    )
