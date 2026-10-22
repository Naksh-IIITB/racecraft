"""Validation and conversion for an all-car race snapshot.

Finish forecasts require the field, not just the target car. This CSV contract
is deliberately small enough to fill from public timing data or a manual sheet.
"""

from __future__ import annotations

import pandas as pd

from racecraft.simulation.twin import DriverProfile, PitPolicy

REQUIRED_SNAPSHOT_COLUMNS = {
    "driver_id",
    "position",
    "gap_to_leader_seconds",
    "clean_lap_seconds",
    "tyre_age",
    "compound",
}
COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def validate_snapshot(snapshot: pd.DataFrame) -> None:
    """Validate the all-car state needed for an honest finish forecast."""
    missing = REQUIRED_SNAPSHOT_COLUMNS - set(snapshot.columns)
    if missing:
        raise ValueError(f"snapshot is missing columns: {', '.join(sorted(missing))}")
    if not 2 <= len(snapshot) <= 20:
        raise ValueError("snapshot must have between 2 and 20 cars")
    if snapshot.driver_id.isna().any() or snapshot.driver_id.duplicated().any():
        raise ValueError("driver_id must be present and unique")
    if snapshot.position.isna().any() or snapshot.position.duplicated().any():
        raise ValueError("position must be present and unique")
    if (snapshot.clean_lap_seconds <= 0).any() or (snapshot.tyre_age < 0).any():
        raise ValueError(
            "clean_lap_seconds must be positive and tyre_age cannot be negative"
        )
    if not set(snapshot.compound.astype(str).str.upper()) <= COMPOUNDS:
        raise ValueError("compound must be SOFT, MEDIUM, or HARD")


def profiles_from_snapshot(snapshot: pd.DataFrame) -> list[DriverProfile]:
    validate_snapshot(snapshot)
    return [
        DriverProfile(
            driver_id=str(row.driver_id),
            baseline_pace=float(row.clean_lap_seconds),
            grid_position=int(row.position),
            compound=str(row.compound).upper(),
            tyre_age=int(row.tyre_age),
            elapsed_offset_seconds=float(row.gap_to_leader_seconds),
        )
        for row in snapshot.itertuples(index=False)
    ]


def policies_from_snapshot(snapshot: pd.DataFrame) -> dict[str, PitPolicy]:
    """Use optional plans when supplied; blank plans mean no scheduled stop."""
    policies: dict[str, PitPolicy] = {}
    for row in snapshot.itertuples(index=False):
        plan = getattr(row, "planned_pit_lap", None)
        next_compound = str(getattr(row, "next_compound", "HARD")).upper()
        pit_lap = int(plan) if pd.notna(plan) else None
        policies[str(row.driver_id)] = PitPolicy(pit_lap, next_compound)
    return policies
