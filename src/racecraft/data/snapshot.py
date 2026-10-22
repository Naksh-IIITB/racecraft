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
    if (
        snapshot.gap_to_leader_seconds.isna().any()
        or (snapshot.gap_to_leader_seconds < 0).any()
    ):
        raise ValueError(
            "gap_to_leader_seconds must be a non-negative number for every car"
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


def policies_from_snapshot(
    snapshot: pd.DataFrame, *, current_lap: int, total_laps: int
) -> dict[str, PitPolicy]:
    """Build visible rival pit policies, inferring only a tyre-life baseline.

    A provided `planned_pit_lap` always wins. For an otherwise blank snapshot,
    the conservative fallback schedules one stop only if the current stint is
    unlikely to reach the finish according to compound-specific tyre life.
    """
    max_stint = {"SOFT": 18, "MEDIUM": 27, "HARD": 36}
    policies: dict[str, PitPolicy] = {}
    for row in snapshot.itertuples(index=False):
        plan = getattr(row, "planned_pit_lap", None)
        compound = str(row.compound).upper()
        remaining_laps = total_laps - current_lap
        next_compound = str(getattr(row, "next_compound", "HARD")).upper()
        if pd.notna(plan):
            pit_lap = int(plan)
        elif int(row.tyre_age) + remaining_laps > max_stint[compound]:
            pit_lap = min(
                total_laps - 1,
                max(current_lap + 1, current_lap + max_stint[compound] - int(row.tyre_age)),
            )
            # A very short final stint is a plausible soft-tyre attack; otherwise
            # hard is the deliberately conservative default.
            if total_laps - pit_lap <= 12:
                next_compound = "SOFT"
        else:
            pit_lap = None
        policies[str(row.driver_id)] = PitPolicy(pit_lap, next_compound)
    return policies
