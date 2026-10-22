import pandas as pd
import pytest

from racecraft.data.snapshot import profiles_from_snapshot, validate_snapshot


def snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "driver_id": ["A", "B"],
            "position": [1, 2],
            "gap_to_leader_seconds": [0.0, 3.2],
            "clean_lap_seconds": [90.0, 90.2],
            "tyre_age": [8, 12],
            "compound": ["HARD", "MEDIUM"],
        }
    )


def test_snapshot_builds_profiles_with_real_gaps() -> None:
    profiles = profiles_from_snapshot(snapshot())
    assert profiles[1].elapsed_offset_seconds == 3.2


def test_snapshot_requires_every_field_column() -> None:
    with pytest.raises(ValueError, match="gap_to_leader_seconds"):
        validate_snapshot(snapshot().drop(columns="gap_to_leader_seconds"))
