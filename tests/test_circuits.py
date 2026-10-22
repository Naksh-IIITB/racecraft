from racecraft.circuits import CIRCUITS


def test_announced_2026_calendar_has_24_circuit_presets() -> None:
    assert len(CIRCUITS) == 24
    assert "Monaco" in " ".join(CIRCUITS)
    assert all(c.total_laps > 10 and c.pit_lane_loss > 0 for c in CIRCUITS.values())
