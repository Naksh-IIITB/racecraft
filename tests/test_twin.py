from racecraft.simulation.twin import (
    DriverProfile,
    EventHazard,
    PitPolicy,
    simulate_twin,
)


def test_twin_returns_one_classification_row_per_car() -> None:
    profiles = [
        DriverProfile("A", 90.0, 1),
        DriverProfile("B", 90.2, 2),
        DriverProfile("C", 90.4, 3),
    ]
    result = simulate_twin(
        profiles,
        {"A": PitPolicy(10), "B": PitPolicy(11), "C": PitPolicy(12)},
        total_laps=20,
        seed=2,
    )
    assert len(result.classification) == 3
    assert result.classification.position.tolist() == [1, 2, 3]
    assert len(result.laps) == 60


def test_target_policy_changes_its_pit_log() -> None:
    profiles = [DriverProfile("A", 90.0, 1), DriverProfile("B", 90.2, 2)]
    result = simulate_twin(
        profiles,
        {"A": PitPolicy(15), "B": PitPolicy(15)},
        total_laps=20,
        target_id="A",
        target_strategy=PitPolicy(5),
        hazard=EventHazard(0),
        seed=7,
    )
    assert result.pit_log.loc[result.pit_log.driver_id == "A", "lap"].iloc[0] == 5
