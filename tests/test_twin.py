from racecraft.simulation.twin import (
    DriverProfile,
    EventHazard,
    PitPolicy,
    simulate_twin,
    simulate_twin_ensemble,
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


def test_current_gap_is_preserved_in_all_car_simulation() -> None:
    profiles = [
        DriverProfile("A", 90.0, 1),
        DriverProfile("B", 90.0, 2, elapsed_offset_seconds=12),
    ]
    result = simulate_twin(
        profiles,
        {"A": PitPolicy(None), "B": PitPolicy(None)},
        total_laps=2,
        seed=1,
    )
    assert result.classification.iloc[0].driver_id == "A"


def test_ensemble_returns_valid_full_field_position_probabilities() -> None:
    profiles = [
        DriverProfile("A", 90.0, 1),
        DriverProfile("B", 90.2, 2),
        DriverProfile("C", 90.4, 3),
    ]
    result = simulate_twin_ensemble(
        profiles,
        {"A": PitPolicy(8), "B": PitPolicy(9), "C": PitPolicy(None)},
        total_laps=15,
        start_lap=5,
        target_id="A",
        target_strategy=PitPolicy(8),
        hazard=EventHazard(0.0),
        n_simulations=10,
        seed=11,
    )
    assert result.scenario_count == 10
    assert len(result.classification) == 3
    assert result.classification.points_chance.between(0, 1).all()
    assert result.classification.win_chance.between(0, 1).all()
    assert set(result.position_progression.driver_id) == {"A", "B", "C"}
