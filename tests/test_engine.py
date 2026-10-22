import pytest

from racecraft.simulation.engine import (
    RaceState,
    Strategy,
    candidate_strategies,
    evaluate_strategies,
    simulate_strategy,
)


def state() -> RaceState:
    return RaceState(30, 57, 20, "MEDIUM", 8, 92.0)


def test_simulation_has_bounded_positions() -> None:
    result = simulate_strategy(
        state(), Strategy(34, "HARD", "Pit lap 34 → Hard"), n_simulations=100
    )
    assert len(result.finish_position_samples) == 100
    assert result.finish_position_samples.min() >= 1
    assert result.finish_position_samples.max() <= 18
    assert 0 <= result.p_points <= 1
    assert len(result.baseline_time_samples) == 100


def test_stay_out_is_neutral_against_its_own_baseline() -> None:
    result = simulate_strategy(
        state(), Strategy(None, None, "Stay out"), n_simulations=100
    )
    assert result.delta_to_baseline_seconds == pytest.approx(0.0)
    assert result.p_beat_baseline == 0.0


def test_invalid_pit_lap_is_rejected() -> None:
    with pytest.raises(ValueError):
        simulate_strategy(state(), Strategy(30, "SOFT", "bad"), n_simulations=10)


def test_candidates_do_not_pit_after_finish() -> None:
    assert all(
        x.pit_lap is None or x.pit_lap < state().total_laps
        for x in candidate_strategies(state())
    )


def test_degradation_multiplier_must_be_positive() -> None:
    with pytest.raises(ValueError):
        simulate_strategy(
            state(), Strategy(None, None, "Stay out"), degradation_multiplier=0
        )


def test_batch_evaluation_skips_invalid_ui_strategy() -> None:
    results, failures = evaluate_strategies(
        state(),
        [Strategy(None, None, "Stay out"), Strategy(57, "SOFT", "Invalid")],
        n_simulations=5,
    )
    assert len(results) == 1
    assert failures and "Invalid" in failures[0]
