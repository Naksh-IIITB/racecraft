"""Small, explicit Monte Carlo engine for comparing pit decisions.

The engine is intentionally policy-neutral: it produces assumptions and uncertainty,
not a claim that a counterfactual strategy is historical fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Compound = Literal["SOFT", "MEDIUM", "HARD"]
BASE_DEGRADATION: dict[str, float] = {"SOFT": 0.075, "MEDIUM": 0.045, "HARD": 0.028}


@dataclass(frozen=True)
class RaceState:
    """Information known immediately before a decision lap."""

    current_lap: int
    total_laps: int
    tyre_age: int
    compound: Compound
    current_position: int
    clean_lap_seconds: float
    cars_on_lead_lap: int = 18


@dataclass(frozen=True)
class Strategy:
    pit_lap: int | None
    next_compound: Compound | None
    label: str


@dataclass(frozen=True)
class StrategyResult:
    strategy: Strategy
    total_time_samples: np.ndarray
    finish_position_samples: np.ndarray
    safety_car_probability: float

    @property
    def expected_finish(self) -> float:
        return float(np.mean(self.finish_position_samples))

    @property
    def p_points(self) -> float:
        return float(np.mean(self.finish_position_samples <= 10))

    @property
    def p10_finish(self) -> float:
        return float(np.quantile(self.finish_position_samples, 0.10))

    @property
    def p90_finish(self) -> float:
        return float(np.quantile(self.finish_position_samples, 0.90))


def _lap_time(
    base: float,
    age: np.ndarray,
    compound: str,
    noise: np.ndarray,
    degradation_multiplier: float = 1.0,
) -> np.ndarray:
    # convex degradation makes very long stints increasingly unattractive
    degradation = degradation_multiplier * (
        BASE_DEGRADATION[compound] * age + 0.0012 * age**2
    )
    return base + degradation + noise


def simulate_strategy(
    state: RaceState,
    strategy: Strategy,
    *,
    n_simulations: int = 2_000,
    safety_car_probability: float = 0.18,
    pit_lane_loss: float = 22.0,
    traffic_penalty: float = 0.35,
    degradation_multiplier: float = 1.0,
    seed: int = 42,
) -> StrategyResult:
    """Simulate remaining race time and a calibrated-position proxy.

    A shared seed makes candidate strategies comparable. Position is a transparent
    pace-relative proxy until the full multi-car simulator replaces it.
    """
    if not 0 <= safety_car_probability <= 1:
        raise ValueError("safety_car_probability must be between 0 and 1")
    if degradation_multiplier <= 0:
        raise ValueError("degradation_multiplier must be positive")
    if (
        strategy.pit_lap is not None
        and not state.current_lap < strategy.pit_lap < state.total_laps
    ):
        raise ValueError(
            "pit_lap must be after the current lap and before the final lap"
        )
    if (strategy.pit_lap is None) != (strategy.next_compound is None):
        raise ValueError(
            "pit_lap and next_compound must either both be set or both be None"
        )

    rng = np.random.default_rng(seed)
    laps = np.arange(state.current_lap + 1, state.total_laps + 1)
    total = np.zeros(n_simulations)
    tyre_age = np.full(n_simulations, state.tyre_age, dtype=float)
    compound = state.compound
    sc_triggered = rng.random(n_simulations) < safety_car_probability
    sc_lap = rng.integers(
        state.current_lap + 1, state.total_laps + 1, size=n_simulations
    )

    for lap in laps:
        if strategy.pit_lap == lap:
            # Safety-car pit stops retain a smaller, sampled lane loss.
            loss = np.where(
                sc_triggered & (np.abs(sc_lap - lap) <= 1),
                pit_lane_loss * 0.48,
                pit_lane_loss,
            )
            total += loss + rng.normal(0, 0.65, n_simulations)
            tyre_age = np.zeros(n_simulations)
            compound = strategy.next_compound or compound
        green_noise = rng.normal(0, 0.42, n_simulations)
        lap_times = _lap_time(
            state.clean_lap_seconds,
            tyre_age,
            compound,
            green_noise,
            degradation_multiplier,
        )
        # SC/VSC reduces pace variation and erases part of an existing pace advantage.
        on_sc = sc_triggered & (np.abs(sc_lap - lap) <= 1)
        lap_times = np.where(
            on_sc,
            state.clean_lap_seconds + 25.0 + rng.normal(0, 0.2, n_simulations),
            lap_times,
        )
        total += lap_times + traffic_penalty * (tyre_age < 2)
        tyre_age += 1

    # 0.8 sec/race is one notional position. Noise acknowledges other cars are not yet jointly simulated.
    relative = total - np.median(total)
    position = state.current_position + np.rint(
        relative / 0.8 + rng.normal(0, 1.15, n_simulations)
    )
    position = np.clip(position, 1, state.cars_on_lead_lap).astype(int)
    return StrategyResult(strategy, total, position, safety_car_probability)


def candidate_strategies(state: RaceState, horizon: int = 12) -> list[Strategy]:
    """Create practical one-stop candidates; no impossible end-of-race stops."""
    choices = [Strategy(None, None, "Stay out")]
    for lap in range(
        state.current_lap + 1,
        min(state.total_laps - 1, state.current_lap + horizon) + 1,
    ):
        for compound in ("SOFT", "MEDIUM", "HARD"):
            choices.append(
                Strategy(lap, compound, f"Pit lap {lap} → {compound.title()}")
            )
    return choices


def evaluate_strategies(
    state: RaceState, strategies: list[Strategy], **simulation_kwargs: object
) -> tuple[list[StrategyResult], list[str]]:
    """Evaluate a batch without allowing one invalid UI action to crash the app."""
    results: list[StrategyResult] = []
    failures: list[str] = []
    for strategy in strategies:
        try:
            results.append(simulate_strategy(state, strategy, **simulation_kwargs))
        except (TypeError, ValueError) as error:
            failures.append(f"{strategy.label}: {error}")
    return results, failures
