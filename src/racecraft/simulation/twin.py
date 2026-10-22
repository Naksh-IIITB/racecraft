"""A compact all-car digital twin for F1-style race-strategy research.

It is designed for counterfactual analysis, not to reproduce proprietary team
simulation systems. Every policy and parameter is visible and replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .engine import BASE_DEGRADATION, Compound

TrackState = Literal["GREEN", "VSC", "SAFETY_CAR"]


@dataclass(frozen=True)
class DriverProfile:
    driver_id: str
    baseline_pace: float
    grid_position: int
    compound: Compound = "MEDIUM"
    tyre_age: int = 0
    elapsed_offset_seconds: float = 0.0


@dataclass(frozen=True)
class PitPolicy:
    """An interpretable opponent or target policy.

    `planned_pit_lap` is a desired lap. Safety-car opportunities may pull a stop
    forward by `sc_opportunism` laps, making competitor behaviour explicit.
    """

    planned_pit_lap: int | None
    next_compound: Compound = "HARD"
    sc_opportunism: int = 2


@dataclass(frozen=True)
class RaceEvent:
    lap: int
    state: TrackState
    duration_laps: int


@dataclass(frozen=True)
class TwinResult:
    classification: pd.DataFrame
    laps: pd.DataFrame
    pit_log: pd.DataFrame
    events: tuple[RaceEvent, ...]


@dataclass(frozen=True)
class TwinEnsembleResult:
    """Distributional finish forecast from multiple complete-race simulations."""

    classification: pd.DataFrame
    scenario_count: int


class EventHazard:
    """Circuit-aware, calibrated-model placeholder for event-time sampling.

    Replace `base_hazard` with a survival model while preserving this interface.
    """

    def __init__(self, base_hazard: float = 0.012, vsc_share: float = 0.30):
        self.base_hazard = base_hazard
        self.vsc_share = vsc_share

    def sample(
        self, start_lap: int, total_laps: int, rng: np.random.Generator
    ) -> tuple[RaceEvent, ...]:
        events: list[RaceEvent] = []
        protected_until = start_lap
        for lap in range(start_lap + 1, total_laps + 1):
            if lap <= protected_until:
                continue
            # Incidents are a little more likely early and late, a transparent prior.
            progress = lap / total_laps
            hazard = self.base_hazard * (
                1.25 if progress < 0.2 or progress > 0.8 else 1.0
            )
            if rng.random() < hazard:
                state: TrackState = (
                    "VSC" if rng.random() < self.vsc_share else "SAFETY_CAR"
                )
                duration = int(
                    rng.integers(1, 3) if state == "VSC" else rng.integers(2, 5)
                )
                events.append(RaceEvent(lap, state, duration))
                protected_until = lap + duration
        return tuple(events)


class PitWindowForecaster:
    """A transparent opponent pit probability baseline.

    It is deliberately a feature-compatible baseline for a later calibrated
    classifier: `predict_probability` accepts only pre-decision state.
    """

    def predict_probability(
        self,
        *,
        lap: int,
        tyre_age: int,
        planned_lap: int | None,
        track_state: TrackState,
    ) -> float:
        if planned_lap is None:
            return 0.0
        distance = abs(lap - planned_lap)
        schedule_signal = float(np.exp(-distance / 1.5))
        wear_signal = max(0.0, (tyre_age - 16) / 20)
        neutralization_bonus = 0.45 if track_state != "GREEN" else 0.0
        return min(
            0.98,
            0.04 + 0.70 * schedule_signal + 0.20 * wear_signal + neutralization_bonus,
        )


def _track_state(lap: int, events: tuple[RaceEvent, ...]) -> TrackState:
    for event in events:
        if event.lap <= lap < event.lap + event.duration_laps:
            return event.state
    return "GREEN"


def _lap_time(
    profile: DriverProfile,
    tyre_age: int,
    compound: Compound,
    track_state: TrackState,
    rng: np.random.Generator,
) -> float:
    if track_state == "SAFETY_CAR":
        return profile.baseline_pace + 28 + rng.normal(0, 0.25)
    if track_state == "VSC":
        return profile.baseline_pace + 16 + rng.normal(0, 0.25)
    degradation = BASE_DEGRADATION[compound] * tyre_age + 0.0012 * tyre_age**2
    return profile.baseline_pace + degradation + rng.normal(0, 0.35)


def simulate_twin(
    profiles: list[DriverProfile],
    policies: dict[str, PitPolicy],
    *,
    total_laps: int,
    start_lap: int = 0,
    target_id: str | None = None,
    target_strategy: PitPolicy | None = None,
    hazard: EventHazard | None = None,
    seed: int = 2026,
) -> TwinResult:
    """Run a single all-car, event-aware counterfactual race simulation."""
    if len({p.driver_id for p in profiles}) != len(profiles):
        raise ValueError("driver_id values must be unique")
    if total_laps <= start_lap:
        raise ValueError("total_laps must exceed start_lap")
    rng = np.random.default_rng(seed)
    events = (hazard or EventHazard()).sample(start_lap, total_laps, rng)
    policies = dict(policies)
    if target_id and target_strategy:
        if target_id not in {p.driver_id for p in profiles}:
            raise ValueError("target_id is not in profiles")
        policies[target_id] = target_strategy

    state = {
        p.driver_id: {
            "age": p.tyre_age,
            "compound": p.compound,
            "elapsed": p.elapsed_offset_seconds,
            "stops": 0,
        }
        for p in profiles
    }
    records: list[dict] = []
    pit_log: list[dict] = []
    forecaster = PitWindowForecaster()
    for lap in range(start_lap + 1, total_laps + 1):
        track = _track_state(lap, events)
        for profile in profiles:
            s = state[profile.driver_id]
            policy = policies.get(profile.driver_id, PitPolicy(None))
            probability = forecaster.predict_probability(
                lap=lap,
                tyre_age=s["age"],
                planned_lap=policy.planned_pit_lap,
                track_state=track,
            )
            # A plan is honoured on its scheduled lap. An early stop is allowed only
            # during a neutralisation window, preventing an opaque random policy.
            sc_window = (
                track != "GREEN"
                and policy.planned_pit_lap is not None
                and lap >= policy.planned_pit_lap - policy.sc_opportunism
            )
            pit = s["stops"] == 0 and (
                lap == policy.planned_pit_lap
                or (sc_window and rng.random() < probability)
            )
            pit_loss = 0.0
            if pit:
                pit_loss = (
                    11.0 if track == "SAFETY_CAR" else 14.0 if track == "VSC" else 22.0
                )
                pit_loss += rng.normal(0, 0.55)
                s["age"], s["compound"], s["stops"] = 0, policy.next_compound, 1
                pit_log.append(
                    {
                        "driver_id": profile.driver_id,
                        "lap": lap,
                        "track_state": track,
                        "pit_loss": pit_loss,
                        "compound": s["compound"],
                    }
                )
            lap_seconds = (
                _lap_time(profile, s["age"], s["compound"], track, rng) + pit_loss
            )
            s["elapsed"] += lap_seconds
            s["age"] += 1
            records.append(
                {
                    "lap": lap,
                    "driver_id": profile.driver_id,
                    "lap_seconds": lap_seconds,
                    "elapsed_seconds": s["elapsed"],
                    "tyre_age": s["age"],
                    "compound": s["compound"],
                    "track_state": track,
                    "pit_probability": probability,
                    "pitted": pit,
                }
            )
    laps = pd.DataFrame(records)
    laps["position"] = (
        laps.groupby("lap")["elapsed_seconds"].rank(method="first").astype(int)
    )
    last = laps.sort_values("lap").groupby("driver_id").tail(1).sort_values("position")
    classification = last[
        ["position", "driver_id", "elapsed_seconds", "compound", "tyre_age"]
    ].reset_index(drop=True)
    return TwinResult(classification, laps, pd.DataFrame(pit_log), events)


def simulate_twin_ensemble(
    profiles: list[DriverProfile],
    policies: dict[str, PitPolicy],
    *,
    total_laps: int,
    start_lap: int,
    target_id: str,
    target_strategy: PitPolicy,
    hazard: EventHazard,
    n_simulations: int = 200,
    seed: int = 2026,
) -> TwinEnsembleResult:
    """Forecast all cars while sampling race events and rival pit calls."""
    if n_simulations < 10:
        raise ValueError("n_simulations must be at least 10")
    if target_id not in {profile.driver_id for profile in profiles}:
        raise ValueError("target_id is not in profiles")

    scenario_rows: list[pd.DataFrame] = []
    for scenario in range(n_simulations):
        result = simulate_twin(
            profiles,
            policies,
            total_laps=total_laps,
            start_lap=start_lap,
            target_id=target_id,
            target_strategy=target_strategy,
            hazard=hazard,
            seed=seed + scenario,
        )
        scenario_rows.append(
            result.classification.assign(scenario=scenario)[
                ["scenario", "driver_id", "position"]
            ]
        )

    positions = pd.concat(scenario_rows, ignore_index=True)
    grouped = positions.groupby("driver_id")["position"]
    summary = grouped.agg(
        expected_position="mean",
        median_position="median",
        best_case=lambda values: values.quantile(0.10),
        worst_case=lambda values: values.quantile(0.90),
        points_chance=lambda values: (values <= 10).mean(),
        win_chance=lambda values: (values == 1).mean(),
    ).reset_index()
    summary["expected_position"] = summary["expected_position"].round(1)
    summary["median_position"] = summary["median_position"].round().astype(int)
    summary["best_case"] = summary["best_case"].round().astype(int)
    summary["worst_case"] = summary["worst_case"].round().astype(int)
    summary["points_chance"] = summary["points_chance"].clip(0, 1)
    summary["win_chance"] = summary["win_chance"].clip(0, 1)
    summary = summary.sort_values(["expected_position", "driver_id"]).reset_index(drop=True)
    return TwinEnsembleResult(summary, n_simulations)


def demo_grid(size: int = 20) -> tuple[list[DriverProfile], dict[str, PitPolicy]]:
    """Synthetic grid only—use source adapters for actual research inputs."""
    profiles, policies = [], {}
    for n in range(1, size + 1):
        driver = f"CAR-{n:02d}"
        profiles.append(DriverProfile(driver, 91.2 + n * 0.11, n, "MEDIUM", 10))
        policies[driver] = PitPolicy(29 + (n % 6), "HARD", 2)
    return profiles, policies
