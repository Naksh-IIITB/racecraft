"""Strategy-oriented presets for every venue on the announced 2026 calendar.

Values are deliberately editable priors for simulation, not official circuit data
or trained estimates. Calibrate them from a chosen historical race before making
research claims.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitPreset:
    name: str
    total_laps: int
    clean_lap_seconds: float
    pit_lane_loss: float
    safety_car_probability: float
    degradation_prior: float
    overtaking: str


def _c(
    name: str,
    laps: int,
    pace: float,
    pit: float,
    sc: float,
    deg: float,
    overtaking: str,
) -> CircuitPreset:
    return CircuitPreset(name, laps, pace, pit, sc, deg, overtaking)


# Venue list follows F1's announced 2026 24-round calendar. Madrid remains a
# provisional venue under the calendar's FIA-homologation caveat.
CIRCUITS: dict[str, CircuitPreset] = {
    x.name: x
    for x in [
        _c("Australia — Melbourne", 58, 82.0, 21.5, 0.30, 1.10, "Medium"),
        _c("China — Shanghai", 56, 95.0, 22.0, 0.23, 1.15, "Medium"),
        _c("Japan — Suzuka", 53, 91.0, 22.5, 0.20, 1.20, "Medium"),
        _c("Bahrain — Sakhir", 57, 94.0, 23.0, 0.19, 1.35, "High"),
        _c("Saudi Arabia — Jeddah", 50, 91.0, 22.5, 0.35, 0.85, "Medium"),
        _c("Miami", 57, 91.0, 22.0, 0.30, 1.05, "Medium"),
        _c("Canada — Montreal", 70, 74.0, 20.5, 0.42, 0.95, "High"),
        _c("Monaco", 78, 74.0, 19.0, 0.16, 0.75, "Very low"),
        _c("Spain — Barcelona-Catalunya", 66, 79.0, 22.5, 0.18, 1.15, "Low"),
        _c("Austria — Spielberg", 71, 67.0, 20.5, 0.27, 1.10, "High"),
        _c("Great Britain — Silverstone", 52, 91.0, 23.0, 0.24, 1.20, "Medium"),
        _c("Belgium — Spa-Francorchamps", 44, 105.0, 23.5, 0.32, 1.10, "High"),
        _c("Hungary — Budapest", 70, 79.0, 22.0, 0.18, 1.15, "Low"),
        _c("Netherlands — Zandvoort", 72, 74.0, 22.0, 0.25, 1.00, "Low"),
        _c("Italy — Monza", 53, 81.0, 22.5, 0.20, 0.80, "High"),
        _c("Spain — Madrid (provisional)", 57, 85.0, 22.5, 0.25, 1.05, "Unknown"),
        _c("Azerbaijan — Baku", 51, 103.0, 21.5, 0.38, 0.80, "High"),
        _c("Singapore", 62, 102.0, 27.0, 0.45, 1.20, "Low"),
        _c("United States — Austin", 56, 96.0, 22.0, 0.25, 1.05, "Medium"),
        _c("Mexico — Mexico City", 71, 78.0, 22.5, 0.34, 0.90, "Medium"),
        _c("Brazil — São Paulo", 71, 72.0, 20.5, 0.34, 1.15, "High"),
        _c("United States — Las Vegas", 50, 95.0, 20.0, 0.30, 0.70, "High"),
        _c("Qatar — Lusail", 57, 85.0, 23.0, 0.18, 1.30, "Medium"),
        _c("Abu Dhabi — Yas Marina", 58, 87.0, 22.0, 0.22, 0.90, "Medium"),
    ]
}
