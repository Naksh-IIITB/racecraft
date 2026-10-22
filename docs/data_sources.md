# Data sources and attribution

RaceCraft does not redistribute Formula 1 session data. It fetches source data into a local ignored cache and preserves the source endpoint/session identifier in derived data.

| Source | Used for | Scope |
|---|---|---|
| [OpenF1](https://openf1.org/docs/) | sessions, laps, stints, pit stops, race-control events | Historical data from 2023 onward; use filtered API requests and cache responses. |
| [FastF1](https://github.com/theOehrly/Fast-F1/tree/main/docs) | timing, tyre data, weather, track status, optional telemetry | Timing and telemetry are generally available from 2018 onward. Requires the optional project extra. |
| [Jolpica-F1](https://github.com/jolpica/jolpica-f1/tree/main/docs) | historical results, grids, constructors, standings | Use a descriptive User-Agent and respect its terms/rate limits. |

Each public result must name the source and retrieval date. Data licensing and availability may change; check the primary documentation before a public deployment.
