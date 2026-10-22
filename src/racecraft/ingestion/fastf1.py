"""Optional FastF1 loader for 2018+ timing and tyre data.

Install `racecraft[fastf1]`; FastF1 manages its own cache under the chosen path.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def fetch_session(year: int, round_number: int, output_dir: Path, cache_dir: Path) -> Path:
    try:
        import fastf1
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise RuntimeError("Install the optional dependency: pip install -e '.[fastf1]'") from error
    fastf1.Cache.enable_cache(str(cache_dir))
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=True, messages=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fastf1_{year}_{round_number}_race_laps.parquet"
    session.laps.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache one FastF1 race session as Parquet.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw/fastf1"))
    parser.add_argument("--cache", type=Path, default=Path("fastf1_cache"))
    args = parser.parse_args()
    print(fetch_session(args.year, args.round, args.out, args.cache))


if __name__ == "__main__":
    main()
