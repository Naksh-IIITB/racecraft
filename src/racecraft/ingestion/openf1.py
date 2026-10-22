"""Polite, cache-first OpenF1 downloader.

OpenF1 data are unofficial; query only the endpoints and dates needed for a study.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
ENDPOINTS = ("sessions", "laps", "stints", "pit", "race_control")


def fetch(endpoint: str, params: dict[str, object], output_dir: Path) -> Path:
    """Fetch one filtered API resource once; return an existing cache if present."""
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Unsupported endpoint: {endpoint}")
    suffix = "_".join(f"{k}-{v}" for k, v in sorted(params.items())).replace("/", "-")
    path = output_dir / f"{endpoint}_{suffix}.parquet"
    if path.exists():
        return path
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30,
                            headers={"User-Agent": "RaceCraft/0.1 (academic strategy research)"})
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    if frame.empty:
        raise ValueError(f"No {endpoint} records returned for {params}")
    frame.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache selected OpenF1 resources by year.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/raw/openf1"))
    parser.add_argument("--session-key", type=int, help="Fetch all modelling resources for one known race session")
    args = parser.parse_args()
    if args.session_key:
        for endpoint in ("laps", "stints", "pit", "race_control"):
            print(fetch(endpoint, {"session_key": args.session_key}, args.out))
    else:
        print(fetch("sessions", {"year": args.year}, args.out))


if __name__ == "__main__":
    main()
