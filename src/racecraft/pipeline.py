"""Command-line, reproducible training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from racecraft.data.features import build_decision_dataset
from racecraft.evaluation.backtest import rolling_backtest
from racecraft.models.pace import PaceModel
from racecraft.models.safety_car import SafetyCarModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an F1 decision dataset and rolling backtests."
    )
    parser.add_argument("--laps", type=Path, required=True)
    parser.add_argument("--stints", type=Path, required=True)
    parser.add_argument("--race-control", type=Path)
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/decisions.parquet")
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Used when an input does not already carry a year column",
    )
    args = parser.parse_args()
    laps, stints = pd.read_parquet(args.laps), pd.read_parquet(args.stints)
    events = pd.read_parquet(args.race_control) if args.race_control else None
    data = build_decision_dataset(laps, stints, events)
    if "year" not in data:
        if args.year is None:
            raise ValueError(
                "Input lacks a year column; pass --year for a single-season build"
            )
        data["year"] = args.year
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(args.out, index=False)
    print(f"wrote {len(data):,} decision rows to {args.out}")
    if data.year.nunique() > 1:
        print("pace backtest")
        print(rolling_backtest(data, PaceModel).to_string(index=False))
        print("safety-car backtest")
        print(
            rolling_backtest(
                data, SafetyCarModel, target="neutralisation_next_5"
            ).to_string(index=False)
        )


if __name__ == "__main__":
    main()
