"""Local, review-first extraction from F1 timing screenshots.

Apple Vision reads text on macOS. The parser is tuned for a familiar leaderboard
layout but intentionally labels uncertain values for user review; a screenshot
does not contain sufficient information to infer every car's clean pace.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image

POSITION = re.compile(r"^\s*(?:P)?([1-9]|1[0-9]|20)(?:\b|\|)")
TYRE_AGE = re.compile(r"^\s*([0-9]{1,2})(?:\s|\(|$)")
NUMBER = re.compile(r"^[+\-]?(?:[0-9]+:)?[0-9]+(?:\.[0-9]+)?$")


def _vision_blocks(image_path: Path) -> list[dict[str, float | str]]:
    if platform.system() != "Darwin":
        raise RuntimeError(
            "Screenshot OCR currently uses Apple's built-in Vision engine and requires macOS."
        )
    script = Path(__file__).with_name("macos_vision.swift")
    binary = Path(tempfile.gettempdir()) / "racecraft-vision-ocr"
    try:
        # `swift script.swift` compiles on every request. Cache the tiny helper so
        # a dashboard click stays responsive after its first run.
        if not binary.exists() or binary.stat().st_mtime < script.stat().st_mtime:
            subprocess.run(
                ["swiftc", str(script), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        completed = subprocess.run(
            [str(binary), str(image_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "Apple Swift is unavailable; use CSV snapshot upload instead."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Screenshot OCR timed out. Try a smaller, sharper screenshot."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Apple Vision OCR failed: {error.stderr.strip()}"
        ) from error
    return json.loads(completed.stdout)


def _as_seconds(value: str) -> float:
    """Read either a timing delta (``+7.834``) or a ``minutes:seconds`` time."""
    cleaned = value.strip().lstrip("+")
    if ":" in cleaned:
        minutes, seconds = cleaned.split(":", maxsplit=1)
        return int(minutes) * 60 + float(seconds)
    return float(cleaned)


def _driver_name(value: str) -> str:
    """Remove OCR'd position markers and visual separators from a driver cell."""
    without_position = POSITION.sub("", value)
    return re.sub(r"[^A-Za-z ]", "", without_position).strip().upper()


def _tyre_colour(image: Image.Image, x: float, y: float) -> str:
    """Classify the coloured tyre mark beside OCR'd age; UNKNOWN stays reviewable."""
    width, height = image.size
    # Start well to the right of the yellow tyre-age digits. Otherwise the
    # number itself can be mistaken for a Medium sidewall.
    left, right = int((x + 0.016) * width), int(min(1, x + 0.050) * width)
    # A timing row is roughly 30 px in a 1080 px capture. A taller crop would
    # mix the coloured sidewalls from the adjacent cars.
    top, bottom = int((1 - y - 0.014) * height), int((1 - y + 0.014) * height)
    pixels = list(
        image.crop((left, max(0, top), right, min(height, bottom)))
        .convert("RGB")
        .getdata()
    )
    red = sum(r > 160 and g < 120 and b < 120 for r, g, b in pixels)
    yellow = sum(r > 150 and g > 125 and b < 100 for r, g, b in pixels)
    bright = sum(r > 170 and g > 170 and b > 170 for r, g, b in pixels)
    if red > max(yellow, bright, 12):
        return "SOFT"
    if yellow > max(red, bright, 12):
        return "MEDIUM"
    if bright > max(red, yellow, 18):
        return "HARD"
    return "UNKNOWN"


def extract_f1_timing_snapshot(
    image_bytes: bytes, fallback_clean_lap: float
) -> pd.DataFrame:
    """Extract an editable all-car snapshot from a leaderboard screenshot.

    The resulting `clean_lap_seconds` uses a user-supplied fallback because a
    one-frame leaderboard normally doesn't show comparable clean pace for all cars.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(image_bytes)
        image_path = Path(handle.name)
    try:
        image = Image.open(image_path).convert("RGB")
        blocks = _vision_blocks(image_path)
    finally:
        image_path.unlink(missing_ok=True)

    # Vision uses bottom-left coordinates. Group into screen rows with a small tolerance.
    rows: list[list[dict[str, float | str]]] = []
    for block in sorted(blocks, key=lambda item: -float(item["y"])):
        row = next(
            (
                candidate
                for candidate in rows
                if abs(float(candidate[0]["y"]) - float(block["y"])) < 0.014
            ),
            None,
        )
        if row is None:
            rows.append([block])
        else:
            row.append(block)

    extracted: list[dict[str, object]] = []
    for row in rows:
        cells = sorted(row, key=lambda item: float(item["x"]))
        row_y = float(cells[0]["y"])
        # Restrict parsing to the leaderboard body, not the event title/menu.
        if not 0.065 <= row_y <= 0.825:
            continue
        position_cell = next(
            (
                c
                for c in cells
                if float(c["x"]) < 0.12 and POSITION.match(str(c["text"]))
            ),
            None,
        )
        position_match = (
            POSITION.match(str(position_cell["text"])) if position_cell else None
        )
        position = int(position_match.group(1)) if position_match else None
        name_cells = [
            c
            for c in cells
            if float(c["x"]) < 0.18 and re.search(r"[A-Za-z]", str(c["text"]))
        ]
        driver_id = " ".join(_driver_name(str(c["text"])) for c in name_cells).strip()
        # F1's position number is sometimes an icon or too low-contrast for
        # OCR. The board itself is ordered, so recover only missing labels from
        # that visible order; users still see and can edit every value.
        if position is None and driver_id:
            position = len(extracted) + 1
        if position is None or not driver_id or driver_id in {"POSITION", "LAP TIME"}:
            continue
        gap_cell = next(
            (
                c
                for c in cells
                # In F1's desktop board, gap-to-leader sits left of the
                # interval column. Do not substitute interval when OCR missed
                # a gap: a blank is safer and explicitly reviewable.
                if 0.22 <= float(c["x"]) < 0.29
                and NUMBER.fullmatch(str(c["text"]).replace("+", "").strip())
            ),
            None,
        )
        interval_cell = next(
            (
                c
                for c in cells
                if 0.29 <= float(c["x"]) < 0.34
                and NUMBER.fullmatch(str(c["text"]).strip())
            ),
            None,
        )
        if position == 1:
            gap = 0.0
        elif gap_cell:
            gap = _as_seconds(str(gap_cell["text"]))
        elif (
            interval_cell
            and extracted
            and position == int(extracted[-1]["position"]) + 1
            and pd.notna(extracted[-1]["gap_to_leader_seconds"])
        ):
            # Some F1 layouts visually omit the repeated leader-gap for cars
            # further down the board. In that case, accumulate the explicit
            # interval to the preceding classified car.
            gap = float(extracted[-1]["gap_to_leader_seconds"]) + _as_seconds(
                str(interval_cell["text"])
            )
        else:
            gap = float("nan")
        tyre_cell = next(
            (
                c
                for c in reversed(cells)
                if float(c["x"]) > 0.86 and TYRE_AGE.match(str(c["text"]))
            ),
            None,
        )
        tyre_age = (
            int(TYRE_AGE.match(str(tyre_cell["text"])).group(1))
            if tyre_cell
            else 0
        )
        compound = (
            _tyre_colour(image, float(tyre_cell["x"]), float(tyre_cell["y"]))
            if tyre_cell
            else "UNKNOWN"
        )
        extracted.append(
            {
                "driver_id": driver_id,
                "position": position,
                "gap_to_leader_seconds": gap,
                "clean_lap_seconds": fallback_clean_lap,
                "tyre_age": tyre_age,
                "compound": compound,
                "planned_pit_lap": pd.NA,
                "next_compound": "HARD",
            }
        )
    if len(extracted) < 2:
        raise ValueError(
            "Could not find at least two leaderboard rows. Upload a sharp, uncropped leaderboard screenshot."
        )
    return pd.DataFrame(extracted).sort_values("position").reset_index(drop=True)
