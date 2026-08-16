from __future__ import annotations

from pathlib import Path
from typing import Iterable
from datetime import datetime


START_YEAR = 2012
INPUT_PATH = Path(__file__).with_name("time.md")
OUTPUT_PATH = Path(__file__).with_name("time_with_years.md")


def parse_timestamp_line(line: str) -> tuple[int, int, str] | None:
    """Parse a line like '05/30 15:30' into (month, day, time_part)."""
    text = line.strip()
    if not text:
        return None
    if len(text) < 11 or "/" not in text or ":" not in text:
        return None

    date_part, time_part = text.split(maxsplit=1)
    month_str, day_str = date_part.split("/")
    return int(month_str), int(day_str), time_part.strip()


def fill_years(lines: Iterable[str], start_year: int = START_YEAR) -> list[str]:
    """
    Add a year to timestamps ordered from top to bottom.

    Rule:
    - The first block starts at start_year.
    - If the month increases while reading downward, we treat it as a year rollover
      and subtract one year.
    """
    results: list[str] = []
    current_year = start_year
    previous_month: int | None = None

    for raw_line in lines:
        parsed = parse_timestamp_line(raw_line)
        if parsed is None:
            continue

        month, day, time_part = parsed

        if previous_month is not None and month > previous_month:
            current_year -= 1

        dt = datetime.strptime(
            f"{current_year}-{month:02d}-{day:02d} {time_part}",
            "%Y-%m-%d %H:%M",
        )
        results.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
        previous_month = month

    return results


def main() -> None:
    lines = INPUT_PATH.read_text(encoding="utf-8").splitlines()
    converted = fill_years(lines, start_year=START_YEAR)
    OUTPUT_PATH.write_text("\n".join(converted) + "\n", encoding="utf-8")
    print(f"Saved {len(converted)} timestamps to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
