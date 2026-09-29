#!/usr/bin/env python3
"""Generate the two war-period alert datasets from the raw alert export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = DATA_DIR / "raw" / "israel-alerts.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "processed"

PERIODS = {
    "first_war.csv": ("2025-06-13", "2025-06-24"),
    "second_war.csv": ("2026-02-28", "2026-06-17"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"source alert CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for generated CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def normalized_date(value: str) -> str:
    day, month, year = value.split(".")
    return f"{year}-{month}-{day}"


def generate(source: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    writers = {}
    counts = {filename: 0 for filename in PERIODS}

    try:
        with source.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames is None:
                raise ValueError(f"Input CSV has no header: {source}")

            for filename in PERIODS:
                handle = (output_dir / filename).open("w", encoding="utf-8", newline="")
                handles[filename] = handle
                writer = csv.DictWriter(handle, fieldnames=reader.fieldnames)
                writers[filename] = writer
                writer.writeheader()

            for row in reader:
                date = normalized_date(row["date"])
                for filename, (start, end) in PERIODS.items():
                    if start <= date <= end:
                        output_row = row.copy()
                        output_row["date"] = date
                        output_row["time"] = f'{row["time"][:5]}:00'
                        writers[filename].writerow(output_row)
                        counts[filename] += 1
    finally:
        for handle in handles.values():
            handle.close()

    return counts


def main() -> None:
    args = parse_args()
    counts = generate(args.input, args.output_dir)
    for filename, count in counts.items():
        print(f"Wrote {count:,} rows to {args.output_dir / filename}")


if __name__ == "__main__":
    main()
