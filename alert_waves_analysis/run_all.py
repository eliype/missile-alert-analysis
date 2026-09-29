#!/usr/bin/env python3
"""Rebuild the alert-wave datasets, analysis outputs, and optional tests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


def run(*arguments: str) -> None:
    subprocess.run(arguments, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--online",
        action="store_true",
        help="Refresh from the official Telegram source instead of using the saved snapshot",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the analysis and source-labeling unit tests after regeneration",
    )
    args = parser.parse_args()

    source_command = [
        sys.executable,
        str(HERE / "source_labeling" / "run_pipeline.py"),
    ]
    if not args.online:
        source_command.append("--offline")
    run(*source_command)
    run(sys.executable, str(HERE / "analysis.py"))

    if args.test:
        run(
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(HERE / "tests"),
            "-v",
        )
        run(
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(HERE / "source_labeling" / "tests"),
            "-v",
        )


if __name__ == "__main__":
    main()
