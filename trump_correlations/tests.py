import os
from pathlib import Path
import pandas as pd


def test_category_names():
    # Resolve project root dynamically (one folder up from trump_correlations)
    project_root = Path(__file__).resolve().parent.parent
    alarms_path = project_root / "data" / "processed" / "second_war.csv"

    print(f"Loading from: {alarms_path}")
    df = pd.read_csv(alarms_path)

    print("Unique categories in unicode:")
    for cat in df["category_desc"].dropna().unique():
        print(cat.encode("unicode_escape").decode("utf-8"))


if __name__ == "__main__":
    test_category_names()