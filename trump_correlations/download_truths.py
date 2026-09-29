import os
import pandas as pd
import ssl

def download_and_filter_truths(
    output_csv_path: str = "data/Trump_Second_War_Truths.csv",
    start_date: str = "2026-02-28",
    end_date: str = "2026-06-17",
) -> pd.DataFrame:
    """Downloads Truth Social archive and filters by date range."""
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    print("Downloading live archive of @realDonaldTrump...")

    url = "https://ix.cnn.io/data/truth-social/truth_archive.csv"
    ssl._create_default_https_context = ssl._create_unverified_context
    df = pd.read_csv(url)

    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    start_dt = pd.to_datetime(start_date, utc=True)
    end_dt = pd.to_datetime(end_date, utc=True)

    mask = (df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)
    df_filtered = df[mask].copy()

    df_filtered["date"] = df_filtered["created_at"].dt.date
    df_filtered["time"] = df_filtered["created_at"].dt.time

    cols_to_keep = [
        "id",
        "date",
        "time",
        "content",
        "replies_count",
        "reblogs_count",
        "favourites_count",
        "url",
    ]
    available_cols = [c for c in cols_to_keep if c in df_filtered.columns]
    df_final = df_filtered[available_cols]

    df_final.to_csv(output_csv_path, index=False)
    print(
        f"Extraction complete! Saved {len(df_final)} truths to: {output_csv_path}"
    )
    return df_final


if __name__ == "__main__":
    download_and_filter_truths()