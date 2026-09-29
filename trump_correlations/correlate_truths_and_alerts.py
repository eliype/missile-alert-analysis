import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from pathlib import Path


def filter_rocket_alarms(
    input_alarms_path: str = None,
    output_alarms_path: str = None,
) -> pd.DataFrame:
    """Filters raw alarm dataset strictly for physical rocket/missile fire."""
    project_root = Path(__file__).resolve().parent.parent

    if input_alarms_path is None:
        input_alarms_path = project_root / "data" / "processed" / "second_war.csv"
    if output_alarms_path is None:
        output_alarms_path = project_root / "data" / "processed" / "second_war_rockets_only.csv"

    print(f"Loading raw alarms from {input_alarms_path}...")
    df = pd.read_csv(input_alarms_path)
    initial_rows = len(df)

    # Corrected unicode escape sequence for rocket/missile fire category
    rocket_category = "\u05d9\u05e8\u05d9 \u05e8\u05e7\u05d8\u05d5\u05ea \u05d5\u05d8\u05d9\u05dc\u05d9\u05dd"
    df_filtered = df[df["category_desc"] == rocket_category].copy()

    os.makedirs(os.path.dirname(output_alarms_path), exist_ok=True)
    df_filtered.to_csv(output_alarms_path, index=False)
    print(
        f"Filtered alarms: {initial_rows:,} -> {len(df_filtered):,} rows saved to {output_alarms_path}"
    )
    return df_filtered


def run_mantel_test(
        pkl_path: str = None,
        alarms_path: str = None,
        n_permutations: int = 1000,
):
    """Executes Distance Matrix Correlation (Mantel Test) with permutation testing."""
    project_root = Path(__file__).resolve().parent.parent

    if pkl_path is None:
        pkl_path = project_root / "data" / "Trump_Second_War_Truth_Vectors.pkl"
    if alarms_path is None:
        alarms_path = project_root / "data" / "processed" / "second_war_rockets_only.csv"

    df_vectors = pd.read_pickle(pkl_path)
    df_alarms = pd.read_csv(alarms_path)

    df_vectors["datetime"] = pd.to_datetime(
        df_vectors["date"].astype(str) + " " + df_vectors["time"].astype(str),
        format="mixed",
    )
    df_alarms["datetime"] = pd.to_datetime(
        df_alarms["date"].astype(str) + " " + df_alarms["time"].astype(str),
        format="mixed",
    )

    min_time = min(
        df_vectors["datetime"].min(), df_alarms["datetime"].min()
    ).floor("10min")
    max_time = max(
        df_vectors["datetime"].max() + pd.Timedelta(hours=24),
        df_alarms["datetime"].max(),
    ).ceil("10min")
    full_idx = pd.date_range(min_time, max_time, freq="10min")

    ten_min_alarms = (
        df_alarms.groupby(df_alarms["datetime"].dt.floor("10min"))
        .size()
        .reindex(full_idx, fill_value=0)
    )

    print("Extracting 24-hour Response Profiles (144 ten-minute intervals)...")
    y_graphs = []
    for _, row in df_vectors.iterrows():
        start_time = row["datetime"].floor("10min")
        target_intervals = pd.date_range(
            start_time, periods=144, freq="10min"
        )
        graph = ten_min_alarms.loc[target_intervals].values
        y_graphs.append(graph)

    X = np.stack(df_vectors["vector"].values)
    Y = np.stack(y_graphs)

    print("Computing pairwise distance matrices...")
    dist_semantic_sq = squareform(pdist(X, metric="cosine"))
    dist_shape_sq = squareform(pdist(Y, metric="euclidean"))

    upper_tri_idx = np.triu_indices(len(X), k=1)
    dist_semantic = dist_semantic_sq[upper_tri_idx]
    dist_shape = dist_shape_sq[upper_tri_idx]

    r_val, naive_p_val = pearsonr(dist_semantic, dist_shape)

    print(
        f"Running Mantel Permutation Test ({n_permutations:,} iterations)..."
    )
    permuted_r_vals = []
    n_elements = len(X)

    for i in range(n_permutations):
        shuffled_idx = np.random.permutation(n_elements)
        shuffled_semantic_sq = dist_semantic_sq[shuffled_idx, :][
            :, shuffled_idx
        ]
        shuffled_semantic = shuffled_semantic_sq[upper_tri_idx]

        r_random, _ = pearsonr(shuffled_semantic, dist_shape)
        permuted_r_vals.append(r_random)

    permuted_r_vals = np.array(permuted_r_vals)
    p_val = np.sum(permuted_r_vals >= r_val) / n_permutations

    print("\n" + "=" * 50)
    print(" MANTEL TEST RESULTS ")
    print("=" * 50)
    print(f"Correlation Coefficient (r): {r_val:.4f}")
    print(f"Permutation P-Value:        {p_val:.4f}")
    print(f"Naive Pearson P-Value:      {naive_p_val:.6f}")

    if p_val < 0.05 and r_val > 0.01:
        print("\n>> Statistically significant correlation detected.")
    else:
        print("\n>> No statistically significant correlation detected.")


if __name__ == "__main__":
    filter_rocket_alarms()
    run_mantel_test()