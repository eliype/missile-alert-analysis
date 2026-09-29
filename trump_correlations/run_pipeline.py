import os
from correlate_truths_and_alerts import filter_rocket_alarms, run_mantel_test
from create_embeddings import generate_truth_embeddings
from download_truths import download_and_filter_truths
from inspect_embeddings import (
    run_kmeans_elbow,
    sample_clusters,
    test_semantic_search,
)

def print_new_lines():
    print("\n\n\n\n\n\n\n")


def main():
    print("=== STARTING TRUMP TRUTHS vs. ALERTS RESEARCH PIPELINE ===")

    # 1. Download & Filter Truths
    truths_csv = "data/Trump_Second_War_Truths.csv"
    download_and_filter_truths(output_csv_path=truths_csv)
    print_new_lines()

    # 2. Extract Embeddings
    vectors_pkl = "data/Trump_Second_War_Truth_Vectors.pkl"
    generate_truth_embeddings(
        input_csv_path=truths_csv, output_pkl_path=vectors_pkl
    )
    print_new_lines()

    # 3. Inspect Embeddings
    test_semantic_search(csv_path=truths_csv, pkl_path=vectors_pkl)
    run_kmeans_elbow(pkl_path=vectors_pkl, min_k=2, max_k=20)
    sample_clusters(
        csv_path=truths_csv,
        pkl_path=vectors_pkl,
        n_clusters=10,
        samples_per_cluster=5,
    )
    print_new_lines()

    # 4. Filter Alarms & Run Correlation Analysis
    raw_alarms = os.path.join("data", "processed", "second_war.csv")
    filtered_alarms = os.path.join("data", "processed", "second_war_rockets_only.csv")

    if os.path.exists(raw_alarms):
        filter_rocket_alarms(
            input_alarms_path=raw_alarms, output_alarms_path=filtered_alarms
        )
        run_mantel_test(
            pkl_path=vectors_pkl,
            alarms_path=filtered_alarms,
            n_permutations=1000,
        )
    else:
        print(
            f"\n[Warning] '{raw_alarms}' not found. Place file in 'data/' to run final Mantel test."
        )
    print_new_lines()

    print("\n=== PIPELINE EXECUTION COMPLETE ===")


if __name__ == "__main__":
    main()