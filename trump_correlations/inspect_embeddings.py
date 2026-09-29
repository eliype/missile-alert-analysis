import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.preprocessing import normalize


def test_semantic_search(
    csv_path: str = "data/Trump_Second_War_Truths.csv",
    pkl_path: str = "data/Trump_Second_War_Truth_Vectors.pkl",
    test_query: str = "fake news media",
    top_k: int = 3,
):
    """Validates vector retrieval accuracy against a query."""
    print(f"\n--- Running semantic search test for: '{test_query}' ---")
    df_text = pd.read_csv(csv_path).dropna(subset=["content"])
    df_vectors = pd.read_pickle(pkl_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vector = model.encode(test_query, convert_to_tensor=True).cpu()
    saved_vectors = np.stack(df_vectors["vector"].values)

    similarities = util.cos_sim(query_vector, saved_vectors)[0].numpy()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    for idx in top_indices:
        score = similarities[idx]
        row_text = df_text.iloc[idx]
        row_vec = df_vectors.iloc[idx]
        clean_text = re.sub(
            r"<[^>]+>", "", str(row_text["content"])
        ).strip()
        print(
            f"[Match Score: {score:.4f}] | Date: {row_vec['date']} | Time: {row_vec['time']}"
        )
        print(f"Content snippet: {clean_text[:150]}...\n")


def run_kmeans_elbow(
    pkl_path: str = "data/Trump_Second_War_Truth_Vectors.pkl",
    min_k: int = 2,
    max_k: int = 20,
):
    """Runs K-Means++ across a range of centroids using normalized L2 distance."""
    print("\n--- Running K-Means++ Elbow Analysis ---")
    df_vectors = pd.read_pickle(pkl_path)
    vectors = np.stack(df_vectors["vector"].values)
    vectors_normalized = normalize(vectors, norm="l2")

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=k, init="k-means++", random_state=42, n_init=10
        )
        kmeans.fit(vectors_normalized)
        assigned_centroids = kmeans.cluster_centers_[kmeans.labels_]
        distances = paired_cosine_distances(
            vectors_normalized, assigned_centroids
        )
        print(
            f"K = {k:<2} | Average Cosine Distance: {np.mean(distances):.4f}"
        )


def sample_clusters(
    csv_path: str = "data/Trump_Second_War_Truths.csv",
    pkl_path: str = "data/Trump_Second_War_Truth_Vectors.pkl",
    n_clusters: int = 10,
    samples_per_cluster: int = 5,
):
    """Clusters embedding vectors and prints representative text samples."""
    print(f"\n--- Sampling Clusters for K={n_clusters} ---")
    df_text = pd.read_csv(csv_path).dropna(subset=["content"])
    df_vectors = pd.read_pickle(pkl_path)

    vectors = np.stack(df_vectors["vector"].values)
    vectors_normalized = normalize(vectors, norm="l2")

    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10
    )
    df_text["cluster"] = kmeans.fit_predict(vectors_normalized)

    for cluster_id in range(n_clusters):
        cluster_df = df_text[df_text["cluster"] == cluster_id]
        print(f"\nCLUSTER #{cluster_id} | Total Truths: {len(cluster_df)}")
        samples = cluster_df.sample(
            min(samples_per_cluster, len(cluster_df)), random_state=42
        )

        for i, (_, row) in enumerate(samples.iterrows(), 1):
            clean_content = re.sub(
                r"<[^>]+>", "", str(row["content"])
            ).strip()
            display_content = (
                clean_content[:200] + "..."
                if len(clean_content) > 200
                else clean_content
            )
            print(
                f"  [{i}] {row['date']} {row['time']} | {display_content}"
            )


if __name__ == "__main__":
    test_semantic_search()
    run_kmeans_elbow()
    sample_clusters()