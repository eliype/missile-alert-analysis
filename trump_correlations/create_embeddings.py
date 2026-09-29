import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer


def clean_html(text: str) -> str:
    """Strips HTML tags from text content."""
    return re.sub(r"<[^>]+>", "", str(text)).strip()


def generate_truth_embeddings(
    input_csv_path: str = "data/Trump_Second_War_Truths.csv",
    output_pkl_path: str = "data/Trump_Second_War_Truth_Vectors.pkl",
    model_name: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """Loads cleaned truths, encodes text with SentenceTransformer, and saves vectors."""
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)

    df = pd.read_csv(input_csv_path).dropna(subset=["content"])
    df["cleaned_content"] = df["content"].apply(clean_html)

    print(f"Initializing embedding model ({model_name})...")
    model = SentenceTransformer(model_name)

    print("Extracting vectors...")
    embeddings = model.encode(
        df["cleaned_content"].tolist(), show_progress_bar=True
    )

    df_vectors = pd.DataFrame(
        {"date": df["date"], "time": df["time"], "vector": list(embeddings)}
    )

    df_vectors.to_pickle(output_pkl_path)
    print(f"Success! Saved {len(df_vectors)} vectors to: {output_pkl_path}")
    return df_vectors


if __name__ == "__main__":
    generate_truth_embeddings()