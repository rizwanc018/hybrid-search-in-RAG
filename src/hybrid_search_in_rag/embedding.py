import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from hybrid_search_in_rag.data.download_fiqa import DATA_DIR

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "text-embedding-3-small"

# DATA_DIR
# INDEX_DIR.mkdir(parents=True, exist_ok=True)


# Embed the corpus in batches
def embed_batch(texts: list[str]) -> np.ndarray:
    """Convert text into vectors"""
    response = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in response.data], dtype=np.float32)


def build_index(doc_texts: list[str], batch_size: int = 256) -> np.ndarray:
    chunks = []

    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Embedding"):
        chunks.append(embed_batch(doc_texts[i : i + batch_size]))
    return np.vstack(chunks)


#  Build or load the cached embedding matrix
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
doc_ids = corpus["_id"].tolist()
doc_texts = [t.strip() or "[empty document]" for t in corpus["text"].tolist()]

embeddings_path = DATA_DIR.parent / "indexes" / "dense" / "embeddings.npy"

if embeddings_path.exists():
    print(f"Loading cached embeddings from {embeddings_path}")
    doc_embeddings = np.load(embeddings_path)
else:
    print(f"Embedding {len(doc_texts)} docs (~$0.22 at text-embedding-3-small)")
    doc_embeddings = build_index(doc_texts)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, doc_embeddings)

doc_embeddings_normed = doc_embeddings / np.linalg.norm(
    doc_embeddings, axis=1, keepdims=True
)


def search_dense(query: str, k: int = 10) -> list[tuple[str, float]]:
    """Return the top-k (doc_id, similarity) pairs for a query."""
    query_vec = embed_batch([query])[0]
    query_vec /= np.linalg.norm(query_vec)
    scores = doc_embeddings_normed @ query_vec
    top_k = np.argsort(-scores)[:k]
    return [(doc_ids[i], float(scores[i])) for i in top_k]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_dense(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:.3f}] {doc_id} {text}\n")


# scores = np.array([0.3, 0.9, 0.5, 0.7])
# k = np.argsort(scores)
# kn = np.argsort(-scores)

# query = "Where should I park my rainy-day fund?"
# query_vec = embed_batch([query])[0]
# query_vec /= np.linalg.norm(query_vec)
# scores = doc_embeddings_normed @ query_vec
# top_k = np.argsort(-scores)[:3]
# print(scores)
