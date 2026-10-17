import bm25s
import pandas as pd
from bm25s.tokenization import Tokenized

from hybrid_search_in_rag.data.download_fiqa import DATA_DIR

INDEX_DIR = DATA_DIR.parent / "indexes" / "bm25"

# Loding the corpus
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
print(corpus["text"].iloc[0])
doc_ids = corpus["_id"].tolist()
doc_texts = corpus["text"].tolist()
print(f"Indexing {len(doc_texts)} documents with BM25...")

# Tokenising and indexing the corpus texts
tokens = bm25s.tokenize(doc_texts, stopwords="en")
assert isinstance(tokens, Tokenized)
print(tokens.ids[:1])
print(list(tokens.vocab.items())[:20])

retriever = bm25s.BM25()
retriever.index(tokens)

# Persist the index to disk
INDEX_DIR.mkdir(parents=True, exist_ok=True)
retriever.save(str(INDEX_DIR))
(INDEX_DIR / "doc_ids.txt").write_text("\n".join(doc_ids))


def search_bm25(query: str, k: int = 10) -> list[tuple[str, float]]:
    query_tokens = bm25s.tokenize([query], stopwords="en")
    indices, scores = retriever.retrieve(query_tokens=query_tokens, k=k)
    return [
        (doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0].tolist())
    ]


if __name__ == "__main__":
    query = "Where should I park my rainy-day fund?"
    print(f"\nQuery: {query}\n")
    for i, (doc_id, score) in enumerate(search_bm25(query, k=5), 1):
        text = corpus.loc[corpus["_id"] == doc_id, "text"].iloc[0]
        print(f"{i}. [{score:6.2f}] {doc_id}  {text[:80]}")

    query_tokens = bm25s.tokenize([query], stopwords="en")
    indices, scores = retriever.retrieve(query_tokens=query_tokens, k=10)
    for j, i in enumerate(indices[0].tolist()):
        print(doc_ids[i], float(scores[0][j]))