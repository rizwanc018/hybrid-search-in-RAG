from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "fiqa"
DATA_DIR.mkdir(exist_ok=True)

corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels = load_dataset("BeIR/fiqa-qrels", split="test")

# print(corpus)
# print(queries)
# print(qrels)

# print(corpus[0])  # First document
# print(queries[0])  # First query
# print(qrels[0])  # First relevance judgment


corpus.to_parquet(DATA_DIR / "corpus.parquet")
queries.to_parquet(DATA_DIR / "queries.parquet")
qrels.to_parquet(DATA_DIR / "qrels.parquet")


if __name__ == "__main__":
    print(f"Corpus:  {len(corpus):>6} docs    -> {DATA_DIR / 'corpus.parquet'}")
    print(f"Queries: {len(queries):>6} queries -> {DATA_DIR / 'queries.parquet'}")
    print(f"Qrels:   {len(qrels):>6} judgments -> {DATA_DIR / 'qrels.parquet'}")
