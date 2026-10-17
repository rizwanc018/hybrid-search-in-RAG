import pandas as pd

from hybrid_search_in_rag.data.download_fiqa import DATA_DIR

# Loading three parquet files
corpus = pd.read_parquet(DATA_DIR / "corpus.parquet")
print(corpus["text"].iloc[0])
queries = pd.read_parquet(DATA_DIR / "queries.parquet")
print(queries["text"].iloc[0])
qrels = pd.read_parquet(DATA_DIR / "qrels.parquet")
print(qrels.iloc[0])

#  Filter the queries to only include the test queries
test_query_ids = set(qrels["query-id"].astype(str))
test_queries = queries[queries["_id"].astype(str).isin(test_query_ids)]
print(f"Queries (test, with qrels): {len(test_queries)}")

# Inspect one query and its relevant docs
query = test_queries.iloc[0]
print("\nExample query")
print(f"  _id:  {query['_id']}")
print(f"  text: {query['text']}")

relevant = qrels[qrels["query-id"].astype(str) == str(query["_id"])]
print(f"\n  Relevant docs for this question: {list(relevant['corpus-id'])}")

relevant_ids = set(relevant["corpus-id"].astype(str))
relevant_docs = corpus[corpus["_id"].astype(str).isin(relevant_ids)]
for _, doc in relevant_docs.iterrows():
    print(f"\n  --- corpus-id: {doc['_id']} ---")
    print(f"  {doc['text']}")
