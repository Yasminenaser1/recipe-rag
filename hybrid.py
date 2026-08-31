import json
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

recipes = json.load(open("recipes.json"))
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes_short")


def short_form(t):
    return t.split("Directions:")[0].strip() if "Directions:" in t else t


corpus = [short_form(r["text"]) for r in recipes]
titles = [r["title"] for r in recipes]
tokenized = [c.lower().split() for c in corpus]
bm25 = BM25Okapi(tokenized)


def vector_search(query, n=20):
    res = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n,
    )
    return [int(i) for i in res["ids"][0]]


def bm25_search(query, n=20):
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:n]


def rrf(lists, k=60):
    """Reciprocal rank fusion - merge ranked lists without needing
    comparable scores. A doc at rank r in a list contributes 1/(k+r)."""
    scores = {}
    for lst in lists:
        for rank, doc_id in enumerate(lst, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(query, n=10):
    v = vector_search(query)
    b = bm25_search(query)
    fused = rrf([v, b])[:n]
    return [titles[i] for i in fused]


if __name__ == "__main__":
    for q in ["a no-bake dessert", "chicken and rice", "something with apples"]:
        print(f"\n{q}")
        for i, t in enumerate(hybrid_search(q, 5), 1):
            print(f"  {i}. {t}")
