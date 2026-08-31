import json
import chromadb
from sentence_transformers import SentenceTransformer

cases = [c for c in json.load(open("eval_set.json")) if c["expected"]]
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes_short")


def rank_of_first_hit(titles, expected):
    for i, t in enumerate(titles):
        if t in expected:
            return i + 1
    return None


total = 0.0
for c in cases:
    res = collection.query(
        query_embeddings=model.encode([c["query"]]).tolist(),
        n_results=10,
    )
    titles = [m["title"] for m in res["metadatas"][0]]
    rank = rank_of_first_hit(titles, c["expected"])
    rr = 1 / rank if rank else 0.0
    total += rr
    print(f"{c['query']:24} rank {rank if rank else '-':<4} rr {rr:.3f}")

print(f"\nMRR: {total/len(cases):.3f}  ({len(cases)} queries)")
