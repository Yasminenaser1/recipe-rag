import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("eval_set.json") as f:
    cases = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes_short")

hits = 0
scored = 0

for c in cases:
    res = collection.query(
        query_embeddings=model.encode([c["query"]]).tolist(),
        n_results=5,
    )
    titles = [m["title"] for m in res["metadatas"][0]]

    if not c["expected"]:
        print(f"[{c['difficulty']}] {c['query']}")
        print(f"    no match expected. got: {titles[0]}")
        continue

    scored += 1
    found = [t for t in titles if t in c["expected"]]
    rank = titles.index(found[0]) + 1 if found else None

    if rank:
        hits += 1
        print(f"[{c['difficulty']}] {c['query']}  ->  HIT at rank {rank}")
    else:
        print(f"[{c['difficulty']}] {c['query']}  ->  MISS")
        print(f"    wanted: {c['expected']}")
        print(f"    got:    {titles}")

print(f"\nrecall@5: {hits}/{scored} = {hits/scored:.0%}")
