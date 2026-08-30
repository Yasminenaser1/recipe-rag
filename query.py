import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes")

q = input("What ingredients do you have? ").strip()

q_emb = model.encode([q]).tolist()
results = collection.query(query_embeddings=q_emb, n_results=5)

print(f"\nTop 5 for: {q}\n")
for title, dist in zip(
    [m["title"] for m in results["metadatas"][0]],
    results["distances"][0],
):
    print(f"  {dist:.3f}  {title}")
