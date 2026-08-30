import json
import chromadb
from sentence_transformers import SentenceTransformer

with open("recipes.json") as f:
    recipes = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [r["text"] for r in recipes]
embeddings = model.encode(texts, show_progress_bar=True)
print("embedding shape:", embeddings.shape)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("recipes")

collection.add(
    ids=[str(r["id"]) for r in recipes],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=[{"title": r["title"]} for r in recipes],
)

print("indexed:", collection.count())
