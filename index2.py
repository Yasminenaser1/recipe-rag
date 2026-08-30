import json
import chromadb
from sentence_transformers import SentenceTransformer

recipes = json.load(open("recipes.json"))
model = SentenceTransformer("all-MiniLM-L6-v2")

def short_form(text):
    # keep everything up to Directions
    if "Directions:" in text:
        return text.split("Directions:")[0].strip()
    return text

texts = [short_form(r["text"]) for r in recipes]
print("sample:\n", texts[0][:300])

embeddings = model.encode(texts, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("recipes_short")
collection.add(
    ids=[str(r["id"]) for r in recipes],
    embeddings=embeddings.tolist(),
    documents=[r["text"] for r in recipes],
    metadatas=[{"title": r["title"]} for r in recipes],
)
print("indexed:", collection.count())
