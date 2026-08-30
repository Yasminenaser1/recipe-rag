import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from retry import with_retry

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes")
groq = Groq()


def answer(query):
    res = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=5,
    )
    retrieved = res["documents"][0]
    titles = [m["title"] for m in res["metadatas"][0]]

    context = "\n\n---\n\n".join(retrieved)
    prompt = f"""Here are 5 recipes from a cookbook:

{context}

The user asked for: {query}

Their request may name ingredients they have, or a dish or type of food
they want. Interpret it sensibly.

First decide: does any recipe above actually satisfy the request? If not,
say so plainly and stop - do not recommend the closest thing. Recommending
an unsuitable recipe is worse than saying you have nothing.

If one does fit, name it, say why, and list what they'd need to buy."""

    resp = with_retry(lambda: groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    ))
    return {
        "query": query,
        "retrieved_titles": titles,
        "retrieved_docs": retrieved,
        "answer": resp.choices[0].message.content,
    }


cases = json.load(open("eval_set.json"))
results = [answer(c["query"]) for c in cases]

json.dump(results, open("answers.json", "w"), indent=2)
print(f"generated {len(results)} answers")
