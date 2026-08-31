import json
import os
from functools import lru_cache

import chromadb
import gradio as gr
from groq import Groq
from sentence_transformers import SentenceTransformer

groq = Groq(api_key=os.environ["GROQ_API_KEY"])


@lru_cache(maxsize=1)
def setup():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    recipes = json.load(open("recipes.json"))

    def short_form(t):
        return t.split("Directions:")[0].strip() if "Directions:" in t else t

    texts = [short_form(r["text"]) for r in recipes]
    embeddings = model.encode(texts)

    client = chromadb.Client()
    coll = client.get_or_create_collection("recipes")
    coll.add(
        ids=[str(r["id"]) for r in recipes],
        embeddings=embeddings.tolist(),
        documents=[r["text"] for r in recipes],
        metadatas=[{"title": r["title"]} for r in recipes],
    )
    return model, coll


def recommend(query):
    if not query.strip():
        return "Tell me what ingredients you have, or what you'd like to make."

    model, coll = setup()
    res = coll.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=5,
    )
    context = "\n\n---\n\n".join(res["documents"][0])

    prompt = f"""Here are 5 recipes from a cookbook:

{context}

The user asked for: {query}

Their request may name ingredients they have, or a dish they want.
Interpret it sensibly.

First decide: does any recipe above actually satisfy the request? If not,
say so plainly and stop - do not recommend the closest thing.

If one does fit, name it, say why, and list what they'd need to buy."""

    try:
        resp = groq.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Something went wrong: {e}"


demo = gr.Interface(
    fn=recommend,
    inputs=gr.Textbox(
        label="What do you have, or what do you want to make?",
        placeholder="chicken and rice",
    ),
    outputs=gr.Markdown(label="Recommendation"),
    title="Recipe RAG",
    description=(
        "Semantic search over 300 recipes, then an LLM picks the best match. "
        "The corpus is mid-century American home cooking, so it knows a lot "
        "about casseroles and very little about pad thai."
    ),
    examples=["chicken and rice", "eggs and cheese", "a dip for a party", "pad thai"],
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
