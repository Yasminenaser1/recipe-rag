import json
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from retry import with_retry

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("recipes_short")
groq = Groq()

MAX_ATTEMPTS = 3


def search(query, n=5):
    res = collection.query(
        query_embeddings=model.encode([query]).tolist(),
        n_results=n,
    )
    return {
        "titles": [m["title"] for m in res["metadatas"][0]],
        "docs": res["documents"][0],
        "distances": res["distances"][0],
    }


def check(original_query, results):
    prompt = f"""The user asked for: {original_query}

A search returned these recipes:
{chr(10).join(results['titles'])}

Do any of these actually satisfy the request? Consider the request
carefully - a dessert that requires baking does not satisfy a request
for a no-bake dessert.

Return ONLY JSON:
{{"satisfied": true/false, "reason": "one sentence", "better_query": "a reformulated search query, or null if satisfied"}}"""

    resp = with_retry(lambda: groq.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ))
    return json.loads(resp.choices[0].message.content)


def agentic_search(query):
    current = query
    log = []
    tried = {query}

    for attempt in range(MAX_ATTEMPTS):
        results = search(current)
        verdict = check(query, results)

        log.append({
            "attempt": attempt + 1,
            "query_used": current,
            "top_distance": round(results["distances"][0], 3),
            "titles": results["titles"],
            "satisfied": verdict["satisfied"],
            "reason": verdict["reason"],
        })

        if verdict["satisfied"]:
            return results, log

        nxt = verdict.get("better_query")
        if not nxt or nxt in tried:
            return results, log

        tried.add(nxt)
        current = nxt

    return results, log


if __name__ == "__main__":
    q = input("What do you want to make? ").strip()
    results, log = agentic_search(q)

    for step in log:
        print(f"\n--- attempt {step['attempt']}: \"{step['query_used']}\"")
        print(f"    top distance: {step['top_distance']}")
        print(f"    got: {', '.join(step['titles'])}")
        print(f"    satisfied: {step['satisfied']} — {step['reason']}")
