# recipe-rag

A retrieval-augmented generation system over 300 recipes, with an eval harness.

## What it does

Semantic search over recipes by ingredients, then an LLM picks the best match
and lists what you're missing.

- `get_data.py` — pulls 300 recipes from the `corbt/all-recipes` dataset
- `index.py` — embeds with `all-MiniLM-L6-v2`, stores in ChromaDB
- `query.py` — retrieval only, returns top 5 with distances
- `ask.py` — full RAG: retrieval + generation via Groq
- `run_eval.py` — scores retrieval against a hand-built eval set

## Results

**recall@5: 92%** (11/12 scored queries)

All 5 easy queries hit at rank 1. The one failure is "a no-bake dessert" —
the embedding covers the full recipe text, so cooking method gets drowned out
by ingredients and instructions.

## Notes

Building the eval set was more instructive than building the pipeline. Two
early "failures" turned out to be wrong expectations rather than bad retrieval
— the corpus had 10 dip recipes when my answer key listed 2.
