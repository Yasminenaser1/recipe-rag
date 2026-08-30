# recipe-rag

A retrieval-augmented generation system over 300 recipes, with evals for both
retrieval and generation quality.

## Files

- `get_data.py` — pulls 300 recipes from the `corbt/all-recipes` dataset
- `index.py` / `index2.py` — embeds with `all-MiniLM-L6-v2`, stores in ChromaDB
- `query.py` — retrieval only
- `generate_answers.py` — full RAG: retrieval + generation via Groq
- `run_eval.py` — scores retrieval against a hand-built eval set
- `judge.py` — LLM-as-judge scoring of generation quality

## Results

**Retrieval: recall@5 went 92% → 100%.** The failing case was "a no-bake
dessert," which returned five cakes. Embedding only the title and ingredients
instead of the full recipe text fixed it — the instructions were diluting the
signal. Caveat: the recovered case landed at rank 5, so the win is fragile, and
one other query dropped from rank 2 to rank 3.

**Generation: LLM-as-judge, validated against my own hand scores.** 14 of 18
scores matched exactly. The disagreements were informative — one was my rubric
being ambiguous, one was the judge hallucinating a groundedness failure.

**A prompt bug found by measuring.** The prompt said `The user has: {query}`,
which framed every query as ingredients on hand. So "pad thai" became "I have
some pad thai," and the system recommended a tuna casserole. Rewriting that
framing moved every failing case to 3/3.

## Notes

Building the eval set taught more than building the pipeline. Two early
"failures" turned out to be wrong expectations rather than bad retrieval — the
corpus had 10 dip recipes when my answer key listed 2.
