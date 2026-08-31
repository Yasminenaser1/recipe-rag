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

## Agentic RAG: made things slightly worse

Added a retrieve → evaluate → reformulate loop. An LLM checks whether the
retrieved recipes satisfy the request and reformulates the query if not,
up to 3 attempts, with repeat detection to stop pointless loops.

Judge scores went down on two queries: "eggs and cheese" dropped from 3 to 2
on groundedness, "pad thai" from 3 to 2 on helpfulness. Everything else was
unchanged. Cost went up 2-3x in LLM calls.

Why it didn't help: reformulation only pays off when the right document
exists but the first phrasing missed it. First-pass retrieval was already at
100% recall@5, so a second search had nothing left to find. The queries that
fail (pad thai, sushi) fail because the corpus lacks the recipe entirely.
