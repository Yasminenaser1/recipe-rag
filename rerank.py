import json
from sentence_transformers import CrossEncoder
from hybrid import hybrid_search, corpus, titles

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
title_to_idx = {t: i for i, t in enumerate(titles)}


def rerank_search(query, n=10, candidates=20):
    shortlist = hybrid_search(query, candidates)
    pairs = [[query, corpus[title_to_idx[t]]] for t in shortlist]
    scores = reranker.predict(pairs)
    ordered = sorted(zip(shortlist, scores), key=lambda x: x[1], reverse=True)
    return [t for t, _ in ordered[:n]]


if __name__ == "__main__":
    cases = [c for c in json.load(open("eval_set.json")) if c["expected"]]
    total = 0.0
    for c in cases:
        results = rerank_search(c["query"], 10)
        rank = next((i + 1 for i, t in enumerate(results) if t in c["expected"]), None)
        rr = 1 / rank if rank else 0.0
        total += rr
        print(f"{c['query']:24} rank {rank if rank else '-':<4} rr {rr:.3f}")
    print(f"\nMRR: {total/len(cases):.3f}  ({len(cases)} queries)")
