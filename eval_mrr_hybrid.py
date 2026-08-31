import json
from hybrid import hybrid_search

cases = [c for c in json.load(open("eval_set.json")) if c["expected"]]

total = 0.0
for c in cases:
    titles = hybrid_search(c["query"], 10)
    rank = next((i + 1 for i, t in enumerate(titles) if t in c["expected"]), None)
    rr = 1 / rank if rank else 0.0
    total += rr
    print(f"{c['query']:24} rank {rank if rank else '-':<4} rr {rr:.3f}")

print(f"\nMRR: {total/len(cases):.3f}  ({len(cases)} queries)")
