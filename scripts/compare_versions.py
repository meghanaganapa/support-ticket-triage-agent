"""Paired comparison of two versions on the same tickets (the locked test set).

Each ticket is scored by both versions, then 2000 bootstrap resamples of the
TICKETS give a 95% interval for the difference (v_new - v_old). If the interval
excludes 0, the change is unlikely to be luck; if it straddles 0, it may be noise.

    python scripts/compare_versions.py old_preds.json new_preds.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def paired_diff(old: list[float], new: list[float], seed: int = 0, n: int = 2000) -> dict:
    rnd = random.Random(seed)
    idx = range(len(old))
    diffs = []
    for _ in range(n):
        s = [rnd.choice(idx) for _ in idx]
        diffs.append(sum(new[i] for i in s) / len(s) - sum(old[i] for i in s) / len(s))
    diffs.sort()
    point = sum(new) / len(new) - sum(old) / len(old)
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]
    return {"diff": round(point, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "verdict": "improved" if lo > 0 else ("worse" if hi < 0 else "within noise")}


def vectors(rows: list[dict], preds: dict) -> dict[str, list[float]]:
    v = {"routing_accuracy": [], "priority_accuracy": [], "escalation_correct": [], "auto_queued": [],
         "retrieval_hit_at_2": []}
    for r in rows:
        p = preds[r["id"]]
        v["routing_accuracy"].append(float(p["category"] == r["category"]))
        v["priority_accuracy"].append(float(p["priority"] == r["priority"]))
        v["escalation_correct"].append(float(p["escalate"] == r["escalate"]))
        v["auto_queued"].append(float(not p["review"]))
        if r.get("kb"):
            v["retrieval_hit_at_2"].append(float(r["kb"] in p["articles"]))
    return v


def main() -> None:
    with open("data/test_locked.jsonl") as f:
        rows = [json.loads(line) for line in f]
    old, new = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    vo, vn = vectors(rows, old), vectors(rows, new)
    urgent = [i for i, r in enumerate(rows) if r["escalate"]]
    out = {k: paired_diff(vo[k], vn[k]) for k in vo}
    out["escalation_recall"] = paired_diff([vo["escalation_correct"][i] for i in urgent],
                                           [vn["escalation_correct"][i] for i in urgent])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
