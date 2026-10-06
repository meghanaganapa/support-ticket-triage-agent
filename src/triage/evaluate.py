"""Evaluate the triage system on the held-out test split.

    python -m triage.evaluate                 # offline (ML + rules), free
    TRIAGE_LLM=anthropic python -m triage.evaluate --limit 40

Writes reports/eval_<backend>.json and prints a markdown summary.
"""

from __future__ import annotations

import argparse
import json
import time

from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data import HARD_DATA, ROOT, load_rows, split, to_ticket
from .llm import get_llm
from .orchestrator import TriageOrchestrator

MANUAL_TRIAGE_MINUTES = 4.0  # assumption: time for a person to read, route and prioritise one ticket


def evaluate(backend: str | None = None, limit: int | None = None, dataset: str = "heldout") -> dict:
    """dataset="heldout": synthetic tickets from templates never seen in training.
    dataset="hard": 30 hand-written tickets with messy, ambiguous, unseen phrasing."""
    rows = load_rows()
    if dataset == "hard":
        train, test = rows, load_rows(HARD_DATA)
    else:
        train, test = split(rows)
    if limit:
        test = test[:limit]
    llm = get_llm(backend)
    orch = TriageOrchestrator(llm=llm, training_rows=train)

    preds, start = [], time.perf_counter()
    for row in test:
        preds.append(orch.triage(to_ticket(row)))
    elapsed = time.perf_counter() - start

    y_cat = [r["category"] for r in test]
    p_cat = [p.category.value for p in preds]
    y_pri = [r["priority"] for r in test]
    p_pri = [p.priority.value for p in preds]
    y_esc = [r["escalate"] for r in test]
    p_esc = [p.escalate for p in preds]

    tp = sum(1 for y, p in zip(y_esc, p_esc) if y and p)
    esc_recall = tp / max(1, sum(y_esc))
    esc_precision = tp / max(1, sum(p_esc))
    auto = sum(1 for p in preds if not p.needs_human_review)
    guard_pass = sum(1 for p in preds if p.guardrail.passed)

    results = {
        "backend": llm.name,
        "dataset": dataset,
        "train_size": len(train),
        "test_size": len(test),
        "category_accuracy": round(accuracy_score(y_cat, p_cat), 3),
        "category_macro_f1": round(f1_score(y_cat, p_cat, average="macro"), 3),
        "priority_accuracy": round(accuracy_score(y_pri, p_pri), 3),
        "escalation_recall": round(esc_recall, 3),
        "escalation_precision": round(esc_precision, 3),
        "auto_queued_pct": round(100 * auto / len(preds), 1),
        "guardrail_pass_pct": round(100 * guard_pass / len(preds), 1),
        "avg_latency_ms": round(1000 * elapsed / len(preds), 1),
        "est_triage_minutes_saved_per_100": round(100 * MANUAL_TRIAGE_MINUTES - 100 * elapsed / len(preds) / 60, 0),
        "llm_usage": getattr(llm, "usage", None).as_dict() if hasattr(llm, "usage") else {},
        "per_category": classification_report(y_cat, p_cat, output_dict=True, zero_division=0),
        "errors": [
            {"id": r["id"], "true": r["category"], "pred": p.category.value, "body": r["body"][:140]}
            for r, p in zip(test, preds) if r["category"] != p.category.value
        ],
    }
    out = ROOT / "reports" / f"eval_{llm.name}_{dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return results


def to_markdown(r: dict) -> str:
    return "\n".join([
        f"| Metric | {r['dataset']} set ({r['backend']}, n={r['test_size']}) |",
        "|---|---|",
        f"| Routing accuracy | {r['category_accuracy']:.1%} |",
        f"| Routing macro-F1 | {r['category_macro_f1']:.3f} |",
        f"| Priority accuracy | {r['priority_accuracy']:.1%} |",
        f"| Escalation recall (urgent tickets caught) | {r['escalation_recall']:.1%} |",
        f"| Escalation precision | {r['escalation_precision']:.1%} |",
        f"| Guardrail pass rate | {r['guardrail_pass_pct']}% |",
        f"| Auto-queued (no human review needed) | {r['auto_queued_pct']}% |",
        f"| Avg latency per ticket | {r['avg_latency_ms']} ms |",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="offline | anthropic | openai")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test tickets")
    ap.add_argument("--dataset", choices=["heldout", "hard", "both"], default="both")
    args = ap.parse_args()
    for ds in (["heldout", "hard"] if args.dataset == "both" else [args.dataset]):
        r = evaluate(args.backend, args.limit, ds)
        print(to_markdown(r))
        if r["errors"]:
            print(f"\n{len(r['errors'])} routing errors, e.g.:")
            for e in r["errors"][:8]:
                print(f"  {e['id']}: true={e['true']} pred={e['pred']} | {e['body']}")
        print()


if __name__ == "__main__":
    main()
