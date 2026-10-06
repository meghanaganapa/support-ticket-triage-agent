"""Evaluation harness.

Datasets:
    heldout  synthetic tickets from templates never seen in training (n=106)
    dev      30 hand-written tickets; the ONLY set used to tune thresholds
    test     90 hand-written tickets, committed before any v0.2 change and
             checksum-locked; report it, never tune on it

Backends:
    offline    ML + rules, free (default)
    anthropic  single LLM calls per agent (Claude)
    openai     OpenAI-compatible (OpenAI, Azure OpenAI, Ollama)
    agent-sdk  Claude Agent SDK orchestrator calling the agents as tools

    python -m triage.evaluate --dataset dev
    python -m triage.evaluate --dataset test --backend agent-sdk --limit 20

Every headline metric is reported with a 95% bootstrap confidence interval,
because on 90 tickets (14 urgent) one ticket moves recall by 7 points.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence

from sklearn.metrics import f1_score

from .data import DEV_DATA, ROOT, TEST_DATA, load_rows, split, to_ticket, training_rows, verify_test_set

BOOTSTRAP_SAMPLES = 2000


def bootstrap_ci(values: Sequence[float], stat: Callable[[Sequence[float]], float],
                 seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of ``stat`` over per-ticket ``values``."""
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    n = len(values)
    samples = sorted(stat([values[rnd.randrange(n)] for _ in range(n)]) for _ in range(BOOTSTRAP_SAMPLES))
    return (round(samples[int(0.025 * BOOTSTRAP_SAMPLES)], 3), round(samples[int(0.975 * BOOTSTRAP_SAMPLES) - 1], 3))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def metric(values: list[float]) -> dict:
    return {"value": round(_mean(values), 3), "ci95": bootstrap_ci(values, _mean), "n": len(values)}


def load_dataset(name: str) -> tuple[list[dict], list[dict]]:
    if name == "heldout":
        synthetic_train, test = split(load_rows())
        extra = [r for r in training_rows() if r not in load_rows()]
        return synthetic_train + extra, test
    if name == "dev":
        return training_rows(), load_rows(DEV_DATA)
    if name == "test":
        verify_test_set()
        return training_rows(), load_rows(TEST_DATA)
    raise ValueError(f"unknown dataset {name}")


def make_orchestrator(backend: str | None, train: list[dict]):
    from .factory import build_orchestrator

    return build_orchestrator(backend or "offline", training_rows=train)


def score(test: list[dict], preds: list, kb_hits: list[float] | None = None) -> dict:
    """Per-ticket correctness vectors -> metrics with confidence intervals."""
    cat_ok = [float(r["category"] == p.category.value) for r, p in zip(test, preds)]
    pri_ok = [float(r["priority"] == p.priority.value) for r, p in zip(test, preds)]
    urgent = [float(p.escalate) for r, p in zip(test, preds) if r["escalate"]]
    flagged = [float(r["escalate"]) for r, p in zip(test, preds) if p.escalate]
    auto = [float(not p.needs_human_review) for p in preds]
    # Of the tickets that skip human review, how many were routed correctly?
    auto_ok = [float(r["category"] == p.category.value) for r, p in zip(test, preds) if not p.needs_human_review]
    guard = [float(p.guardrail.passed) for p in preds]
    hits = [float(any(a.id == r["kb"] for a in p.articles)) for r, p in zip(test, preds) if r.get("kb")]

    return {
        "routing_accuracy": metric(cat_ok),
        "routing_macro_f1": round(f1_score([r["category"] for r in test],
                                           [p.category.value for p in preds], average="macro"), 3),
        "priority_accuracy": metric(pri_ok),
        "escalation_recall": metric(urgent),
        "escalation_precision": metric(flagged),
        "auto_queued_rate": metric(auto),
        "auto_queued_routing_accuracy": metric(auto_ok),
        "retrieval_hit_at_2": metric(hits),
        "guardrail_pass_rate": metric(guard),
    }


def evaluate(backend: str | None = None, limit: int | None = None, dataset: str = "heldout") -> dict:
    train, test = load_dataset(dataset)
    if limit:
        test = test[:limit]
    orch = make_orchestrator(backend, train)

    preds, start = [], time.perf_counter()
    for row in test:
        preds.append(orch.triage(to_ticket(row)))
    elapsed = time.perf_counter() - start

    usage = orch.usage_summary() if hasattr(orch, "usage_summary") else {}
    results = {
        "backend": backend or "offline",
        "dataset": dataset,
        "train_size": len(train),
        "test_size": len(test),
        **score(test, preds),
        "avg_latency_ms": round(1000 * elapsed / len(preds), 1),
        "usage": usage,
        "cost_per_ticket_usd": round(usage.get("cost_usd", 0.0) / len(preds), 5),
        "errors": [
            {"id": r["id"], "true": r["category"], "pred": p.category.value,
             "true_priority": r["priority"], "pred_priority": p.priority.value,
             "escalate": r["escalate"], "pred_escalate": p.escalate, "body": r["body"][:140]}
            for r, p in zip(test, preds)
            if r["category"] != p.category.value or r["escalate"] != p.escalate
        ],
    }
    # Legacy flat keys used by the CI regression gate.
    results["category_accuracy"] = results["routing_accuracy"]["value"]
    results["escalation_recall_value"] = results["escalation_recall"]["value"]
    results["guardrail_pass_pct"] = round(100 * results["guardrail_pass_rate"]["value"], 1)

    out = ROOT / "reports" / f"eval_{results['backend']}_{dataset}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return results


def _fmt(m: dict) -> str:
    if not m["n"]:
        return "n/a (no labels)"
    lo, hi = m["ci95"]
    return f"{m['value']:.1%} ({lo:.0%}–{hi:.0%}, n={m['n']})"


def to_markdown(r: dict) -> str:
    lines = [
        f"| Metric | {r['dataset']} set · {r['backend']} · value (95% CI) |",
        "|---|---|",
        f"| Routing accuracy | {_fmt(r['routing_accuracy'])} |",
        f"| Priority accuracy | {_fmt(r['priority_accuracy'])} |",
        f"| Escalation recall | {_fmt(r['escalation_recall'])} |",
        f"| Escalation precision | {_fmt(r['escalation_precision'])} |",
        f"| Auto-queued (no human needed) | {_fmt(r['auto_queued_rate'])} |",
        f"| Routing accuracy of auto-queued | {_fmt(r['auto_queued_routing_accuracy'])} |",
        f"| Retrieval hit@2 | {_fmt(r['retrieval_hit_at_2'])} |",
        f"| Guardrail pass rate | {_fmt(r['guardrail_pass_rate'])} |",
        f"| Latency per ticket | {r['avg_latency_ms']} ms |",
    ]
    if r["usage"]:
        lines.append(f"| Cost per ticket | ${r['cost_per_ticket_usd']:.4f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None, help="offline | anthropic | openai | agent-sdk")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N tickets")
    ap.add_argument("--dataset", choices=["heldout", "dev", "test", "all"], default="all")
    args = ap.parse_args()
    for ds in (["heldout", "dev", "test"] if args.dataset == "all" else [args.dataset]):
        r = evaluate(args.backend, args.limit, ds)
        print(to_markdown(r))
        if r["errors"]:
            print(f"\n{len(r['errors'])} routing/escalation errors, e.g.:")
            for e in r["errors"][:6]:
                print(f"  {e['id']}: cat {e['true']}->{e['pred']} esc {e['escalate']}->{e['pred_escalate']} | {e['body']}")
        print()


if __name__ == "__main__":
    main()
