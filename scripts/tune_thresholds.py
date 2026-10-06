"""Choose the review-gate thresholds on dev + held-out data (never the locked test set).

Business rules, in order:
  1. Escalation: maximise F2 on the escalate flag (recall weighted twice as much
     as precision); break ties on priority accuracy. Pure recall was tried first
     and chose "escalate almost everything" (precision 14%), so it was rejected.
  2. Review gate: maximise the share of tickets that skip human review, subject
     to those tickets being routed correctly at least TARGET_AUTO_ACCURACY of the time.

Writes config/thresholds.json, which the orchestrator loads at start-up.

    python scripts/tune_thresholds.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triage.data import ROOT, to_ticket
from triage.evaluate import load_dataset
from triage.llm import OfflineLLM
from triage.orchestrator import TriageOrchestrator

TARGET_AUTO_ACCURACY = 0.95
ESCALATION_GRID = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
PRIORITY_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]
CONFIDENCE_GRID = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]


def run(orchs, esc, prio, conf):
    out = []
    for orch, rows in orchs:
        orch.thresholds["confidence"] = conf
        orch.priority.escalation_threshold = esc
        orch.priority.priority_threshold = prio
        out += [(r, orch.triage(to_ticket(r))) for r in rows]
    return out


def main() -> None:
    orchs = []
    for name in ("dev", "heldout"):
        train, rows = load_dataset(name)
        orchs.append((TriageOrchestrator(llm=OfflineLLM(), training_rows=train), rows))

    # Stage 1: escalation + priority thresholds.
    best = None
    for esc, prio in itertools.product(ESCALATION_GRID, PRIORITY_GRID):
        res = run(orchs, esc, prio, 0.55)
        urgent = [p.escalate for r, p in res if r["escalate"]]
        flagged = [r["escalate"] for r, p in res if p.escalate]
        recall = sum(urgent) / len(urgent)
        precision = sum(flagged) / max(1, len(flagged))
        pri_acc = sum(r["priority"] == p.priority.value for r, p in res) / len(res)
        f2 = 5 * precision * recall / max(1e-9, 4 * precision + recall)
        key = (round(f2, 4), round(pri_acc, 4))
        print(f"  escalation {esc:.2f} priority {prio:.1f}: F2 {f2:.3f} recall {recall:.3f} "
              f"precision {precision:.3f} priority acc {pri_acc:.3f}")
        if best is None or key > best[0]:
            best = (key, esc, prio, recall, precision)
    (f2, pri_acc), esc, prio, recall, precision = best
    print(f"stage 1: escalation={esc} priority={prio} -> F2 {f2:.3f}, recall {recall:.3f}, "
          f"precision {precision:.3f}, priority acc {pri_acc:.3f}")

    # Stage 2: confidence threshold for the review gate.
    chosen = None
    for conf in CONFIDENCE_GRID:
        res = run(orchs, esc, prio, conf)
        auto = [(r, p) for r, p in res if not p.needs_human_review]
        acc = sum(r["category"] == p.category.value for r, p in auto) / max(1, len(auto))
        rate = len(auto) / len(res)
        print(f"  confidence {conf:.2f}: auto-queued {rate:.1%}, routed correctly {acc:.1%}")
        if acc >= TARGET_AUTO_ACCURACY and (chosen is None or rate > chosen[1]):
            chosen = (conf, rate, acc)
    if chosen is None:
        chosen = (CONFIDENCE_GRID[-1], None, None)
    print(f"stage 2: confidence={chosen[0]}")

    out = ROOT / "config" / "thresholds.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "thresholds": {"confidence": chosen[0], "escalation": esc, "priority": prio},
        "tuned_on": ["dev", "heldout"],
        "objective": {
            "stage1": "max escalation F2 (recall weighted 2x), then priority accuracy",
            "stage2": f"max auto-queued share with auto-queued routing accuracy >= {TARGET_AUTO_ACCURACY}",
        },
        "tuning_metrics": {"escalation_f2": f2, "escalation_recall": recall, "priority_accuracy": pri_acc,
                           "escalation_precision": precision,
                           "auto_queued_rate": chosen[1], "auto_queued_routing_accuracy": chosen[2]},
    }, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
