"""Command-line demo.

    triage "Our whole team is locked out, 503 errors everywhere"
    triage --file data/tickets.jsonl --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import Ticket
from .orchestrator import TriageOrchestrator


def render(result) -> str:
    flag = "🚨 ESCALATE" if result.escalate else ""
    review = "HUMAN REVIEW" if result.needs_human_review else "auto-queued"
    lines = [
        f"── {result.ticket_id} ─────────────────────────────",
        f"Route:    {result.category.value} → {result.team} (confidence {result.category_confidence:.2f})",
        f"Priority: {result.priority.value} · SLA {result.sla_minutes} min {flag}",
        f"Signals:  {', '.join(result.signals) or '-'}",
        f"KB:       {', '.join(f'{a.id} ({a.score})' for a in result.articles) or '-'}",
        f"Status:   {review}" + (f" - {'; '.join(result.review_reasons)}" if result.review_reasons else ""),
        "Draft reply:",
        *("  " + line for line in result.draft_reply.text.splitlines()),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Triage support tickets with a team of agents.")
    ap.add_argument("text", nargs="?", help="ticket text")
    ap.add_argument("--file", type=Path, help="JSONL file of tickets")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    orch = TriageOrchestrator()
    if args.file:
        with args.file.open() as f:
            tickets = [Ticket(**json.loads(line)) for line in list(f)[: args.limit]]
    elif args.text:
        tickets = [Ticket(id="CLI-1", body=args.text)]
    else:
        ap.error("give ticket text or --file")

    for t in tickets:
        result = orch.triage(t)
        print(result.model_dump_json(indent=2) if args.json else render(result))
        print()


if __name__ == "__main__":
    main()
