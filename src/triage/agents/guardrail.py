"""Guardrail agent: checks a draft reply before any human sees it.

Pure Python on purpose. Guardrails must be deterministic, fast, and testable,
so they don't call an LLM. They:

* redact sensitive data (card numbers, TFNs, phone numbers, emails)
* block promises the business hasn't authorised ("guarantee", "full refund today")
* block citations to articles that weren't retrieved (hallucinated sources)
* block other customers' names leaking into a reply
"""

from __future__ import annotations

import re

from ..models import DraftReply, GuardrailReport, KBArticle

PII_PATTERNS = {
    "card_number": r"\b(?:\d[ -]?){13,16}\b",
    "tfn": r"\b\d{3}[ -]?\d{3}[ -]?\d{3}\b",
    "au_phone": r"\b(?:\+?61|0)4\d{2}[ -]?\d{3}[ -]?\d{3}\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
}
ALLOWED_EMAILS = {"no-reply@cloudledger.example"}

FORBIDDEN_PROMISES = [
    r"\bguarantee[ds]?\b",
    r"\bwe will refund\b",
    r"\byou will (get|receive) a (full )?refund\b",
    r"\bfree (month|year)\b",
    r"\b(this|it) will never happen again\b",
]

CITATION = re.compile(r"\[(KB-\d{3})\]")


class GuardrailAgent:
    name = "guardrail"

    def __init__(self, known_customers: list[str] | None = None):
        self.known_customers = known_customers or []

    def redact(self, text: str) -> tuple[str, int]:
        count = 0
        for label, pattern in PII_PATTERNS.items():
            def _sub(m: re.Match, label: str = label) -> str:
                nonlocal count
                if label == "email" and m.group(0).lower() in ALLOWED_EMAILS:
                    return m.group(0)
                count += 1
                return f"[REDACTED {label.upper()}]"
            text = re.sub(pattern, _sub, text)
        return text, count

    def run(self, draft: DraftReply, articles: list[KBArticle],
            customer: str) -> tuple[DraftReply, GuardrailReport]:
        issues: list[str] = []
        text, redactions = self.redact(draft.text)

        for pattern in FORBIDDEN_PROMISES:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(f"unauthorised promise: /{pattern}/")

        retrieved = {a.id for a in articles}
        cited = set(CITATION.findall(text)) | set(draft.cited_articles)
        hallucinated = sorted(cited - retrieved)
        if hallucinated:
            issues.append(f"cites articles that were not retrieved: {hallucinated}")
        if articles and not cited:
            issues.append("reply is not grounded in any help article")

        for other in self.known_customers:
            if other != customer and other.lower() in text.lower():
                issues.append(f"mentions another customer: {other}")

        if len(text.split()) > 180:
            issues.append("reply too long (>180 words)")

        clean = DraftReply(text=text, cited_articles=sorted(cited & retrieved))
        return clean, GuardrailReport(passed=not issues, issues=issues, redactions=redactions)
