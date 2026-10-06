"""Priority agent: how urgent is it, and must a human lead see it now?

Escalation is safety-critical, so it never depends on the LLM alone. Explicit
rules catch outages, security incidents and legal threats. The LLM can raise
priority (it reads nuance the rules miss) but can never lower a rule-detected
P1 or remove an escalation.
"""

from __future__ import annotations

import re

from ..llm import LLM
from ..models import Category, Priority, PriorityAssessment, Ticket

# (signal name, regex, priority it implies, escalate?)
RULES: list[tuple[str, str, Priority, bool]] = [
    ("outage", r"\b(down|503|outage|can'?t log in|locked out|every request|broken in production)\b", Priority.P1, True),
    ("security", r"\b(hack(ed)?|breach|unauthori[sz]ed|don'?t recognise|don'?t recognize|without them|bank details)\b", Priority.P1, True),
    ("data_exposure", r"(another company'?s|data has been exposed|exposed)", Priority.P1, True),
    ("legal_threat", r"\b(lawyer|legal action|accc|ombudsman|complaint with|post about this publicly)\b", Priority.P2, True),
    ("access_at_risk", r"\b(suspended|lose access)\b", Priority.P2, False),
    ("money_at_risk", r"\b(three times|charged .* times|costing us money|wrong gst|wrong totals)\b", Priority.P2, False),
    ("deadline", r"\b(urgent(ly)?|asap|tomorrow|today)\b", Priority.P3, False),
]

SYSTEM = """You assess urgency for CloudLedger support tickets.
P1 = service outage, security incident, data exposure. P2 = money or access at risk,
legal/regulator threats, deadline within a day. P3 = normal issue. P4 = feature idea.
Return JSON: {"priority": "P1"|"P2"|"P3"|"P4", "escalate": bool, "signals": [str]}"""

ORDER = [Priority.P1, Priority.P2, Priority.P3, Priority.P4]


def _higher(a: Priority, b: Priority) -> Priority:
    return a if ORDER.index(a) <= ORDER.index(b) else b


class PriorityAgent:
    """Combines three independent signals and takes the most urgent:

    1. hand-written rules (precise, explainable, brittle to new wording)
    2. a learned escalation model trained on labelled history (generalises better)
    3. the LLM, when one is configured (reads nuance and context)
    """

    name = "priority"
    ESCALATION_THRESHOLD = 0.35  # recall matters more than precision: a missed outage costs far more

    def __init__(self, llm: LLM):
        self.llm = llm
        self.escalation_model = None

    def fit(self, tickets: list[Ticket], escalate: list[bool]) -> PriorityAgent:
        from .classifier import build_text_model, ticket_text

        if len(set(escalate)) > 1:
            self.escalation_model = build_text_model()
            self.escalation_model.fit([ticket_text(t) for t in tickets], escalate)
        return self

    def escalation_probability(self, ticket: Ticket) -> float | None:
        if self.escalation_model is None:
            return None
        from .classifier import ticket_text

        classes = list(self.escalation_model.classes_)
        return float(self.escalation_model.predict_proba([ticket_text(ticket)])[0][classes.index(True)])

    def assess(self, ticket: Ticket, category: Category) -> PriorityAssessment:
        """Rules + learned model, no LLM."""
        base = self.rules(ticket, category)
        prob = self.escalation_probability(ticket)
        if prob is not None and prob >= self.ESCALATION_THRESHOLD and not base.escalate:
            # Security/data issues are P1; legal and refund threats are P2.
            implied = Priority.P1 if category in (Category.technical, Category.account) else Priority.P2
            return PriorityAssessment(priority=_higher(base.priority, implied), escalate=True,
                                      signals=base.signals + [f"ml_escalation:{prob:.2f}"])
        return base

    def rules(self, ticket: Ticket, category: Category) -> PriorityAssessment:
        text = f"{ticket.subject} {ticket.body}".lower()
        priority = Priority.P4 if category == Category.feature_request else Priority.P3
        escalate = False
        signals: list[str] = []
        for name, pattern, prio, esc in RULES:
            if re.search(pattern, text):
                # A deadline word in a feature request doesn't make it urgent.
                if category == Category.feature_request and name == "deadline":
                    continue
                signals.append(name)
                priority = _higher(priority, prio)
                escalate = escalate or esc
        return PriorityAssessment(priority=priority, escalate=escalate, signals=signals)

    def run(self, ticket: Ticket, category: Category) -> PriorityAssessment:
        base = self.assess(ticket, category)
        reply = self.llm.complete_json(SYSTEM, f"Category: {category.value}\n{ticket.subject}\n{ticket.body}")
        if not reply:
            return base
        try:
            llm_priority = Priority(reply["priority"])
        except (KeyError, ValueError):
            return base
        return PriorityAssessment(
            priority=_higher(base.priority, llm_priority),  # LLM may raise, never lower
            escalate=base.escalate or bool(reply.get("escalate", False)),
            signals=sorted(set(base.signals) | {f"llm:{s}" for s in reply.get("signals", [])}),
        )
