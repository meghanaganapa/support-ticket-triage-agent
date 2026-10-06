"""Responder agent: drafts a reply grounded in the retrieved help articles.

The draft is never sent automatically. It goes to the guardrail agent and then
to a human agent's queue, pre-filled so they can edit and send in seconds.
"""

from __future__ import annotations

from ..llm import LLM
from ..models import DraftReply, KBArticle, PriorityAssessment, Ticket

SYSTEM = """You draft replies for CloudLedger support. Rules:
- Use ONLY facts from the provided help articles. Cite article IDs like [KB-102].
- Never promise refunds, credits, or timelines that the articles don't state.
- If the ticket is escalated, acknowledge urgency and say a specialist is on it.
- Friendly, plain Australian English, under 120 words. No sign-off name.
Return JSON: {"text": str, "cited_articles": [str]}"""

ESCALATION_OPENER = ("Thanks for letting us know - we've flagged this as urgent and a "
                     "specialist from our {team} team is looking at it now.")


class ResponderAgent:
    name = "responder"

    def __init__(self, llm: LLM):
        self.llm = llm

    def template_reply(self, ticket: Ticket, articles: list[KBArticle],
                       assessment: PriorityAssessment, team: str) -> DraftReply:
        lines = ["Hi there,", ""]
        if assessment.escalate:
            lines.append(ESCALATION_OPENER.format(team=team))
        else:
            lines.append("Thanks for getting in touch.")
        if articles:
            top = articles[0]
            # First two sentences of the best article keep the reply short and grounded.
            summary = ". ".join(top.text.split(". ")[:2]).rstrip(".") + "."
            lines.append(f"{summary} [{top.id}]")
        lines += ["", "We'll follow up shortly with next steps."]
        return DraftReply(text="\n".join(lines), cited_articles=[a.id for a in articles[:1]])

    def run(self, ticket: Ticket, articles: list[KBArticle],
            assessment: PriorityAssessment, team: str) -> DraftReply:
        context = "\n\n".join(f"[{a.id}] {a.title}: {a.text}" for a in articles) or "No articles found."
        user = (f"Help articles:\n{context}\n\nEscalated: {assessment.escalate} (team: {team})\n\n"
                f"Ticket from {ticket.customer}:\n{ticket.subject}\n{ticket.body}")
        reply = self.llm.complete_json(SYSTEM, user)
        if not reply or "text" not in reply:
            return self.template_reply(ticket, articles, assessment, team)
        return DraftReply(text=str(reply["text"]),
                          cited_articles=[str(c) for c in reply.get("cited_articles", [])])
