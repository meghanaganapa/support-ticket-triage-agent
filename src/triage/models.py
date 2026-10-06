"""Typed data models shared by every agent.

Using Pydantic means each agent's output is validated before the next agent
sees it. If an LLM returns a category that doesn't exist, it fails loudly here
instead of silently corrupting the pipeline.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    refund = "refund"
    shipping = "shipping"
    feature_request = "feature_request"


class Priority(str, Enum):
    P1 = "P1"  # urgent: outage, security, data exposure
    P2 = "P2"  # high: money or deadlines at risk, threats of complaint
    P3 = "P3"  # normal
    P4 = "P4"  # low: feature ideas, nice-to-haves


# Which team receives each category.
ROUTING = {
    Category.billing: "Billing Ops",
    Category.technical: "Tier-2 Engineering",
    Category.account: "Account Support",
    Category.refund: "Billing Ops",
    Category.shipping: "Hardware Logistics",
    Category.feature_request: "Product",
}

# First-response targets in minutes, by priority.
SLA_MINUTES = {Priority.P1: 15, Priority.P2: 60, Priority.P3: 480, Priority.P4: 2880}


class Ticket(BaseModel):
    id: str
    customer: str = "Unknown customer"
    plan: str = "Starter"
    subject: str = ""
    body: str


class Classification(BaseModel):
    category: Category
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class PriorityAssessment(BaseModel):
    priority: Priority
    escalate: bool
    signals: list[str] = Field(default_factory=list)


class KBArticle(BaseModel):
    id: str
    title: str
    text: str
    keywords: str = ""  # search-only synonyms; never shown to customers
    score: float = 0.0


class DraftReply(BaseModel):
    text: str
    cited_articles: list[str] = Field(default_factory=list)


class GuardrailReport(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    redactions: int = 0


class TriageResult(BaseModel):
    ticket_id: str
    category: Category
    category_confidence: float
    priority: Priority
    escalate: bool
    team: str
    sla_minutes: int
    signals: list[str]
    articles: list[KBArticle]
    draft_reply: DraftReply
    guardrail: GuardrailReport
    needs_human_review: bool
    review_reasons: list[str]
    trace: list[dict] = Field(default_factory=list)
