"""Classifier agent: decides which team owns the ticket.

Two strategies, used together:

1. An LLM reads the ticket and returns a category with a confidence score.
2. A TF-IDF + logistic regression model trained on labelled history.

The ML model is always trained. It is the offline fallback, and when an LLM is
active it acts as a cross-check: if the two disagree, confidence is lowered,
which routes the ticket to human review instead of guessing.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ..llm import LLM
from ..models import Category, Classification, Ticket

SYSTEM = """You are the routing agent for CloudLedger customer support.
Classify the ticket into exactly one category:
- billing: charges, invoices for the subscription, payment methods, plan pricing
- technical: bugs, errors, outages, sync, exports, API, wrong calculations
- account: users, logins, passwords, 2FA, ownership, security or hacked accounts
- refund: the customer explicitly wants money back
- shipping: card reader hardware orders, delivery, damaged devices
- feature_request: suggestions for new functionality
Return JSON: {"category": str, "confidence": float 0-1, "rationale": str (one sentence)}"""


def ticket_text(t: Ticket) -> str:
    return f"{t.subject}. {t.body}"


def build_text_model(class_weight: str | None = "balanced", calibrated: bool = True) -> Pipeline:
    """Word + character n-gram TF-IDF into (calibrated) logistic regression.

    Character n-grams (3-5 chars within word boundaries) make the model robust to
    typos ("cilents") and word variants ("refunded" / "refund") it never saw in training.

    Calibration (Platt scaling, 3-fold) makes ``predict_proba`` mean something:
    a ticket scored 0.8 is right about 80% of the time. Without it, the review
    threshold is a magic number; with it, the threshold is a business decision.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight=class_weight)
    if calibrated:
        clf = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    return Pipeline([("features", features), ("clf", clf)])


class ClassifierAgent:
    name = "classifier"

    def __init__(self, llm: LLM):
        self.llm = llm
        self.model: Pipeline | None = None

    def fit(self, tickets: list[Ticket], labels: list[str]) -> ClassifierAgent:
        self.model = build_text_model()
        self.model.fit([ticket_text(t) for t in tickets], labels)
        return self

    def _ml_predict(self, ticket: Ticket) -> Classification:
        if self.model is None:
            raise RuntimeError("ClassifierAgent.fit() must be called before predicting")
        probs = self.model.predict_proba([ticket_text(ticket)])[0]
        best = int(probs.argmax())
        label = self.model.classes_[best]
        return Classification(category=Category(label), confidence=float(probs[best]),
                              rationale="TF-IDF + logistic regression baseline")

    def run(self, ticket: Ticket) -> Classification:
        ml = self._ml_predict(ticket)
        reply = self.llm.complete_json(SYSTEM, ticket_text(ticket))
        if not reply:
            return ml
        try:
            llm_result = Classification(
                category=Category(reply["category"]),
                confidence=float(reply.get("confidence", 0.7)),
                rationale=str(reply.get("rationale", "")),
            )
        except (KeyError, ValueError):
            return ml  # malformed LLM output: fall back rather than crash
        if llm_result.category != ml.category:
            # Disagreement between two independent methods is a useful uncertainty signal.
            llm_result.confidence = min(llm_result.confidence, 0.5)
            llm_result.rationale += f" (ML baseline suggested {ml.category.value})"
        return llm_result
