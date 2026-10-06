"""Knowledge agent: finds the help-centre articles relevant to a ticket.

Retrieval is TF-IDF cosine similarity over help-centre sections, boosted toward
articles in the ticket's category (KB-1xx billing, 2xx technical, ...). This is
deliberately simple and explainable; swapping in embeddings + a vector store
(e.g. Azure AI Search) only means replacing ``search``.
"""

from __future__ import annotations

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import Category, KBArticle, Ticket

CATEGORY_PREFIX = {
    Category.billing: "KB-1", Category.refund: "KB-1", Category.technical: "KB-2",
    Category.account: "KB-3", Category.shipping: "KB-4", Category.feature_request: "KB-5",
}

DEFAULT_KB = Path(__file__).resolve().parents[3] / "kb" / "help_centre.md"


def load_articles(path: Path = DEFAULT_KB) -> list[KBArticle]:
    text = path.read_text()
    articles = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        header, _, body = block.partition("\n")
        art_id, _, title = header.strip().partition(" ")
        keywords = ""
        lines = body.strip().splitlines()
        if lines and lines[0].startswith("Keywords:"):
            keywords = lines.pop(0).removeprefix("Keywords:").strip().rstrip(".")
        articles.append(KBArticle(id=art_id, title=title.strip(), text="\n".join(lines).strip(),
                                  keywords=keywords))
    return articles


class KnowledgeAgent:
    name = "knowledge"

    def __init__(self, articles: list[KBArticle] | None = None, category_boost: float = 0.15):
        self.articles = articles or load_articles()
        self.boost = category_boost
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        # Keywords are indexed (repeated to weight them) but never returned in article text.
        self.matrix = self.vectorizer.fit_transform(
            [f"{a.title} {a.keywords} {a.keywords} {a.text}" for a in self.articles])

    def search(self, query: str, category: Category | None = None, k: int = 2) -> list[KBArticle]:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        scored = []
        for art, sim in zip(self.articles, sims):
            score = float(sim)
            if category and art.id.startswith(CATEGORY_PREFIX[category]):
                score += self.boost
            scored.append(art.model_copy(update={"score": round(score, 3)}))
        scored.sort(key=lambda a: a.score, reverse=True)
        return [a for a in scored[:k] if a.score > 0.05]

    def run(self, ticket: Ticket, category: Category) -> list[KBArticle]:
        return self.search(f"{ticket.subject} {ticket.body}", category)
