"""Load the labelled dataset and split it reproducibly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import Ticket

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "tickets.jsonl"


def load_rows(path: Path = DEFAULT_DATA) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


HARD_DATA = ROOT / "data" / "hard_tickets.jsonl"


def is_test(key: str, test_fraction: float = 0.3) -> bool:
    """Hash-based split: the same key always lands in the same split, whatever the row order."""
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return bucket < int(test_fraction * 100)


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Group split by template: no phrasing seen in training appears in the test set."""
    key = "template_id" if rows and "template_id" in rows[0] else "id"
    train = [r for r in rows if not is_test(r[key])]
    test = [r for r in rows if is_test(r[key])]
    return train, test


def to_ticket(row: dict) -> Ticket:
    return Ticket(**{k: row[k] for k in ("id", "customer", "plan", "subject", "body") if k in row})
