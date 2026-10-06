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


DEV_DATA = ROOT / "data" / "dev_tickets.jsonl"        # 30 hand-written; used for tuning thresholds
TEST_DATA = ROOT / "data" / "test_locked.jsonl"       # 90 hand-written; never used for tuning
TEST_SHA = ROOT / "data" / "test_locked.sha256"
EXTRA_TRAIN = ROOT / "data" / "train_varied.jsonl"    # hand-written varied phrasings for training


def training_rows() -> list[dict]:
    """Everything the models may learn from: synthetic tickets + varied hand-written ones."""
    rows = load_rows()
    if EXTRA_TRAIN.exists():
        rows += load_rows(EXTRA_TRAIN)
    return rows


def verify_test_set() -> None:
    """Refuse to evaluate if the locked test set was edited after it was committed."""
    import hashlib

    expected = TEST_SHA.read_text().split()[0]
    actual = hashlib.sha256(TEST_DATA.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("data/test_locked.jsonl changed since it was locked - results would not be comparable")


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
