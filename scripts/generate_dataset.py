"""Generate a labelled synthetic support-ticket dataset.

Real support data is private, so this script builds a realistic stand-in for a
SaaS company ("CloudLedger", an invoicing and accounting app). Every ticket is
labelled with:

* category   - which team owns it
* priority   - P1 (urgent) .. P4 (low)
* escalate   - must a human lead see it immediately (security, legal, outage)?

Tickets are built from templates plus random noise (greetings, typos, sign-offs,
filler sentences, mixed signals) so a classifier cannot simply memorise strings.
The output is deterministic for a given seed, so evaluation results reproduce.

Usage:
    python scripts/generate_dataset.py --n 400 --seed 7 --out data/tickets.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CUSTOMERS = ["Acme Pty Ltd", "Bluegum Cafe", "Northside Physio", "Kestrel Logistics",
             "Harbour Dental", "Wattle Consulting", "Redgum Builders", "Lumen Studio",
             "Coastal Plumbing", "Brightpath Tutoring", "Ironbark Legal", "Saltwater Surf Co"]

PLANS = ["Starter", "Growth", "Business", "Enterprise"]

GREETINGS = ["Hi team,", "Hello,", "Hi there", "Good morning,", "Hey support,", "To whom it may concern,", "", "", "Hi,"]
SIGNOFFS = ["Thanks", "Cheers", "Regards", "Thanks in advance", "Kind regards", "", "", "Ta"]
FILLER = [
    "We've been using CloudLedger for about two years.",
    "Not sure if this is the right place to ask.",
    "I checked the help centre but couldn't find anything.",
    "This is the second time I'm writing about this.",
    "Our accountant flagged this today.",
    "Sorry if this has been asked before.",
    "",
    "",
    "",
]

# (category, priority, escalate, templates). {x} placeholders are filled below.
TEMPLATES: list[tuple[str, str, bool, list[str]]] = [
    # ---------------- billing ----------------
    ("billing", "P3", False, [
        "I was charged {amount} this month but our {plan} plan should be cheaper. Can you check the invoice?",
        "Our card was billed twice for the {month} subscription. Please look into the duplicate charge.",
        "How do I change the credit card on file for our subscription?",
        "Can I get a tax invoice with our ABN on it for the last payment?",
        "We want to switch from monthly to annual billing. What does that cost?",
        "The invoice for {month} shows {seats} seats but we only have {seats_small} users.",
        "Is there a discount for non-profits on the {plan} plan?",
    ]),
    ("billing", "P2", False, [
        "Our payment failed and now the account says it will be suspended in 2 days. Please help before we lose access.",
        "We've been charged {amount} three times today. That's money out of a small business account, please fix urgently.",
    ]),
    # ---------------- technical ----------------
    ("technical", "P3", False, [
        "The PDF export of invoices is cutting off the last line item.",
        "Bank feed for {bank} stopped syncing since {day}. Transactions after that are missing.",
        "The mobile app crashes when I try to attach a receipt photo.",
        "Recurring invoices didn't send this morning for some clients.",
        "Getting 'Error 500' when I open the reports page in Chrome.",
        "The Xero import tool hangs at 80% and never finishes.",
        "Dashboard charts are blank since the last update.",
        "Our API integration gets 429 rate limit errors when syncing {seats} invoices.",
    ]),
    ("technical", "P1", True, [
        "Nobody in our company can log in - the whole app is down with a 503 error. We can't invoice anyone today.",
        "CloudLedger is completely down for us, all {seats} staff locked out. This is costing us money every hour.",
        "Your API is returning errors for every request since {time}, our checkout is broken in production.",
    ]),
    ("technical", "P2", False, [
        "Invoices are being sent with the wrong GST amount calculated. Customers are complaining.",
        "Payroll export is producing wrong totals and payroll runs tomorrow.",
    ]),
    # ---------------- account ----------------
    ("account", "P3", False, [
        "How do I add a new user to our account?",
        "I need to change the account owner to {name} because I'm leaving the company.",
        "Can you reset two-factor authentication for {name}? They lost their phone.",
        "How do I change our company name and logo on invoices?",
        "Please remove {name} from our account, they no longer work here.",
        "I forgot my password and the reset email never arrives.",
    ]),
    ("account", "P1", True, [
        "Someone logged into our account from overseas and changed the bank details on our invoices. I think we've been hacked.",
        "We got an alert of a login we don't recognise and {name}'s password was changed without them doing it. Possible breach.",
        "I think our data has been exposed - a customer received another company's invoice from your system.",
    ]),
    # ---------------- refund ----------------
    ("refund", "P3", False, [
        "We cancelled last week but were still charged {amount}. Can we get a refund?",
        "I'd like a refund for the annual plan, we only used it for {days} days.",
        "We downgraded from {plan} but were billed the old price. Please refund the difference.",
    ]),
    ("refund", "P2", True, [
        "If the {amount} isn't refunded by Friday I'll be lodging a complaint with the ACCC and talking to my lawyer.",
        "This is unacceptable. Refund the {amount} or we will take legal action and post about this publicly.",
    ]),
    # ---------------- shipping (hardware card readers) ----------------
    ("shipping", "P3", False, [
        "We ordered a card reader {days} days ago and it still hasn't arrived. Tracking hasn't updated.",
        "The card reader arrived damaged, the screen is cracked. How do I get a replacement?",
        "Can I change the delivery address for order {order}?",
        "We received two card readers but only ordered one.",
    ]),
    # ---------------- feature request ----------------
    ("feature_request", "P4", False, [
        "It would be great if we could schedule invoices to send at a specific time.",
        "Any plans to support multi-currency invoices? We bill some clients in NZD.",
        "Please add a dark mode to the web app, would really help.",
        "Could you add an integration with {bank} for direct debits?",
        "Feature idea: let us customise the email template for payment reminders.",
        "Would love to be able to export reports to Google Sheets directly.",
    ]),
]

FILLS = {
    "amount": lambda r: f"${r.choice([29, 49, 79, 129, 249, 499, 1188])}",
    "plan": lambda r: r.choice(PLANS),
    "month": lambda r: r.choice(["July", "August", "September", "October"]),
    "seats": lambda r: str(r.choice([12, 15, 25, 40, 120])),
    "seats_small": lambda r: str(r.choice([3, 5, 8, 10])),
    "bank": lambda r: r.choice(["CommBank", "Westpac", "ANZ", "NAB", "Stripe"]),
    "day": lambda r: r.choice(["Monday", "the weekend", "yesterday", "the 3rd"]),
    "time": lambda r: r.choice(["9am", "about an hour ago", "11:40 AEST", "this morning"]),
    "name": lambda r: r.choice(["Priya", "Tom", "Mei", "Jack", "Aisha", "Liam"]),
    "days": lambda r: str(r.choice([4, 9, 12, 20])),
    "order": lambda r: f"#{r.randint(10000, 99999)}",
}


def _fill(template: str, rnd: random.Random) -> str:
    out = template
    for key, fn in FILLS.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, fn(rnd), 1)
    return out


def _typo(text: str, rnd: random.Random) -> str:
    """Introduce a light typo in ~20% of tickets (swap two adjacent letters)."""
    if rnd.random() > 0.2 or len(text) < 20:
        return text
    i = rnd.randint(5, len(text) - 3)
    if text[i].isalpha() and text[i + 1].isalpha():
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text


def make_ticket(idx: int, rnd: random.Random) -> dict:
    # Weight categories roughly like a real queue: technical + billing dominate.
    weights = [len(t[3]) * (3 if t[1] in ("P3", "P4") else 1) for t in TEMPLATES]
    group = rnd.choices(range(len(TEMPLATES)), weights=weights, k=1)[0]
    category, priority, escalate, templates = TEMPLATES[group]
    t_idx = rnd.randrange(len(templates))
    body = _typo(_fill(templates[t_idx], rnd), rnd)
    parts = [rnd.choice(GREETINGS), rnd.choice(FILLER), body, rnd.choice(SIGNOFFS)]
    text = " ".join(p for p in parts if p).strip()
    subject_words = body.split()[: rnd.randint(4, 7)]
    return {
        "id": f"T{idx:04d}",
        "customer": rnd.choice(CUSTOMERS),
        "plan": rnd.choice(PLANS),
        "subject": " ".join(subject_words).rstrip(".,!?"),
        "body": text,
        "category": category,
        "priority": priority,
        "escalate": escalate,
        # Which template produced it. The train/test split holds out whole templates,
        # so evaluation measures generalisation to unseen phrasings, not memorisation.
        "template_id": f"{group}-{t_idx}",
    }


def generate(n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    return [make_ticket(i + 1, rnd) for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/tickets.jsonl"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(args.n, args.seed)
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} tickets to {args.out}")


if __name__ == "__main__":
    main()
