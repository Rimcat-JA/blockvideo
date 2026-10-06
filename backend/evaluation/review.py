"""Build an offline human-review page; opening it never grants approval."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.contracts import Case, ReviewLedger
from evaluation.corpus import eligibility, pending_ledger


def review_html(cases: list[Case], review: ReviewLedger | None = None) -> str:
    if review is not None:
        eligibility(cases, review, None)  # Strict full-corpus, role and case-hash check.
    payload = json.dumps({"cases": [c.model_dump(mode="json") for c in cases],
                          "ledger": (review or pending_ledger(cases)).model_dump(mode="json")}, ensure_ascii=True)
    # Data remains inert even if an example contains HTML, script end tags or JS.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    return template.replace("__D24_DATA__", payload)
