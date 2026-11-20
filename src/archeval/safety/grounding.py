"""Groundedness and privacy checks on answers.

``check_grounding`` verifies that every cited source id was actually retrieved during the run
and that every fact-bearing sentence carries a citation. ``redact`` removes personal data
(emails, phone numbers) before anything is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?!\d)")
FACT_SIGNALS = re.compile(r"\d|AES|region|India|Europe|US\b|Asia", re.I)


@dataclass
class GroundingReport:
    citations: list[str] = field(default_factory=list)
    unknown_citations: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        return not self.unknown_citations and not self.uncited_sentences


def check_grounding(answer: str, retrieved_ids: set[str]) -> GroundingReport:
    cites = CITATION.findall(answer)
    rep = GroundingReport(citations=sorted(set(cites)))
    rep.unknown_citations = sorted({c for c in cites if c not in retrieved_ids})
    for sentence in re.split(r"(?<=[.;])\s+|\n", answer):
        s = sentence.strip()
        if not s or "not found" in s.lower() or "could not" in s.lower() or s.startswith("No "):
            continue
        if FACT_SIGNALS.search(s) and not CITATION.search(s):
            rep.uncited_sentences.append(s[:200])
    return rep


def redact(text: str) -> str:
    return PHONE.sub("[phone]", EMAIL.sub("[email]", text))
