"""Complexity router: send each question to the cheapest architecture likely to answer it.

Breadth is what makes orchestrator-worker pay off: independent lookups that can run in parallel,
each in a fresh context. The router estimates breadth from the question itself (products named
times distinct facts asked, or a catalog-wide scan), with no model call, so routing adds no
cost or latency. The threshold is a tunable that the benchmark reports on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

PRODUCT = re.compile(r"\bContoso (?:[A-Z][A-Za-z]+)\b")
CATALOG_SCAN = re.compile(r"\b(which|list|all|every|each)\b[^?]*\b(products|services)\b", re.I)
CONCEPTS = {
    "sla": ("uptime", "sla"),
    "regions": ("region", "residency", "store data"),
    "price": ("price", "pricing", "cost"),
    "users": ("users", "seats"),
    "encryption": ("encryption", "encrypted"),
    "support": ("p1", "response time", "support"),
}


@dataclass(frozen=True)
class Route:
    architecture: Literal["single", "multi"]
    breadth: int
    reason: str


def concepts(question: str) -> set[str]:
    low = question.lower()
    return {k for k, words in CONCEPTS.items() if any(w in low for w in words)}


def route(question: str, threshold: int = 4) -> Route:
    if CATALOG_SCAN.search(question):
        return Route("multi", 99, "catalog-wide scan")
    products = len(set(PRODUCT.findall(question)))
    facts = max(1, len(concepts(question)))
    breadth = max(1, products) * facts
    reason = f"{products} product(s) x {facts} fact(s) = breadth {breadth}"
    return Route("multi" if breadth >= threshold else "single", breadth, reason)
