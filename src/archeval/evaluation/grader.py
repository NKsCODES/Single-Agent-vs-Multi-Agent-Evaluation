"""Deterministic grading against ground truth. No LLM judge, so grades are reproducible and free.

A task succeeds only if the answer is **correct and grounded**:

* correct: every requested fact for every product is stated with the right value, in that
  product's part of the answer (multihop must also name the right product; aggregate must name
  exactly the matching set);
* grounded: every citation points at a passage that was actually retrieved in the run and no
  fact-bearing sentence is uncited. A right value with no source is a lucky guess, and a
  production system should not get credit for it.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from archeval.agents.system import AnswerResult
from archeval.corpus.contoso import FACTS, PRODUCTS
from archeval.evaluation.tasks import EvalTask

PRODUCT_RE = re.compile("|".join(re.escape(p) for p in sorted(PRODUCTS, key=len, reverse=True)))


@dataclass
class Grade:
    task_id: str
    kind: str
    architecture: str
    correct: bool
    grounded: bool
    success: bool
    facts_total: int
    facts_correct: int
    wrong_values: int
    error: str | None = None

    @property
    def fact_recall(self) -> float:
        return self.facts_correct / self.facts_total if self.facts_total else 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"fact_recall": round(self.fact_recall, 4)}


def _value_pattern(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\d.,$])" + re.escape(value) + r"(?![\d%]|\.\d|,\d)", re.I)


def segments(answer: str) -> dict[str, str]:
    """Split an answer into the text that talks about each product."""
    marks = [(m.start(), m.group(0)) for m in PRODUCT_RE.finditer(answer)]
    out: dict[str, str] = {}
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(answer)
        out[name] = out.get(name, "") + " " + answer[start:end]
    return out


def has_fact(segment: str, product: str, attr: str) -> bool:
    truth = FACTS[product]
    if attr == "regions":
        return all(_value_pattern(r).search(segment) for r in truth.regions)
    return bool(_value_pattern(truth.value(attr)).search(segment))


def grade(task: EvalTask, result: AnswerResult) -> Grade:
    answer = result.answer or ""
    segs = segments(answer)
    total = correct_n = wrong = 0
    correct = True
    if task.kind == "aggregate":
        named = set(PRODUCT_RE.findall(answer))
        expected = set(task.expected_set or [])
        total = len(expected) or 1
        correct_n = len(named & expected) if expected else int(not named)
        wrong = len(named - expected)
        correct = named == expected and bool(answer)
    else:
        if task.kind == "multihop" and task.owner not in segs:
            correct = False
        for product in task.entities:
            seg = segs.get(product, "")
            for attr in task.attrs:
                total += 1
                if seg and has_fact(seg, product, attr):
                    correct_n += 1
                else:
                    correct = False
                    if seg and "not found" not in seg.lower():
                        wrong += 1
    grounded = result.grounded
    return Grade(task.id, task.kind, result.architecture, correct, grounded,
                 correct and grounded, total, correct_n, wrong, result.error)
