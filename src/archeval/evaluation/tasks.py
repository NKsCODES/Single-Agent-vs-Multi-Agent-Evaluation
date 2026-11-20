"""The benchmark: 60 questions over the Contoso knowledge base, with ground truth.

Four task types, 15 each, chosen to span the range from "one lookup" to "breadth-first":

* lookup     one product, one or two facts. A single agent should win on cost.
* multihop   find which product introduced a feature, then look up its facts. Sequential:
             the second step depends on the first, so parallel workers cannot help much.
* compare    three to eight products, two or three facts each. Independent lookups: the
             breadth-first case where orchestrator-worker is expected to pay off.
* aggregate  scan the whole catalog of 12 products and filter on two facts. The widest task.

Questions are generated from a fixed seed, so the benchmark is the same on every run.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from archeval.corpus.contoso import ATTRIBUTES, FACTS, PRODUCTS

TaskKind = Literal["lookup", "multihop", "compare", "aggregate"]
AGG_REGIONS = ["India Central", "India South", "West Europe", "North Europe", "Southeast Asia"]


@dataclass(frozen=True)
class EvalTask:
    id: str
    kind: TaskKind
    question: str
    entities: list[str] = field(default_factory=list)  # products the answer must cover
    attrs: list[str] = field(default_factory=list)
    owner: str | None = None  # multihop: the product that introduced the feature
    expected_set: list[str] | None = None  # aggregate: the products that match

    @property
    def breadth(self) -> int:
        if self.kind == "aggregate":
            return len(PRODUCTS) * 2
        return max(1, len(self.entities)) * max(1, len(self.attrs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"breadth": self.breadth}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _labels(attrs: list[str]) -> str:
    return _join([ATTRIBUTES[a] for a in attrs])


def build_tasks(seed: int = 2026, per_kind: int = 15) -> list[EvalTask]:
    rng = random.Random(seed)
    attr_keys = list(ATTRIBUTES)
    tasks: list[EvalTask] = []
    seen: set[str] = set()

    def add(task: EvalTask) -> bool:
        if task.question in seen:
            return False
        seen.add(task.question)
        tasks.append(task)
        return True

    i = 0
    while i < per_kind:
        product = rng.choice(PRODUCTS)
        attrs = rng.sample(attr_keys, 1 if i % 2 == 0 else 2)
        verb = "is" if len(attrs) == 1 else "are"
        i += add(EvalTask(f"lookup-{i:02d}", "lookup",
                          f"What {verb} the {_labels(attrs)} of {product}?", [product], attrs))

    i = 0
    while i < per_kind:
        owner = rng.choice(PRODUCTS)
        feature = rng.choice(FACTS[owner].releases)[2]
        attrs = rng.sample(attr_keys, 1 if i % 3 else 2)
        verb = "is" if len(attrs) == 1 else "are"
        i += add(EvalTask(f"multihop-{i:02d}", "multihop",
                          f"What {verb} the {_labels(attrs)} of the product that introduced "
                          f"{feature}?", [owner], attrs, owner=owner))

    for i in range(per_kind):
        n = 3 + (i % 6)  # 3..8 products
        products = rng.sample(PRODUCTS, n)
        attrs = rng.sample(attr_keys, 2 if i < per_kind // 2 else 3)
        tasks.append(EvalTask(f"compare-{i:02d}", "compare",
                              f"Compare the {_labels(attrs)} of {_join(products)}.",
                              products, attrs))

    combos = [(s, r) for s in ("99.9%", "99.95%", "99.99%") for r in AGG_REGIONS]
    rng.shuffle(combos)
    for i, (sla, region) in enumerate(combos[:per_kind]):
        expected = sorted(p for p, f in FACTS.items() if f.sla == sla and region in f.regions)
        tasks.append(EvalTask(f"aggregate-{i:02d}", "aggregate",
                              f"Which Contoso products have an uptime SLA of {sla} and store "
                              f"data in {region}?", [], ["sla", "regions"],
                              expected_set=expected))
    return tasks


TASKS = build_tasks()
