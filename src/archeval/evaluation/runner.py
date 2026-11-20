"""Run the benchmark: every task on every architecture, optionally several seeds.

Runs are concurrent (bounded by a semaphore) and isolated: each task gets its own usage ledger
through a context variable, so concurrent runs never mix their token counts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from archeval.agents.system import ARCHITECTURES, AgentSystem, AnswerResult
from archeval.config import Settings
from archeval.corpus.contoso import DOCUMENTS
from archeval.evaluation.grader import Grade, grade
from archeval.evaluation.tasks import TASKS, EvalTask
from archeval.llm.clients import ClientFactory
from archeval.llm.sim import SimConfig
from archeval.retrieval.embeddings import HashingEmbedder
from archeval.retrieval.store import HybridStore, InMemoryHybridStore

DEFAULT_ARCHS = ["single-small", "single-large", "multi-large", "multi-mixed", "routed"]


@dataclass
class RunRecord:
    task: EvalTask
    result: AnswerResult
    grade: Grade
    repeat: int

    def to_dict(self) -> dict[str, Any]:
        r = self.result.to_dict()
        r.pop("ledger", None)
        return {"repeat": self.repeat, "task": self.task.to_dict(), "result": r,
                "grade": self.grade.to_dict()}


@dataclass
class BenchmarkRun:
    records: list[RunRecord] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


async def default_store(settings: Settings) -> HybridStore:
    store = InMemoryHybridStore(HashingEmbedder(settings.embedding_dimensions))
    await store.upsert(DOCUMENTS)
    return store


async def run_benchmark(settings: Settings, archs: Sequence[str] = DEFAULT_ARCHS,
                        tasks: Sequence[EvalTask] = TASKS, repeats: int = 1,
                        concurrency: int = 16, store: HybridStore | None = None,
                        sim_config: SimConfig | None = None) -> BenchmarkRun:
    store = store or await default_store(settings)
    sem = asyncio.Semaphore(concurrency)
    base = sim_config or SimConfig()
    started = time.perf_counter()
    records: list[RunRecord] = []

    async def one(system: AgentSystem, task: EvalTask, arch: str, rep: int) -> None:
        async with sem:
            result = await system.run(task.question, ARCHITECTURES[arch])
        records.append(RunRecord(task, result, grade(task, result), rep))

    jobs = []
    for rep in range(repeats):
        cfg = replace(base, seed=f"{base.seed}-r{rep}")
        system = AgentSystem(settings, store, ClientFactory(settings, cfg))
        jobs += [one(system, t, a, rep) for t in tasks for a in archs]
    await asyncio.gather(*jobs)
    records.sort(key=lambda r: (r.repeat, r.task.id, list(archs).index(r.result.architecture)))
    return BenchmarkRun(records, {"provider": settings.llm_provider, "repeats": repeats,
                                  "limits": settings.limits.model_dump(),
                                  "prices": {k: v.model_dump() for k, v in
                                             settings.prices.items()},
                                  "sim": None if settings.llm_provider != "sim" else
                                  {"seed": base.seed, "knee_tokens": base.knee_tokens,
                                   "recall_slope_per_1k": base.recall_slope_per_1k,
                                   "base_recall": base.base_recall,
                                   "hallucination_rate": base.hallucination_rate}},
                        round(time.perf_counter() - started, 2))
