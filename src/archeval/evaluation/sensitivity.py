"""Which assumptions drive the result? Re-run the breadth tasks while changing one at a time.

* step budget of the single agent (max tool-loop iterations): is single-agent failure just a
  budget artefact?
* context degradation slope: how much of the gap comes from long contexts losing facts?
* structured hand-off: if worker findings were no easier to read than raw passages, how much
  of the multi-agent advantage would remain?

Only meaningful with the simulator, where these are explicit parameters. Against a real model,
the budget sweep still applies and the other two are properties of the model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from archeval.config import Settings
from archeval.evaluation.economics import pick, summarize
from archeval.evaluation.runner import default_store, run_benchmark
from archeval.evaluation.tasks import TASKS
from archeval.llm.sim import SimConfig

BREADTH_TASKS = [t for t in TASKS if t.kind in ("compare", "aggregate")]
ARCHS = ["single-large", "multi-mixed"]


@dataclass
class SweepPoint:
    factor: str
    value: float
    single_success: float
    multi_success: float
    single_cost: float
    multi_cost: float
    single_latency_s: float
    multi_latency_s: float

    @property
    def gap(self) -> float:
        return round(self.multi_success - self.single_success, 4)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__ | {"gap": self.gap}


async def _point(settings: Settings, cfg: SimConfig, factor: str, value: float,
                 repeats: int) -> SweepPoint:
    store = await default_store(settings)
    run = await run_benchmark(settings, ARCHS, BREADTH_TASKS, repeats, store=store,
                              sim_config=cfg)
    sums = summarize(run.records)
    s, m = pick(sums, "single-large"), pick(sums, "multi-mixed")
    return SweepPoint(factor, value, s.success_rate, m.success_rate, s.cost_per_task,
                      m.cost_per_task, s.latency_p50_s, m.latency_p50_s)


async def sweep(settings: Settings, repeats: int = 2) -> list[SweepPoint]:
    base = SimConfig()
    points = []
    for budget in (4, 6, 10, 16, 24):
        limits = settings.limits.model_copy(update={"single_max_iterations": budget})
        points.append(await _point(settings.model_copy(update={"limits": limits}), base,
                                   "single_step_budget", budget, repeats))
    for mult in (0.0, 0.5, 1.0, 2.0):
        slope = {k: v * mult for k, v in base.recall_slope_per_1k.items()}
        points.append(await _point(settings, replace(base, recall_slope_per_1k=slope),
                                   "degradation_multiplier", mult, repeats))
    for factor in (0.2, 0.5, 1.0):
        points.append(await _point(settings, replace(base, finding_miss_factor=factor),
                                   "finding_miss_factor", factor, repeats))
    return points
