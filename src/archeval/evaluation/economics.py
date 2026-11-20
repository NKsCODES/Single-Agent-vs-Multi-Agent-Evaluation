"""Turn graded runs into the numbers a production decision needs.

For each architecture (overall and per task type): success rate with a 95% Wilson interval,
fact recall, groundedness, mean cost per task, cost per successful task, tokens, modeled
latency, and the break-even task value against a baseline.

Break-even value. If a successful answer is worth V dollars, an architecture's expected net
value per task is ``V * success_rate - cost_per_task``. A more expensive architecture B beats a
cheaper A only when ``V > (cost_B - cost_A) / (success_B - success_A)``. That threshold is the
question to ask before paying for multi-agent: is a correct answer worth more than V*?
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from archeval.evaluation.runner import RunRecord


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Summary:
    architecture: str
    kind: str
    n: int
    success_rate: float
    ci_low: float
    ci_high: float
    correct_rate: float
    grounded_rate: float
    fact_recall: float
    cost_per_task: float
    cost_per_success: float | None
    tokens_per_task: float
    calls_per_task: float
    latency_p50_s: float
    latency_p95_s: float
    errors: int
    routed_multi_share: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(records: Iterable[RunRecord]) -> list[Summary]:
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        groups[(r.result.architecture, r.task.kind)].append(r)
        groups[(r.result.architecture, "all")].append(r)
    out = []
    for (arch, kind), rs in groups.items():
        n = len(rs)
        wins = sum(r.grade.success for r in rs)
        cost = sum(r.result.cost_usd for r in rs)
        lo, hi = wilson(wins, n)
        lat = [r.result.latency_s for r in rs]
        routed = [r for r in rs if r.result.route_reason is not None]
        out.append(Summary(
            architecture=arch, kind=kind, n=n, success_rate=round(wins / n, 4),
            ci_low=round(lo, 4), ci_high=round(hi, 4),
            correct_rate=round(sum(r.grade.correct for r in rs) / n, 4),
            grounded_rate=round(sum(r.grade.grounded for r in rs) / n, 4),
            fact_recall=round(sum(r.grade.fact_recall for r in rs) / n, 4),
            cost_per_task=round(cost / n, 6),
            cost_per_success=round(cost / wins, 6) if wins else None,
            tokens_per_task=round(sum(r.result.input_tokens + r.result.output_tokens
                                      for r in rs) / n, 1),
            calls_per_task=round(sum(r.result.calls for r in rs) / n, 2),
            latency_p50_s=round(_pct(lat, 0.5), 2), latency_p95_s=round(_pct(lat, 0.95), 2),
            errors=sum(1 for r in rs if r.result.error),
            routed_multi_share=round(sum(r.result.ran_as.startswith("multi") for r in routed)
                                     / len(routed), 3) if routed else None,
        ))
    return sorted(out, key=lambda s: (s.kind != "all", s.kind, s.architecture))


def break_even(base: Summary, candidate: Summary) -> float | None:
    """Value of one successful task above which ``candidate`` beats ``base``. None means the
    candidate never wins (it is no better), 0.0 means it always wins (better and cheaper)."""
    ds = candidate.success_rate - base.success_rate
    dc = candidate.cost_per_task - base.cost_per_task
    if ds <= 0:
        return 0.0 if (ds == 0 and dc < 0) else None
    return max(0.0, dc / ds)


def verdict(base: Summary, candidate: Summary) -> str:
    """Plain-language decision rule for ``candidate`` against ``base``."""
    ds = candidate.success_rate - base.success_rate
    dc = candidate.cost_per_task - base.cost_per_task
    if candidate.architecture == base.architecture:
        return "baseline"
    if ds >= 0 and dc <= 0:
        return "always (as good and cheaper)" if ds == 0 else "always (better and cheaper)"
    if ds <= 0 and dc >= 0:
        return "never (no better, costs more)"
    if ds > 0:
        return f"if a correct answer is worth more than ${dc / ds:.3f}"
    return f"only if a correct answer is worth less than ${dc / ds:.3f}"


def pick(summaries: list[Summary], arch: str, kind: str = "all") -> Summary:
    return next(s for s in summaries if s.architecture == arch and s.kind == kind)
