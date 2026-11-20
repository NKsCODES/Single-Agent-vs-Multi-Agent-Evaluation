"""Per-task accounting of every LLM call: tokens, cost and latency.

Calls are recorded by an httpx response hook on the model client, so lead, worker and single
agents are all measured the same way, whichever provider is behind the client. The agent that
made a call is read from a context variable, which asyncio copies into each worker task.
"""

from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any

from archeval.config import TierPrice


@dataclass
class CallRecord:
    agent: str  # "single", "lead" or "worker-<n>"
    tier: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    wave: int = 0


@dataclass
class UsageLedger:
    calls: list[CallRecord] = field(default_factory=list)
    retrieved_ids: set[str] = field(default_factory=set)
    workers_started: int = 0

    def add(self, rec: CallRecord) -> None:
        self.calls.append(rec)

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    def tokens_by_tier(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        for c in self.calls:
            out[c.tier]["input"] += c.input_tokens
            out[c.tier]["output"] += c.output_tokens
        return dict(out)

    def cost(self, prices: dict[str, TierPrice]) -> float:
        return sum(prices[c.tier].cost(c.input_tokens, c.output_tokens) for c in self.calls)

    def critical_path_latency(self) -> float:
        """Lead and single-agent calls run one after another; workers in a wave run in
        parallel, so a wave costs as long as its slowest worker."""
        total = sum(c.latency_s for c in self.calls if not c.agent.startswith("worker"))
        waves: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for c in self.calls:
            if c.agent.startswith("worker"):
                waves[c.wave][c.agent] += c.latency_s
        return total + sum(max(w.values()) for w in waves.values())

    def to_dict(self) -> dict[str, Any]:
        return {"calls": [asdict(c) for c in self.calls], "workers": self.workers_started,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


CURRENT_LEDGER: ContextVar[UsageLedger | None] = ContextVar("archeval_ledger", default=None)
CURRENT_AGENT: ContextVar[str] = ContextVar("archeval_agent", default="single")
CURRENT_WAVE: ContextVar[int] = ContextVar("archeval_wave", default=0)
