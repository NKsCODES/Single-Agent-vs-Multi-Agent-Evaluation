"""Logs, metrics and traces.

* Structured JSON logs (structlog), with personal data redacted from questions.
* Prometheus metrics for the decision this service exists to inform: cost, tokens, success
  signals and latency, labelled by the architecture that actually ran.
* OpenTelemetry traces. Microsoft Agent Framework emits GenAI spans (agent runs, model calls,
  tool calls with token usage); when an Application Insights connection string is set they go
  to Azure Monitor, otherwise nothing is exported.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog
from prometheus_client import Counter, Histogram

from archeval.config import Settings

REQUESTS = Counter("archeval_requests_total", "Answered questions",
                   ["architecture", "ran_as", "outcome"])
COST = Counter("archeval_cost_usd_total", "Model spend in USD", ["ran_as"])
TOKENS = Counter("archeval_tokens_total", "Model tokens", ["ran_as", "direction"])
UNGROUNDED = Counter("archeval_ungrounded_answers_total",
                     "Answers with an unknown citation or an uncited fact", ["ran_as"])
BLOCKED = Counter("archeval_blocked_requests_total", "Requests rejected by input screening",
                  ["reason"])
LATENCY = Histogram("archeval_answer_latency_seconds", "Wall time to answer", ["ran_as"],
                    buckets=(0.5, 1, 2, 5, 10, 20, 40, 80))
WORKERS = Histogram("archeval_workers_per_answer", "Worker agents started per answer",
                    ["ran_as"], buckets=(0, 1, 2, 3, 4, 6, 8, 12))


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=settings.log_level, format="%(message)s")
    renderer: Any = (structlog.processors.JSONRenderer() if settings.log_json
                     else structlog.dev.ConsoleRenderer())
    structlog.configure(processors=[structlog.contextvars.merge_contextvars,
                                    structlog.processors.add_log_level,
                                    structlog.processors.TimeStamper(fmt="iso"), renderer],
                        wrapper_class=structlog.make_filtering_bound_logger(
                            logging.getLevelName(settings.log_level)))


def configure_tracing(settings: Settings, app: Any = None) -> bool:  # pragma: no cover
    """Send traces and metrics to Azure Monitor when configured. Returns True if enabled."""
    if not settings.applicationinsights_connection_string:
        return False
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(
        connection_string=settings.applicationinsights_connection_string,
        logger_name="archeval")
    from agent_framework.observability import enable_instrumentation

    enable_instrumentation(enable_sensitive_data=False)  # no prompt text in traces
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz,metrics")
    return True


def record(result: Any) -> None:
    ran = result.ran_as
    outcome = "error" if result.error else ("grounded" if result.grounded else "ungrounded")
    REQUESTS.labels(result.architecture, ran, outcome).inc()
    COST.labels(ran).inc(result.cost_usd)
    TOKENS.labels(ran, "input").inc(result.input_tokens)
    TOKENS.labels(ran, "output").inc(result.output_tokens)
    LATENCY.labels(ran).observe(result.wall_s)
    WORKERS.labels(ran).observe(result.workers)
    if not result.grounded:
        UNGROUNDED.labels(ran).inc()
