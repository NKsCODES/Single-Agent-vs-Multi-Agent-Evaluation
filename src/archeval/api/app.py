"""HTTP API.

POST /v1/ask                 answer a question; mode auto (router), single, multi, or a named
                             architecture. Returns the answer, citations, groundedness, tokens,
                             cost and latency for that request.
GET  /v1/architectures       the architectures and the router's decision for a question
GET  /v1/benchmark/latest    the latest benchmark summary (written by CI or `archeval bench`)
GET  /healthz /readyz        liveness and readiness (readiness checks the vector store)
GET  /metrics                Prometheus metrics
"""

from __future__ import annotations

import asyncio
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from archeval import telemetry
from archeval.agents.router import route
from archeval.agents.system import ARCHITECTURES, AgentSystem
from archeval.config import Settings, get_settings
from archeval.llm.clients import ClientFactory
from archeval.retrieval.factory import build_store
from archeval.safety.grounding import redact
from archeval.safety.injection import PromptShields, screen_question

log = structlog.get_logger("archeval.api")
MODES = {"auto": "routed", "single": "single-large", "multi": "multi-mixed"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    mode: Literal["auto", "single", "multi", "single-small", "single-large", "multi-large",
                  "multi-mixed", "routed"] = "auto"


class AskResponse(BaseModel):
    request_id: str
    architecture: str
    ran_as: str
    route_reason: str | None
    answer: str
    citations: list[str]
    grounded: bool
    workers: int
    calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    wall_s: float


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        telemetry.configure_logging(settings)
        store = await build_store(settings, seed=True)
        app.state.store = store
        app.state.system = AgentSystem(settings, store, ClientFactory(settings))
        app.state.shields = PromptShields(settings.content_safety_endpoint) \
            if settings.content_safety_endpoint else None
        log.info("startup", provider=settings.llm_provider, store=settings.vector_store)
        yield

    app = FastAPI(title="Agent Architecture Evaluation", version="1.0.0", lifespan=lifespan)
    telemetry.configure_tracing(settings, app)

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key and not (x_api_key and
                                     hmac.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(401, "missing or invalid API key")

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        structlog.contextvars.bind_contextvars(request_id=rid)
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["x-request-id"] = rid
        response.headers["server-timing"] = f"app;dur={(time.perf_counter() - start) * 1000:.0f}"
        return response

    @app.post("/v1/ask", response_model=AskResponse, dependencies=[Depends(require_key)])
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        screen = screen_question(body.question, settings.max_question_chars)
        if screen.allowed and request.app.state.shields is not None:  # pragma: no cover
            attacked = await asyncio.to_thread(request.app.state.shields.attack_detected,
                                               body.question)
            if attacked:
                screen = screen.__class__(False, "prompt shield")
        if not screen.allowed:
            telemetry.BLOCKED.labels(screen.reason or "blocked").inc()
            log.warning("blocked", reason=screen.reason)
            raise HTTPException(422, f"question rejected: {screen.reason}")
        arch = ARCHITECTURES[MODES.get(body.mode, body.mode)]
        result = await request.app.state.system.run(body.question, arch)
        telemetry.record(result)
        log.info("answered", question=redact(body.question)[:200], architecture=arch.name,
                 ran_as=result.ran_as, grounded=result.grounded, cost_usd=result.cost_usd,
                 tokens=result.input_tokens + result.output_tokens, workers=result.workers,
                 error=result.error)
        if result.error and not result.answer:
            raise HTTPException(502, "the model call failed; try again")
        return AskResponse(request_id=request.state.request_id, route_reason=result.route_reason,
                           **{k: v for k, v in result.to_dict().items()
                              if k in AskResponse.model_fields and k not in
                              ("request_id", "route_reason")})

    @app.get("/v1/architectures", dependencies=[Depends(require_key)])
    async def architectures(question: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"architectures": {k: v.__dict__ for k, v in
                                                 ARCHITECTURES.items()}, "modes": MODES}
        if question:
            out["route"] = route(question).__dict__
        return out

    @app.get("/v1/benchmark/latest", dependencies=[Depends(require_key)])
    async def benchmark_latest() -> dict[str, Any]:
        path = Path("reports/latest/summary.json")
        if not path.exists():
            raise HTTPException(404, "no benchmark has been run yet")
        data = json.loads(path.read_text())
        return {"settings": data["settings"],
                "overall": [s for s in data["summaries"] if s["kind"] == "all"],
                "decision": data["decision"]}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict[str, str]:
        if not await request.app.state.store.ping():
            raise HTTPException(503, "vector store unavailable")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
