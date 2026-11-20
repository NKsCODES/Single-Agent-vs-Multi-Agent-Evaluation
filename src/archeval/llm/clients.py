"""Chat clients for Microsoft Agent Framework, one per model tier.

* ``azure``: Azure OpenAI with Entra ID (``DefaultAzureCredential``, i.e. managed identity in
  Azure) or a key.
* ``openai``: the OpenAI API.
* ``sim``: the offline simulator behind an ``httpx.MockTransport``. Everything above the
  transport is the production path.

Every client gets an httpx response hook that records tokens and latency in the task's ledger.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx2 as httpx  # the HTTP client the OpenAI SDK v3 is built on
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncAzureOpenAI, AsyncOpenAI

from archeval.config import Settings, Tier
from archeval.llm.ledger import CURRENT_AGENT, CURRENT_LEDGER, CURRENT_WAVE, CallRecord
from archeval.llm.sim import SimConfig, SimLLM


def _tier_of(settings: Settings, model: str) -> str:
    if model == settings.small_deployment:
        return "small"
    return "large"


def _hooks(settings: Settings) -> dict[str, list[Any]]:
    async def on_request(request: httpx.Request) -> None:
        request.extensions["archeval_start"] = time.perf_counter()

    async def on_response(response: httpx.Response) -> None:
        ledger = CURRENT_LEDGER.get()
        if ledger is None or not response.request.url.path.endswith("/chat/completions"):
            return
        await response.aread()
        if response.status_code >= 400:
            return
        usage = response.json().get("usage") or {}
        model = json.loads(response.request.content or b"{}").get("model", "")
        sim_latency = response.headers.get("x-sim-latency")
        started = response.request.extensions.get("archeval_start", time.perf_counter())
        latency = float(sim_latency) if sim_latency else time.perf_counter() - started
        ledger.add(CallRecord(agent=CURRENT_AGENT.get(), tier=_tier_of(settings, model),
                              input_tokens=int(usage.get("prompt_tokens", 0)),
                              output_tokens=int(usage.get("completion_tokens", 0)),
                              latency_s=round(latency, 4), wave=CURRENT_WAVE.get()))

    return {"request": [on_request], "response": [on_response]}


class ClientFactory:
    """Builds one Agent Framework chat client per tier, sharing one HTTP connection pool."""

    def __init__(self, settings: Settings, sim_config: SimConfig | None = None) -> None:
        self.settings = settings
        hooks = _hooks(settings)
        if settings.llm_provider == "sim":
            self.sim = SimLLM(sim_config, {settings.large_deployment: "large",
                                           settings.small_deployment: "small"})
            http = httpx.AsyncClient(transport=httpx.MockTransport(self.sim.handle),
                                     event_hooks=hooks)
            self.openai: AsyncOpenAI = AsyncOpenAI(api_key="sim", base_url="http://sim.local/v1",
                                                   http_client=http, max_retries=0)
        elif settings.llm_provider == "azure":  # pragma: no cover - needs Azure
            http = httpx.AsyncClient(event_hooks=hooks, timeout=60)
            kwargs: dict[str, Any] = {"azure_endpoint": settings.azure_openai_endpoint,
                                      "api_version": settings.azure_openai_api_version,
                                      "http_client": http, "max_retries": 3}
            if settings.azure_openai_api_key:
                kwargs["api_key"] = settings.azure_openai_api_key
            else:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider

                kwargs["azure_ad_token_provider"] = get_bearer_token_provider(
                    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            self.openai = AsyncAzureOpenAI(**kwargs)
        else:  # pragma: no cover - needs a key
            http = httpx.AsyncClient(event_hooks=hooks, timeout=60)
            self.openai = AsyncOpenAI(api_key=settings.openai_api_key, http_client=http,
                                      max_retries=3)

    def chat(self, tier: Tier) -> OpenAIChatCompletionClient:
        model = self.settings.large_deployment if tier == "large" else \
            self.settings.small_deployment
        return OpenAIChatCompletionClient(model=model, async_client=self.openai)
