from __future__ import annotations

import pytest

from archeval.agents.system import AgentSystem
from archeval.config import Settings
from archeval.llm.clients import ClientFactory
from archeval.retrieval.factory import build_store


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="sim", vector_store="memory", embeddings="hashing",
                    log_json=False, api_key=None, applicationinsights_connection_string=None,
                    content_safety_endpoint=None)


@pytest.fixture
async def store(settings: Settings):  # type: ignore[no-untyped-def]
    return await build_store(settings, seed=True)


@pytest.fixture
async def system(settings: Settings, store):  # type: ignore[no-untyped-def]
    return AgentSystem(settings, store, ClientFactory(settings))
