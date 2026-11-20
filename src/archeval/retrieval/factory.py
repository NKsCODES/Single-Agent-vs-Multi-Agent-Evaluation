"""Build the configured retrieval store: in-memory hybrid index or Azure Cosmos DB for NoSQL."""

from __future__ import annotations

from archeval.config import Settings
from archeval.corpus.contoso import DOCUMENTS
from archeval.retrieval.cosmos import CosmosHybridStore
from archeval.retrieval.embeddings import AzureOpenAIEmbedder, Embedder, HashingEmbedder
from archeval.retrieval.store import HybridStore, InMemoryHybridStore


def build_embedder(settings: Settings) -> Embedder:
    if settings.embeddings == "azure":  # pragma: no cover - needs Azure
        if not settings.azure_openai_endpoint:
            raise ValueError("ARCHEVAL_AZURE_OPENAI_ENDPOINT is required for azure embeddings")
        return AzureOpenAIEmbedder(settings.azure_openai_endpoint, settings.embedding_deployment,
                                   settings.azure_openai_api_version,
                                   settings.azure_openai_api_key, settings.embedding_dimensions)
    return HashingEmbedder(settings.embedding_dimensions)


async def build_store(settings: Settings, seed: bool = True) -> HybridStore:
    """``seed`` loads the knowledge base into the in-memory index. Cosmos DB is persistent, so
    it is loaded once with ``archeval ingest`` instead of on every start."""
    embedder = build_embedder(settings)
    if settings.vector_store == "cosmos":  # pragma: no cover - needs Azure
        if not settings.cosmos_endpoint:
            raise ValueError("ARCHEVAL_COSMOS_ENDPOINT is required when vector_store=cosmos")
        return CosmosHybridStore(embedder, settings.cosmos_endpoint, settings.cosmos_database,
                                 settings.cosmos_container, settings.cosmos_key)
    store = InMemoryHybridStore(embedder)
    if seed:
        await store.upsert(DOCUMENTS)
    return store
