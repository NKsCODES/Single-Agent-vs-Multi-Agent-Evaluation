"""Configuration from environment variables (prefix ``ARCHEVAL_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Tier = Literal["large", "small"]


class TierPrice(BaseModel):
    """USD per one million tokens. Placeholders: set them to your deployment's price."""

    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_million
                + output_tokens * self.output_per_million) / 1_000_000


class AgentLimits(BaseModel):
    """Budgets that bound every agent the same way, so the comparison is fair."""

    single_max_iterations: int = Field(6, ge=1, le=50)
    worker_max_iterations: int = Field(4, ge=1, le=50)
    lead_max_iterations: int = Field(4, ge=1, le=20)
    max_workers: int = Field(4, ge=1, le=10)
    search_top_k: int = Field(3, ge=1, le=10)
    context_window_tokens: int = Field(16_000, ge=2_000)
    max_output_tokens: int = Field(800, ge=64)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARCHEVAL_", env_file=".env",
                                      env_nested_delimiter="__", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # Models. "sim" is the offline OpenAI-compatible simulator; "azure" uses Azure OpenAI.
    llm_provider: Literal["sim", "azure", "openai"] = "sim"
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_api_key: str | None = None  # leave empty to use Entra ID (managed identity)
    large_deployment: str = "large"
    small_deployment: str = "small"
    embedding_deployment: str = "text-embedding-3-small"
    openai_api_key: str | None = None
    prices: dict[str, TierPrice] = {
        "large": TierPrice(input_per_million=2.00, output_per_million=8.00),
        "small": TierPrice(input_per_million=0.40, output_per_million=1.60),
    }

    # Retrieval. "memory" is the in-process hybrid index; "cosmos" is Azure Cosmos DB for NoSQL.
    vector_store: Literal["memory", "cosmos"] = "memory"
    embeddings: Literal["hashing", "azure"] = "hashing"
    cosmos_endpoint: str | None = None
    cosmos_key: str | None = None  # leave empty to use Entra ID
    cosmos_database: str = "archeval"
    cosmos_container: str = "chunks"
    embedding_dimensions: int = 384

    # Responsible AI and operations
    content_safety_endpoint: str | None = None  # Azure AI Content Safety, optional
    applicationinsights_connection_string: str | None = None
    api_key: str | None = None
    max_question_chars: int = 2_000

    limits: AgentLimits = AgentLimits()


@lru_cache
def get_settings() -> Settings:
    return Settings()
