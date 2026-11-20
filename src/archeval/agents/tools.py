"""The retrieval tool shared by every agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from archeval.llm.ledger import CURRENT_LEDGER
from archeval.retrieval.store import HybridStore
from archeval.safety.injection import neutralize

MAX_CHUNK_CHARS = 700


def make_search_tool(store: HybridStore, top_k: int) -> Callable[[str], Awaitable[str]]:
    async def search_knowledge_base(query: str) -> str:
        """Search Contoso's product knowledge base (overviews, security, support policies,
        release notes, catalog). Returns the most relevant passages with their source ids."""
        hits = await store.search(query[:300], top_k=top_k)
        ledger = CURRENT_LEDGER.get()
        if ledger is not None:
            ledger.retrieved_ids.update(h.id for h in hits)
        if not hits:
            return "No results."
        return "\n".join(f"[{h.id}] {neutralize(h.text[:MAX_CHUNK_CHARS])}" for h in hits)

    return search_knowledge_base
