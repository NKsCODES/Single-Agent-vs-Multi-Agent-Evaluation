"""Prompt-injection defences.

* ``screen_question`` checks user input with local heuristics, and with Azure AI Content Safety
  Prompt Shields when an endpoint is configured.
* ``neutralize`` defuses instructions hidden in retrieved documents (indirect injection) before
  they reach a model: the text stays readable as data, but imperative override phrases are
  marked as quoted content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the |all )?(system|previous) (prompt|instructions)",
    r"you are now (?:in )?(?:developer|dan|jailbreak) mode",
    r"reveal (your|the) (system prompt|instructions)",
    r"<\s*/?\s*system\s*>",
]
INJECTION = re.compile("|".join(_PATTERNS), re.I)


@dataclass(frozen=True)
class ScreenResult:
    allowed: bool
    reason: str | None = None


def screen_question(text: str, max_chars: int = 2000) -> ScreenResult:
    if not text.strip():
        return ScreenResult(False, "empty question")
    if len(text) > max_chars:
        return ScreenResult(False, f"question longer than {max_chars} characters")
    if INJECTION.search(text):
        return ScreenResult(False, "possible prompt injection")
    return ScreenResult(True)


def neutralize(text: str) -> str:
    return INJECTION.sub(lambda m: f"[quoted text: {m.group(0)}]", text)


class PromptShields:  # pragma: no cover - needs an Azure AI Content Safety resource
    """Azure AI Content Safety Prompt Shields, called over REST with Entra ID."""

    def __init__(self, endpoint: str) -> None:
        import httpx
        from azure.identity import DefaultAzureCredential

        self.endpoint = endpoint.rstrip("/")
        self.credential = DefaultAzureCredential()
        self.http = httpx.Client(timeout=10)

    def attack_detected(self, user_prompt: str, documents: list[str] | None = None) -> bool:
        token = self.credential.get_token("https://cognitiveservices.azure.com/.default").token
        r = self.http.post(
            f"{self.endpoint}/contentsafety/text:shieldPrompt",
            params={"api-version": "2024-09-01"},
            headers={"Authorization": f"Bearer {token}"},
            json={"userPrompt": user_prompt, "documents": documents or []},
        )
        r.raise_for_status()
        data = r.json()
        user = data.get("userPromptAnalysis", {}).get("attackDetected", False)
        docs = any(d.get("attackDetected") for d in data.get("documentsAnalysis", []))
        return bool(user or docs)
