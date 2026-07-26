"""
Tests for the Groq adapter itself.

The other suites fake at the `LLMPort` boundary, which leaves the adapter's own
behaviour — JSON parsing, tier fallback, and failure containment — unverified.
These tests stub the Groq client so that logic is exercised without a network
call or a valid API key.
"""

from __future__ import annotations

import types

import pytest

from app.intake.infrastructure.llm_groq import GroqJSONAdapter

pytestmark = pytest.mark.asyncio


def _completion(content: str):
    """Mimic the shape of a Groq chat completion response."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class StubGroqClient:
    """
    Stands in for `AsyncGroq`.

    `outcomes` is consumed one entry per call: a string is returned as content,
    an Exception is raised. This lets a test script per-tier behaviour precisely.
    """

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.models_tried: list[str] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.models_tried.append(kwargs.get("model"))
        if not self.outcomes:
            raise RuntimeError("no more scripted outcomes")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _completion(outcome)


def _adapter(outcomes: list) -> tuple[GroqJSONAdapter, StubGroqClient]:
    adapter = GroqJSONAdapter(api_key="test-key")
    stub = StubGroqClient(outcomes)
    adapter._client = stub
    return adapter, stub


class TestJsonParsing:
    async def test_parses_clean_json(self):
        adapter, _ = _adapter(['{"entities": [{"kind": "symptom"}]}'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")
        assert result == {"entities": [{"kind": "symptom"}]}

    async def test_recovers_json_wrapped_in_prose(self):
        adapter, _ = _adapter(['Here you go:\n```json\n{"ok": true}\n```\nHope that helps'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")
        assert result == {"ok": True}

    async def test_requests_json_mode(self):
        adapter, stub = _adapter(['{"ok": true}'])
        captured = {}

        async def capture(**kwargs):
            captured.update(kwargs)
            return _completion('{"ok": true}')

        stub.chat.completions.create = capture
        await adapter.complete_json(system_prompt="s", user_content="u")
        assert captured["response_format"] == {"type": "json_object"}

    async def test_non_object_json_falls_through_to_next_tier(self):
        """A bare list is not a valid payload; the adapter should retry."""
        adapter, stub = _adapter(["[1, 2, 3]", '{"ok": true}'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")
        assert result == {"ok": True}
        assert len(stub.models_tried) == 2

    @pytest.mark.parametrize("junk", ["", "   ", "not json at all", "{broken"])
    async def test_unparseable_content_yields_empty_dict(self, junk):
        adapter, _ = _adapter([junk, junk, junk])
        result = await adapter.complete_json(system_prompt="s", user_content="u")
        assert result == {}


class TestTierFallback:
    async def test_falls_back_to_next_model_on_error(self):
        adapter, stub = _adapter([RuntimeError("rate limited"), '{"ok": true}'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {"ok": True}
        assert len(stub.models_tried) == 2
        assert stub.models_tried[0] != stub.models_tried[1]

    async def test_returns_empty_dict_when_all_tiers_fail(self):
        """
        The port contract forbids raising: every workflow node treats `{}` as
        'model unavailable' and falls back deterministically.
        """
        adapter, stub = _adapter(
            [RuntimeError("down"), RuntimeError("down"), RuntimeError("down")]
        )
        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {}
        assert len(stub.models_tried) == 3

    async def test_first_tier_success_does_not_retry(self):
        adapter, stub = _adapter(['{"ok": true}'])
        await adapter.complete_json(system_prompt="s", user_content="u")
        assert len(stub.models_tried) == 1


class TestConfiguration:
    """
    Credential resolution now runs through `app.core.ai_provider`, so 'no key
    anywhere' is simulated with an empty central config rather than by clearing
    a module-level setting.
    """

    @pytest.fixture
    def unconfigured(self, monkeypatch):
        """
        Simulate no credential anywhere.

        The adapter resolves its key through `app.core.ai_provider`, so the
        central config is what must report empty.
        """
        from app.core.ai_provider import AIProviderConfig

        empty = AIProviderConfig()
        empty._file_values = {}
        monkeypatch.setattr(empty, "_resolve", lambda key, default="": default)
        return GroqJSONAdapter(config=empty)

    async def test_missing_api_key_returns_empty_without_calling(self, unconfigured):
        assert unconfigured._client is None
        assert (
            await unconfigured.complete_json(system_prompt="s", user_content="u") == {}
        )

    async def test_health_reports_unhealthy_without_key(self, unconfigured):
        health = await unconfigured.health()
        assert health["status"] == "unhealthy"
        assert "GROQ_API_KEY" in health["error"]

    async def test_all_services_share_one_credential(self):
        """The whole point of centralising: one key across every AI service."""
        from app.assistant.infrastructure.llm import AssistantGroqAdapter
        from app.core.ai_provider import get_ai_provider_config, get_groq_api_key

        central = get_ai_provider_config().fingerprint
        assert GroqJSONAdapter()._config.fingerprint == central
        assert AssistantGroqAdapter()._config.fingerprint == central
        assert get_groq_api_key() == get_ai_provider_config().api_key

    async def test_health_reports_healthy_when_configured(self):
        adapter, _ = _adapter([])
        health = await adapter.health()
        assert health["status"] == "healthy"
        assert health["provider"] == "groq"
        assert health["model"]
        # Never leak the credential itself, only a digest.
        assert "key_fingerprint" in health
        assert adapter._config.api_key not in str(health)
