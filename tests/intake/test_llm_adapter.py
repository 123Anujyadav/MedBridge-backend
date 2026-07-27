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
        adapter, stub = _adapter([RuntimeError("bad request"), '{"ok": true}'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {"ok": True}
        assert len(stub.models_tried) == 2
        assert stub.models_tried[0] != stub.models_tried[1]

    async def test_returns_empty_dict_when_all_tiers_fail(self):
        """
        The port contract forbids raising: every workflow node treats `{}` as
        'model unavailable' and falls back deterministically.
        """
        outcomes = [RuntimeError("down")] * len(_adapter([])[0]._config.models)
        adapter, stub = _adapter(outcomes)
        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {}
        assert set(stub.models_tried) == set(adapter._config.models)

    async def test_first_tier_success_does_not_retry(self):
        adapter, stub = _adapter(['{"ok": true}'])
        await adapter.complete_json(system_prompt="s", user_content="u")
        assert len(stub.models_tried) == 1

    async def test_no_configured_model_is_decommissioned(self):
        """
        Regression guard for the assistant's degraded-mode outage.

        `llama3-70b-8192` sat in the fallback chain long after Groq
        decommissioned it, so the last tier answered 400 every single time.
        Nothing in the chain may be a known-dead model.
        """
        from app.core.ai_provider import get_ai_provider_config

        decommissioned = {
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma-7b-it",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
            "llama-3.1-70b-versatile",
        }
        assert not decommissioned & set(get_ai_provider_config().models)

    async def test_transient_failure_retries_the_same_model(self):
        """
        A 429 or timeout is worth retrying on the same model; only a hard
        failure should cost a tier. Previously any error burned a tier, so three
        rate limits exhausted the whole chain and the caller degraded.
        """
        adapter, stub = _adapter([RuntimeError("rate limit exceeded"), '{"ok": true}'])
        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {"ok": True}
        assert stub.models_tried == [adapter._config.models[0]] * 2


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

    async def test_rotates_to_next_credential_when_key_is_rejected(self):
        """
        The production failure: a *revoked* key resolved ahead of a valid one.

        Resolution used to stop at the first non-empty key, so a stale
        credential took every AI feature down while a working key sat unused in
        another source. An auth failure must now advance to the next candidate
        and retry the same model.
        """
        adapter, stub = _adapter(
            [RuntimeError("Invalid API Key"), '{"ok": true}']
        )
        adapter._candidates = [("stale source", "dead-key"), ("good source", "live-key")]
        adapter._candidate_index = 0
        # Keep the stub in place across rotation; only the credential changes.
        adapter._open_client = lambda: True

        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {"ok": True}
        assert adapter._candidate_index == 1
        assert len(stub.models_tried) == 2

    async def test_reports_error_when_every_credential_is_rejected(self):
        adapter, stub = _adapter([RuntimeError("Invalid API Key")])
        adapter._candidates = [("only source", "dead-key")]
        adapter._candidate_index = 0

        result = await adapter.complete_json(system_prompt="s", user_content="u")

        assert result == {}
        # Auth failures must not be retried across models: the key is wrong, not
        # the model, so one attempt is all it should cost.
        assert len(stub.models_tried) == 1
        assert "credential" in (adapter.last_error or "")

    async def test_health_probe_detects_a_rejected_key(self):
        """Config-level health passes with a revoked key; the probe must not."""
        adapter, _ = _adapter([RuntimeError("Invalid API Key")])
        adapter._candidates = [("only source", "dead-key")]
        adapter._candidate_index = 0

        assert (await adapter.health())["status"] == "healthy"

        probed = await adapter.health(probe=True)
        assert probed["status"] == "unhealthy"
        assert probed["probe"] == "failed"

    async def test_health_reports_healthy_when_configured(self):
        adapter, _ = _adapter([])
        health = await adapter.health()
        assert health["status"] == "healthy"
        assert health["provider"] == "groq"
        assert health["model"]
        # Never leak the credential itself, only a digest.
        assert "key_fingerprint" in health
        assert adapter._config.api_key not in str(health)
