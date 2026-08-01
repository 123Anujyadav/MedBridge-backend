"""
Rate limiting on the AI routes.

These are the only endpoints on the platform that spend money per request, and
the only ones where a single request occupies a worker for seconds. Until this
existed, `/ai/*` was the one unmetered surface: an authenticated caller could
run the assistant in a loop and the bill was the only signal.

The tests below exercise the matching and bucketing rules directly rather than
through the middleware stack, because `RateLimitMiddleware.dispatch` disables
itself under `ENVIRONMENT=testing` — which is what keeps the rest of the suite
from tripping over it.
"""

from __future__ import annotations

import base64
import json

import pytest

from app.core.config import settings
from app.middleware.rate_limit import (
    AI_PATH_PREFIX,
    PROTECTED_PATHS,
    _token_subject,
)


def match(path: str):
    """Reproduce the middleware's first-match-wins lookup."""
    for pattern, limit in PROTECTED_PATHS.items():
        if path == pattern or path.startswith(pattern + "/"):
            return pattern, limit
    return None, None


class TestEveryAiRouteIsCovered:
    """No AI route may be reachable without a budget."""

    @pytest.mark.parametrize(
        "path",
        [
            "/ai/chat",
            "/ai/symptom-intake",
            "/ai/analyze-report",
            "/ai/analyze-text",
            "/ai/assistant/messages",
            "/ai/intake/sessions",
            "/ai/intake/sessions/abc-123/turns",
            "/ai/intake/sessions/abc-123/select-doctor",
            "/ai/intake/sessions/abc-123",
            "/ai/health",
            "/ai/intake/health",
            "/ai/assistant/health",
            "/ai/assistant/conversations",
            "/ai/assistant/conversations/abc-123",
            # A route nobody has written yet is still covered by the catch-all.
            "/ai/some/future/generative/route",
        ],
    )
    def test_route_has_a_limit(self, path):
        pattern, limit = match(path)
        assert pattern is not None, f"{path} is not rate limited"
        assert isinstance(limit, int) and limit > 0

    def test_the_catch_all_exists(self):
        assert AI_PATH_PREFIX in PROTECTED_PATHS


class TestGenerativeRoutesShareOneBudget:
    """
    The expensive routes are metered together.

    Per-route budgets would let one caller run the assistant, the intake agent
    and report analysis at the same time, each at the full allowance, so the
    real ceiling would be a multiple of the configured number.
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/ai/chat",
            "/ai/symptom-intake",
            "/ai/analyze-report",
            "/ai/assistant/messages",
            "/ai/intake/sessions",
            "/ai/intake/sessions/abc-123/turns",
        ],
    )
    def test_generative_routes_bucket_together(self, path):
        pattern, limit = match(path)
        assert pattern == AI_PATH_PREFIX
        assert limit == settings.RATE_LIMIT_AI

    def test_a_conversation_shares_the_session_budget(self):
        """Starting and continuing an intake draw on the same allowance."""
        assert match("/ai/intake/sessions")[0] == match(
            "/ai/intake/sessions/abc/turns"
        )[0]


class TestReadsAreBudgetedApart:
    """A health probe must not consume a patient's conversation allowance."""

    @pytest.mark.parametrize(
        "path",
        [
            "/ai/health",
            "/ai/intake/health",
            "/ai/assistant/health",
            "/ai/assistant/conversations",
            "/ai/assistant/conversations/abc-123",
        ],
    )
    def test_read_routes_use_the_read_budget(self, path):
        _, limit = match(path)
        assert limit == settings.RATE_LIMIT_AI_READ

    def test_reads_are_more_generous_than_generation(self):
        assert settings.RATE_LIMIT_AI_READ > settings.RATE_LIMIT_AI


class TestExistingLimitsAreUnchanged:
    """Adding the AI entries must not disturb what was already protected."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/auth/login", settings.RATE_LIMIT_LOGIN),
            ("/auth/login/doctor", settings.RATE_LIMIT_LOGIN),
            ("/auth/signup/patient", settings.RATE_LIMIT_REGISTER),
            ("/auth/signup/doctor", settings.RATE_LIMIT_REGISTER),
            ("/patient/emergency", settings.RATE_LIMIT_EMERGENCY),
        ],
    )
    def test_limit_is_preserved(self, path, expected):
        assert match(path)[1] == expected

    def test_clinician_login_still_shares_the_login_budget(self):
        assert match("/auth/login/doctor")[0] == "/auth/login"

    @pytest.mark.parametrize(
        "path",
        [
            "/patient/emergency-profile",
            "/patient/emergency-profile/location",
            "/patient/dashboard",
            "/doctor/cases",
            "/admin/dashboard",
            "/shared/notifications",
            "/patient/sos",
        ],
    )
    def test_unrelated_routes_stay_unlimited(self, path):
        assert match(path)[0] is None


def _jwt(claims: dict) -> str:
    """A token shaped like a JWT. Only the payload segment is ever read."""
    segment = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"header.{segment}.signature"


def _bearer(claims: dict) -> str:
    """The full `Authorization` header value for those claims."""
    return f"Bearer {_jwt(claims)}"


class _FakeRequest:
    def __init__(self, headers: dict):
        self.headers = headers


class TestCallerIdentity:
    """
    AI traffic is charged to the account, not the address.

    Per-address bucketing alone would let one credential spend without limit by
    moving between addresses, and would make a clinic behind one NAT address
    share a single allowance between everybody in the building.
    """

    def test_subject_is_read_from_the_token(self):
        req = _FakeRequest({"Authorization": _bearer({"sub": "user-123"})})
        assert _token_subject(req) == "user-123"

    def test_bearer_prefix_is_case_insensitive(self):
        req = _FakeRequest({"Authorization": f"BeArEr {_jwt({'sub': 'user-123'})}"})
        assert _token_subject(req) == "user-123"

    @pytest.mark.parametrize(
        "claims,expected",
        [
            ({"sub": "abc"}, "abc"),
            ({"user_id": "def"}, "def"),
            ({"uid": "ghi"}, "ghi"),
            ({"sub": "", "user_id": "fallback"}, "fallback"),
        ],
    )
    def test_subject_claim_variants(self, claims, expected):
        req = _FakeRequest({"Authorization": _bearer(claims)})
        assert _token_subject(req) == expected

    @pytest.mark.parametrize(
        "header",
        [
            "",
            "Basic abc",
            "Bearer",
            "Bearer notajwt",
            "Bearer a.b",
            "Bearer a.b.c.d",
            "Bearer a.!!!not-base64!!!.c",
            "Bearer a." + base64.urlsafe_b64encode(b"not json").decode().rstrip("=") + ".c",
            "Bearer a." + base64.urlsafe_b64encode(b'"a string"').decode().rstrip("=") + ".c",
        ],
    )
    def test_unusable_tokens_fall_back_to_the_address(self, header):
        """A malformed token must never raise — it just means "no account"."""
        assert _token_subject(_FakeRequest({"Authorization": header})) is None

    def test_missing_header_is_anonymous(self):
        assert _token_subject(_FakeRequest({})) is None

    def test_subject_is_bounded(self):
        """An oversized claim cannot be used to grow the cache key without end."""
        req = _FakeRequest({"Authorization": _bearer({"sub": "x" * 5000})})
        assert len(_token_subject(req)) == 128

    def test_no_signature_verification_is_implied(self):
        """
        The value is a bucket label, not a credential.

        Reading it unverified is safe only because the route's own dependencies
        authenticate afterwards: a forged subject buys a fresh bucket for a
        request that is about to be refused, and never reaches an LLM call.
        """
        req = _FakeRequest({"Authorization": _bearer({"sub": "forged", "role": "admin"})})
        assert _token_subject(req) == "forged"
