"""
Every response must carry CORS headers — including the ones nothing returns.

A browser will not hand a cross-origin response to the page unless it carries
`Access-Control-Allow-Origin`. A reply that lacks it is not "a 429" or "a 401"
as far as the page is concerned; it is a CORS failure, which axios reports as
an opaque `Network Error` with no status. That is exactly how a deployed
sign-in failed while the backend was healthy and answering.

The cause was middleware order. `add_middleware` prepends, so the last one
added is the outermost; CORS was added first and therefore ran *innermost*.
Anything outside it that short-circuited — the rate limiter answering 429 —
produced a response CORS never saw.

These tests pin the property rather than the ordering, so they keep holding if
the stack is rearranged again for some other reason.
"""

import os

import pytest
from fastapi import APIRouter
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio

ORIGIN = "http://localhost:5173"
"""An origin the application is configured to allow in every environment."""


@pytest.fixture(scope="module")
def cors_app():
    """
    The real application, with the rate limiter live.

    The suite normally runs with `ENVIRONMENT=testing`, which switches rate
    limiting off — so a 429 could never be produced and the very response that
    broke production would go untested. This fixture puts the environment back
    and adds routes that raise on demand.
    """
    from app.core.config import settings
    from app.main import app

    original_env = settings.ENVIRONMENT
    original_testing = os.environ.get("TESTING")

    settings.ENVIRONMENT = "staging"      # not "testing": keeps the limiter on
    os.environ.pop("TESTING", None)

    probe = APIRouter()

    @probe.get("/__cors_probe/boom")
    async def boom():
        raise RuntimeError("deliberate failure")

    @probe.get("/__cors_probe/ok")
    async def ok():
        return {"ok": True}

    app.include_router(probe)
    try:
        yield app
    finally:
        settings.ENVIRONMENT = original_env
        if original_testing is not None:
            os.environ["TESTING"] = original_testing
        app.router.routes = [
            r for r in app.router.routes
            if "__cors_probe" not in getattr(r, "path", "")
        ]


@pytest.fixture
async def cors_client(cors_app):
    async with AsyncClient(
        transport=ASGITransport(app=cors_app, raise_app_exceptions=False),
        base_url="https://api.medbridge.test",
    ) as client:
        yield client


def acao(response) -> str | None:
    return response.headers.get("access-control-allow-origin")


class TestMiddlewareOrder:
    def test_cors_is_the_outermost_middleware(self, cors_app):
        """
        Position 0 is outermost. If CORS is anywhere else, some middleware can
        answer without it and the browser sees a network failure.
        """
        names = [m.cls.__name__ for m in cors_app.user_middleware]
        assert names[0] == "CORSMiddleware", names

    def test_cors_is_outside_the_rate_limiter(self, cors_app):
        names = [m.cls.__name__ for m in cors_app.user_middleware]
        assert names.index("CORSMiddleware") < names.index("RateLimitMiddleware")


class TestEveryStatusCarriesTheHeader:
    async def test_200(self, cors_client):
        r = await cors_client.get("/__cors_probe/ok", headers={"Origin": ORIGIN})
        assert r.status_code == 200
        assert acao(r) == ORIGIN

    async def test_401_unauthenticated(self, cors_client):
        r = await cors_client.get("/api/v1/patient/emergency-profile",
                                  headers={"Origin": ORIGIN})
        assert r.status_code == 401
        assert acao(r) == ORIGIN, "an unauthenticated reply had no CORS header"

    async def test_403_wrong_role(self, cors_client):
        r = await cors_client.get("/api/v1/admin/dashboard",
                                  headers={"Origin": ORIGIN,
                                           "Authorization": "Bearer not-a-token"})
        assert r.status_code in (401, 403)
        assert acao(r) == ORIGIN

    async def test_404_unknown_route(self, cors_client):
        r = await cors_client.get("/api/v1/does-not-exist",
                                  headers={"Origin": ORIGIN})
        assert r.status_code == 404
        assert acao(r) == ORIGIN

    async def test_422_validation_failure(self, cors_client):
        r = await cors_client.post("/api/v1/auth/login",
                                   json={"email": "not-an-email"},
                                   headers={"Origin": ORIGIN})
        assert r.status_code == 422
        assert acao(r) == ORIGIN

    async def test_500_unhandled_exception(self, cors_client):
        """
        The catch-all handler turns this into a JSON 500 *inside* the stack, so
        it still travels back out through CORS.
        """
        r = await cors_client.get("/__cors_probe/boom", headers={"Origin": ORIGIN})
        assert r.status_code == 500
        assert acao(r) == ORIGIN, "a 500 had no CORS header"

    async def test_429_rate_limited(self, cors_client):
        """
        The response that broke production.

        The rate limiter returns its 429 from inside its own `dispatch`, never
        touching the router. With CORS innermost that reply had no headers and
        the browser reported a network error instead of a throttle.
        """
        from app.core.config import settings

        limit = settings.RATE_LIMIT_LOGIN
        seen_429 = None
        for _ in range(limit + 5):
            r = await cors_client.post(
                "/api/v1/auth/login",
                json={"email": "cors.probe@example.com", "password": "wrong-pass"},
                headers={"Origin": ORIGIN},
            )
            if r.status_code == 429:
                seen_429 = r
                break

        assert seen_429 is not None, (
            f"never hit the {limit}/min login limit; the 429 path is untested"
        )
        assert acao(seen_429) == ORIGIN, "the 429 had no CORS header"


class TestPreflight:
    async def test_preflight_is_answered_with_the_full_header_set(self, cors_client):
        r = await cors_client.options(
            "/api/v1/auth/login/doctor",
            headers={
                "Origin": ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        assert r.status_code == 200
        assert acao(r) == ORIGIN
        allowed = r.headers.get("access-control-allow-headers", "").lower()
        assert "authorization" in allowed
        assert "content-type" in allowed
        assert "POST" in r.headers.get("access-control-allow-methods", "")

    async def test_preflight_does_not_consume_the_rate_limit(self, cors_client):
        """
        CORS now answers preflights before the limiter sees them.

        Previously each sign-in cost two requests against the budget — the
        preflight and the POST — halving the effective allowance.
        """
        for _ in range(30):
            r = await cors_client.options(
                "/api/v1/auth/login",
                headers={"Origin": ORIGIN,
                         "Access-Control-Request-Method": "POST"},
            )
            assert r.status_code == 200, f"a preflight was throttled: {r.status_code}"

    async def test_an_unlisted_origin_is_not_granted_access(self, cors_client):
        """The fix must widen nothing: an unknown origin still gets no grant."""
        r = await cors_client.get("/__cors_probe/ok",
                                  headers={"Origin": "https://attacker.example"})
        assert acao(r) != "https://attacker.example"
