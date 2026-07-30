"""
The deployment configuration, asserted as code.

The application runs behind Railway's reverse proxy. Uvicorn enables
`--proxy-headers` by default, but that alone does nothing: `forwarded_allow_ips`
defaults to `127.0.0.1`, and the proxy reaches the container from an internal
address that is not loopback. Every `X-Forwarded-*` header was therefore
discarded, and two things followed —

* `request.client.host` was the proxy's address, so the entire platform shared
  a single rate-limit bucket. Five bad sign-in attempts from any one person
  locked out every other user for the rest of the minute.
* `request.url.scheme` was `http` behind TLS termination.

Neither shows up in a functional test, and neither is visible locally, because
a local client genuinely *is* 127.0.0.1 and so is trusted by the default. The
only thing that catches a regression here is asserting the launch command
itself.
"""

import re
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "app.dockerfile"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    if not DOCKERFILE.is_file():
        pytest.skip(f"{DOCKERFILE} not found")
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def cmd_line(dockerfile: str) -> str:
    match = re.search(r"^CMD\s+(.+)$", dockerfile, re.MULTILINE)
    assert match, "the image has no CMD"
    return match.group(1)


class TestProxyConfiguration:
    def test_the_server_is_told_to_read_proxy_headers(self, cmd_line):
        assert "--proxy-headers" in cmd_line

    def test_the_proxy_is_actually_trusted(self, cmd_line):
        """
        `--proxy-headers` without this is a no-op behind a non-loopback proxy.
        """
        assert "--forwarded-allow-ips" in cmd_line, (
            "forwarded_allow_ips defaults to 127.0.0.1, which Railway's proxy "
            "is not — every X-Forwarded-* header would be discarded"
        )

    def test_the_trusted_range_is_configurable(self, cmd_line):
        """
        Hard-coding `*` would be wrong anywhere the container is directly
        addressable: a client could then choose the IP its rate limit counts
        against. The value has to be overridable per environment.
        """
        assert "FORWARDED_ALLOW_IPS" in cmd_line

    def test_the_port_comes_from_the_platform(self, cmd_line):
        """Railway injects `$PORT`; a hard-coded port cannot receive traffic."""
        assert "${PORT" in cmd_line

    def test_the_port_still_defaults_for_local_use(self, cmd_line):
        """`docker run` and compose must keep working with no PORT set."""
        assert "${PORT:-8000}" in cmd_line

    def test_uvicorn_replaces_the_shell_so_signals_arrive(self, cmd_line):
        """
        Without `exec`, uvicorn runs as a child of `sh` and never receives
        SIGTERM — the platform's graceful shutdown becomes a hard kill.
        """
        assert "exec uvicorn" in cmd_line

    def test_the_wildcard_is_quoted(self, cmd_line):
        """
        An unquoted `*` is glob-expanded by the shell into a directory listing,
        and uvicorn then rejects the extra arguments and exits. This bit during
        development, so it is pinned.
        """
        # The quotes are backslash-escaped inside the CMD JSON array, so the
        # pattern allows for that rather than requiring a bare `"`.
        assert re.search(
            r'\\?"\$\{FORWARDED_ALLOW_IPS:-\*\}\\?"', cmd_line
        ), cmd_line


class TestMigrationsRunBeforeTraffic:
    def test_the_entrypoint_applies_migrations(self):
        entrypoint = DOCKERFILE.parent / "entrypoint.sh"
        if not entrypoint.is_file():
            pytest.skip("entrypoint.sh not found")
        body = entrypoint.read_text(encoding="utf-8")
        assert "alembic upgrade head" in body
        assert 'exec "$@"' in body, (
            "the entrypoint must exec the CMD so uvicorn keeps PID 1"
        )


class TestRateLimiterUsesTheResolvedClient:
    def test_the_limiter_keys_on_request_client_host(self):
        """
        The limiter must read `request.client.host`, which uvicorn rewrites from
        `X-Forwarded-For` once the proxy is trusted. Parsing the header itself
        would bypass uvicorn's trust check and let any caller spoof the address
        its limit is counted against.
        """
        source = (
            Path(__file__).resolve().parents[1]
            / "app" / "middleware" / "rate_limit.py"
        ).read_text(encoding="utf-8")

        assert "request.client.host" in source
        assert "x-forwarded-for" not in source.lower(), (
            "the limiter must not parse the header itself; uvicorn does it "
            "behind a trust boundary"
        )
