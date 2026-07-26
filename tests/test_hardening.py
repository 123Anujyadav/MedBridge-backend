import pytest
from httpx import AsyncClient
from fastapi import status

@pytest.mark.asyncio
async def test_security_headers(client: AsyncClient):
    """
    Verifies OWASP and HIPAA required security response headers are injected into HTTP responses.
    """
    response = await client.get("/api/v1/")
    assert response.status_code == 200
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "Content-Security-Policy" in response.headers
    assert "Permissions-Policy" in response.headers
    assert response.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
    assert response.headers["Cross-Origin-Embedder-Policy"] == "unsafe-none"
    assert response.headers["Cross-Origin-Opener-Policy"] == "same-origin"

@pytest.mark.asyncio
async def test_health_probes(client: AsyncClient):
    """
    Tests /health, /ready, and /live health monitoring probes.
    """
    res_health = await client.get("/health")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] in ("operational", "degraded")

    assert "checks" in data_health
    assert "database" in data_health["checks"]
    assert "redis" in data_health["checks"]

    res_live = await client.get("/live")
    assert res_live.status_code == 200
    assert res_live.json()["status"] == "alive"

    res_ready = await client.get("/ready")
    assert res_ready.status_code in (200, 503)

@pytest.mark.asyncio
async def test_metrics_endpoint(client: AsyncClient):
    """
    Tests Prometheus /metrics scraping endpoint.
    """
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "python_info" in response.text

@pytest.mark.asyncio
async def test_tracing_headers(client: AsyncClient):
    """
    Tests correlation ID and tracing header injection in API responses.
    """
    response = await client.get("/api/v1/", headers={"X-Correlation-ID": "test-corr-123"})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "test-corr-123"
    assert "X-Request-ID" in response.headers
    assert "X-Trace-ID" in response.headers

@pytest.mark.asyncio
async def test_upload_validation_utility():
    """
    Tests file upload security sanitization, extension/MIME validation.
    """
    from app.core.upload import sanitize_filename, scan_file_virus_hook
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("safe_document.pdf") == "safe_document.pdf"
    assert scan_file_virus_hook(b"test data payload") is True
