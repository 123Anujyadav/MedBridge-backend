"""
Supabase Auth (GoTrue) client.

Supabase is the identity provider and nothing else: it holds credentials,
issues tokens, and sends verification and recovery email. It holds no patient
data, no roles and no permissions — those stay in this database, which remains
the source of truth.

The service-role key is used only by the admin calls in this module. It never
reaches a response body, a log line or the browser; the frontend receives access
and refresh tokens exactly as it did before, and nothing else.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.exceptions import (
    AuthenticationException,
    BusinessRuleValidationException,
)

logger = logging.getLogger(__name__)


class SupabaseAuthError(Exception):
    """A call to the identity provider failed."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SupabaseAuthClient:
    """
    Thin wrapper over the GoTrue REST API.

    Deliberately not the `supabase` SDK: the surface needed here is a handful of
    endpoints, and a direct client keeps both the dependency footprint and the
    failure modes obvious.
    """

    def __init__(
        self,
        base_url: str | None = None,
        anon_key: str | None = None,
        service_role_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = (base_url or settings.SUPABASE_URL).rstrip("/")
        self.anon_key = anon_key or settings.SUPABASE_ANON_KEY
        self.service_role_key = service_role_key or settings.SUPABASE_SERVICE_ROLE_KEY
        self.timeout = timeout or settings.SUPABASE_TIMEOUT_SECONDS

    # ── transport ────────────────────────────────────────────────────────

    @property
    def auth_url(self) -> str:
        return f"{self.base_url}/auth/v1"

    def _headers(self, *, service_role: bool = False,
                 access_token: str | None = None) -> dict[str, str]:
        key = self.service_role_key if service_role else self.anon_key
        return {
            "apikey": key,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token or key}",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        service_role: bool = False,
        access_token: str | None = None,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.auth_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method, url, json=json, params=params,
                    headers=self._headers(service_role=service_role,
                                          access_token=access_token),
                )
        except httpx.HTTPError as exc:
            # A provider outage must read as an upstream failure, not as bad
            # credentials — otherwise it looks to the user like a wrong password.
            logger.error("[SUPABASE_UNREACHABLE] %s %s: %s", method, path, exc)
            raise SupabaseAuthError(
                "The authentication service is temporarily unavailable.", 503
            ) from exc

        if response.status_code >= 400:
            detail = _error_message(response)
            logger.warning(
                "[SUPABASE_ERROR] %s %s -> %s: %s",
                method, path, response.status_code, detail,
            )
            raise SupabaseAuthError(detail, response.status_code)

        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return {}

    # ── credential flows (anon key) ──────────────────────────────────────

    async def sign_in(self, email: str, password: str) -> dict[str, Any]:
        """Exchange a password for a Supabase session."""
        return await self._request(
            "POST", "/token", params={"grant_type": "password"},
            json={"email": email, "password": password},
        )

    async def refresh_session(self, refresh_token: str) -> dict[str, Any]:
        """Rotate a session. Supabase invalidates the presented refresh token."""
        return await self._request(
            "POST", "/token", params={"grant_type": "refresh_token"},
            json={"refresh_token": refresh_token},
        )

    async def sign_out(self, access_token: str) -> None:
        """End the Supabase session so its refresh token can no longer be used."""
        await self._request("POST", "/logout", access_token=access_token)

    async def get_user(self, access_token: str) -> dict[str, Any]:
        """Resolve the owner of an access token; also used to verify remotely."""
        return await self._request("GET", "/user", access_token=access_token)

    async def send_password_reset(
        self, email: str, redirect_to: str | None = None
    ) -> None:
        """Ask Supabase to email a recovery link."""
        await self._request(
            "POST", "/recover",
            json={"email": email},
            params={"redirect_to": redirect_to} if redirect_to else None,
        )

    async def resend_verification(self, email: str) -> None:
        """Re-send the signup confirmation email."""
        await self._request(
            "POST", "/resend", json={"type": "signup", "email": email}
        )

    # ── administrative flows (service-role key, server-side only) ────────

    async def admin_create_user(
        self,
        email: str,
        password: str,
        *,
        email_confirm: bool = False,
        user_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create the identity for a new account.

        Called from the signup endpoints, so the identity and the local profile
        are created by one request and the browser never needs a Supabase key in
        order to register.
        """
        return await self._request(
            "POST", "/admin/users", service_role=True,
            json={
                "email": email,
                "password": password,
                "email_confirm": email_confirm,
                "user_metadata": user_metadata or {},
            },
        )

    async def admin_get_user_by_email(self, email: str) -> Optional[dict[str, Any]]:
        """Find an existing identity, so signup can link instead of duplicating."""
        try:
            result = await self._request(
                "GET", "/admin/users", service_role=True,
                params={"filter": email, "page": 1, "per_page": 1},
            )
        except SupabaseAuthError:
            return None

        if isinstance(result, dict):
            users = result.get("users") or []
        elif isinstance(result, list):
            users = result
        else:
            users = []

        for user in users:
            if (user.get("email") or "").lower() == email.lower():
                return user
        return None

    async def admin_delete_user(self, supabase_user_id: str) -> None:
        """
        Remove an identity.

        Used to roll back a half-finished signup: if the local profile cannot be
        written, the orphaned identity is deleted so the address stays free to
        register again.
        """
        await self._request(
            "DELETE", f"/admin/users/{supabase_user_id}", service_role=True
        )

    async def admin_update_user(
        self, supabase_user_id: str, **fields: Any
    ) -> dict[str, Any]:
        """Update an identity — password, email confirmation, or ban status."""
        return await self._request(
            "PUT", f"/admin/users/{supabase_user_id}",
            service_role=True, json=fields,
        )


def _error_message(response: httpx.Response) -> str:
    """Pull the most useful message out of a GoTrue error body."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "")[:200] or f"HTTP {response.status_code}"

    if isinstance(body, dict):
        for key in ("error_description", "msg", "message", "error"):
            value = body.get(key)
            if isinstance(value, str) and value:
                return value
    return f"HTTP {response.status_code}"


_client: SupabaseAuthClient | None = None


def get_supabase_auth_client() -> SupabaseAuthClient:
    """The process-wide client. Replaceable in tests."""
    global _client
    if _client is None:
        _client = SupabaseAuthClient()
    return _client


def set_supabase_auth_client(client: SupabaseAuthClient | None) -> None:
    """Install a different client (tests, or a future provider change)."""
    global _client
    _client = client


def translate_auth_error(exc: SupabaseAuthError) -> Exception:
    """
    Map a provider failure onto this platform's existing error vocabulary, so
    responses keep the shape and status codes clients already handle.
    """
    if exc.status_code in (400, 401, 403):
        message = exc.message or "Incorrect email or password."
        lowered = message.lower()
        if "already" in lowered and ("registered" in lowered or "exists" in lowered):
            return BusinessRuleValidationException(
                "This email address is already registered."
            )
        if "not confirmed" in lowered:
            return AuthenticationException(
                "Please confirm your email address before signing in."
            )
        return AuthenticationException(message)
    if exc.status_code == 422:
        return BusinessRuleValidationException(exc.message)
    return AuthenticationException(
        "The authentication service is temporarily unavailable."
    )
