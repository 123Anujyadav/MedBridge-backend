"""
Token verification, behind one interface.

Two things this module exists to guarantee:

* **There is exactly one place a token is verified.** `api/deps.py` and the
  WebSocket endpoint previously each decoded JWTs with their own copy of the
  `jwt.decode(...)` call, so a change to signing or claim validation had to be
  made in several places and could silently drift out of step.
* **The provider can change without touching endpoints.** Everything above this
  module deals in a `VerifiedIdentity` — a subject id and a token type — and
  never in provider-specific claims. Moving to Supabase means adding a verifier
  here that validates its JWTs (via the project's JWT secret or JWKS) and
  returns the same `VerifiedIdentity`; no route, service or dependency changes.

What this module deliberately does **not** do is tell the caller who the user
*is*. A token proves possession of a credential, nothing more. Role, active
status and clinical verification are read from the database by
`api/deps.py`, because a role claim inside a token is asserted by whoever minted
the token and must never be the basis of an authorisation decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import jwt

from app.core.config import settings

logger = logging.getLogger(__name__)


class InvalidTokenError(Exception):
    """The token was absent, malformed, expired or not signed by us."""


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """
    The only thing a verified token is allowed to establish.

    Note there is no `role` field, by design: adding one would invite a caller
    to authorise against a claim the token's issuer controls.

    `subject` means "whoever this token is for" in the issuer's namespace — the
    local user id under the built-in provider, the Supabase user id under
    Supabase. Resolving that to a row in `users` is `api/deps.py`'s job.
    """

    subject: str
    token_type: str
    email: str | None = None
    """Present for Supabase tokens; used to link an existing account by email."""

    provider: str = "local"

    def require_access_token(self) -> "VerifiedIdentity":
        """Reject a refresh token presented where an access token is required."""
        if self.token_type != "access":
            raise InvalidTokenError("Expected an access token.")
        return self


class TokenVerifier(Protocol):
    """Implemented per identity provider."""

    def verify(self, token: str) -> VerifiedIdentity:
        """Return the verified identity, or raise `InvalidTokenError`."""
        ...


class LocalJWTVerifier:
    """
    Verifies the HS256 tokens this backend issues itself.

    Tokens are minted by `app.core.security`; this is the counterpart that
    checks them.
    """

    def verify(self, token: str) -> VerifiedIdentity:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM],
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError("Signature verification failed or token expired.") from exc

        subject = payload.get("sub")
        token_type = payload.get("type")
        if not subject or not token_type:
            raise InvalidTokenError("Token is missing required claims.")

        return VerifiedIdentity(subject=str(subject), token_type=str(token_type))


class SupabaseJWTVerifier:
    """
    Verifies access tokens issued by Supabase Auth.

    Supabase projects sign tokens in one of two ways depending on their vintage,
    and a project can be migrated between them, so all three routes are
    supported and tried in order of cost:

    1. **HS256** with the project's JWT secret, when `SUPABASE_JWT_SECRET` is set.
    2. **Asymmetric (ES256/RS256)** against the project's published JWKS, cached.
    3. **The Auth API** as a last resort — `GET /auth/v1/user` asks Supabase to
       validate the token. Always correct, one network call per request, so it
       is only reached when neither key source is usable.

    Signature checks come first. The `sub` and `email` claims are read only
    after the token is proven authentic.
    """

    def __init__(self) -> None:
        self._jwks: dict[str, Any] | None = None
        self._jwks_fetched_at: float = 0.0

    # -- key sources ------------------------------------------------------

    def _fetch_jwks(self) -> dict[str, Any] | None:
        """Fetch and cache the project's public keys."""
        import time

        import httpx

        fresh = (
            self._jwks is not None
            and (time.time() - self._jwks_fetched_at)
            < settings.SUPABASE_JWKS_CACHE_SECONDS
        )
        if fresh:
            return self._jwks

        url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        try:
            response = httpx.get(url, timeout=settings.SUPABASE_TIMEOUT_SECONDS)
            response.raise_for_status()
            self._jwks = response.json()
            self._jwks_fetched_at = time.time()
            return self._jwks
        except Exception as exc:  # network, DNS, malformed payload
            logger.warning("[SUPABASE_JWKS_UNAVAILABLE] %s", exc)
            return None

    def _verify_with_jwks(self, token: str) -> dict[str, Any] | None:
        jwks = self._fetch_jwks()
        if not jwks or not jwks.get("keys"):
            return None
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            for key in jwks["keys"]:
                if kid and key.get("kid") != kid:
                    continue
                public_key = jwt.PyJWK(key).key
                return jwt.decode(
                    token,
                    public_key,
                    algorithms=[key.get("alg", header.get("alg", "ES256"))],
                    audience="authenticated",
                    options={"verify_aud": False},
                )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError("Supabase token failed verification.") from exc
        except Exception as exc:
            logger.warning("[SUPABASE_JWKS_VERIFY_FAILED] %s", exc)
        return None

    def _verify_with_secret(self, token: str) -> dict[str, Any] | None:
        if not settings.SUPABASE_JWT_SECRET:
            return None
        try:
            return jwt.decode(
                token,
                settings.SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated",
                options={"verify_aud": False},
            )
        except jwt.PyJWTError as exc:
            raise InvalidTokenError("Supabase token failed verification.") from exc

    def _verify_remotely(self, token: str) -> dict[str, Any]:
        """Let Supabase itself judge the token."""
        import httpx

        from app.core.supabase import get_supabase_auth_client

        client = get_supabase_auth_client()
        try:
            response = httpx.get(
                f"{client.auth_url}/user",
                headers={"apikey": client.anon_key,
                         "Authorization": f"Bearer {token}"},
                timeout=client.timeout,
            )
        except httpx.HTTPError as exc:
            raise InvalidTokenError(
                "The authentication service is temporarily unavailable."
            ) from exc

        if response.status_code >= 400:
            raise InvalidTokenError("Supabase rejected this access token.")

        body = response.json()
        return {"sub": body.get("id"), "email": body.get("email"), "role": "authenticated"}

    # -- verification -----------------------------------------------------

    def verify(self, token: str) -> VerifiedIdentity:
        claims = self._verify_with_secret(token)
        if claims is None:
            claims = self._verify_with_jwks(token)
        if claims is None:
            claims = self._verify_remotely(token)

        subject = claims.get("sub")
        if not subject:
            raise InvalidTokenError("Supabase token is missing its subject.")

        # Supabase issues access tokens here; refresh tokens are opaque strings
        # exchanged at the token endpoint and never presented as bearer tokens.
        return VerifiedIdentity(
            subject=str(subject),
            token_type="access",
            email=claims.get("email"),
            provider="supabase",
        )


def _build_verifier() -> TokenVerifier:
    if settings.AUTH_PROVIDER == "supabase":
        logger.info("[IDENTITY] Supabase is the identity provider.")
        return SupabaseJWTVerifier()
    logger.info("[IDENTITY] Built-in JWT is the identity provider.")
    return LocalJWTVerifier()


_verifier: TokenVerifier = _build_verifier()


def get_token_verifier() -> TokenVerifier:
    """The active verifier. Swapping providers happens here and nowhere else."""
    return _verifier


def set_token_verifier(verifier: TokenVerifier) -> None:
    """Install a different provider (used by tests, and by a future migration)."""
    global _verifier
    _verifier = verifier
