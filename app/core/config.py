import json
import logging
import os
from pathlib import Path
from typing import Any, List, Union
from pydantic import AnyHttpUrl, BeforeValidator, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated

logger = logging.getLogger(__name__)

INSECURE_DEFAULT_JWT_SECRET = "supersecretjwtkeyforaronofyhealthapplication2026"
"""
The development fallback signing key.

It is committed to source control, so any deployment still using it can have
its access tokens forged by anyone with a copy of this repository. Production
startup refuses to proceed with it — see `_reject_insecure_production_config`.
"""

MIN_PRODUCTION_SECRET_LENGTH = 32

def parse_cors_origins(v: Any) -> Any:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, (list, str)):
        return v
    raise ValueError(v)

def _resolve_env_file() -> str:
    """
    Which env file this process configures itself from.

    Defaults to `Backend/.env`, so nothing about local development changes.
    Set `APP_ENV_FILE` to point at another one — `APP_ENV_FILE=../.prod.env`
    runs migrations, scripts and the server against the production Railway
    database without editing, copying or overriding any file.

    The name is resolved relative to the current directory first and then to the
    repository root, because these processes are started from both places
    (`alembic` from `Backend/`, deployment tooling from the root) and a config
    file that silently fails to load is far worse than one that cannot be found:
    the first produces a process quietly pointed at the wrong database.
    """
    name = os.getenv("APP_ENV_FILE", ".env")

    candidate = Path(name)
    if candidate.is_file():
        return str(candidate.resolve())

    # `app/core/config.py` -> `app/core` -> `app` -> `Backend` -> repo root
    for base in (Path(__file__).resolve().parents[2],
                 Path(__file__).resolve().parents[3]):
        candidate = base / name
        if candidate.is_file():
            return str(candidate.resolve())

    if os.getenv("APP_ENV_FILE"):
        raise RuntimeError(
            f"APP_ENV_FILE is set to '{name}' but no such file was found. "
            "Refusing to start against an unknown configuration."
        )
    return name


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_resolve_env_file(),
        env_ignore_empty=True,
        extra="ignore"
    )
    
    PROJECT_NAME: str = "MedBridge Backend"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgrespassword@localhost:5432/aronofy_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Security
    JWT_SECRET: str = INSECURE_DEFAULT_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Rate Limiting
    RATE_LIMIT_LOGIN: int = 5
    RATE_LIMIT_REGISTER: int = 5
    RATE_LIMIT_EMERGENCY: int = 10
    RATE_LIMIT_DEFAULT: int = 60

    RATE_LIMIT_AI: int = 30
    """
    Requests per minute against the generative AI endpoints, per caller.

    These are the only routes on the platform that spend money per request: an
    LLM call to Groq. Left unlimited they are both a billing exposure and the
    cheapest denial-of-service target here, because one request occupies a
    worker for seconds rather than milliseconds.

    Thirty is chosen to be unreachable by a person and obvious to a script. A
    model call takes roughly two to twenty seconds, so a patient working
    quickly manages a handful a minute; a loop reaches this in under a second.
    """

    RATE_LIMIT_AI_READ: int = 90
    """
    Requests per minute for the AI routes that only read stored state.

    Conversation history and the agent health probes cost a database query, not
    a model call, so they are budgeted separately — otherwise a dashboard
    polling `/ai/*/health` would consume the same allowance the patient needs
    in order to hold a conversation.
    """

    # AI Agent Integration
    GROQ_API_KEY: str = ""

    # ── Identity provider ────────────────────────────────────────────────
    #
    # Supabase is the identity provider only. This database stays the source
    # of truth for users, roles, permissions and every clinical record; no
    # application data is stored in Supabase.
    AUTH_PROVIDER: str = "local"
    """`local` (built-in JWT) or `supabase`. Switching is an env-var change."""

    SUPABASE_URL: str = ""
    SUPABASE_ANON_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    """Server-side only. Never returned by an API and never sent to a browser."""

    SUPABASE_JWT_SECRET: str = ""
    """
    Optional. Set for projects that still sign access tokens with HS256; the
    verifier falls back to the project's JWKS, and then to the Auth API, when
    it is absent.
    """

    SUPABASE_TIMEOUT_SECONDS: float = 15.0
    SUPABASE_JWKS_CACHE_SECONDS: int = 600

    SUPABASE_JWT_LEEWAY_SECONDS: int = 60
    """
    Clock tolerance when validating `iat`, `nbf` and `exp` on Supabase tokens.

    Supabase stamps `iat` from its own clock. A host running even a fraction of
    a second behind it therefore sees a freshly minted token as issued in the
    future, and PyJWT refuses it — so a user signs in successfully and is
    immediately logged out by their first authenticated request. The failure is
    intermittent by nature, which is what makes it worth a deliberate tolerance
    rather than an assumption that the clocks agree.

    A minute of leeway does not meaningfully extend a token's life (they run to
    an hour) and is the usual allowance for distributed clock drift.
    """

    SUPABASE_AUTOCONFIRM_SIGNUP: bool = False
    """
    Whether a newly registered address is confirmed without an email round trip.

    False keeps email verification with Supabase, which is what the policy asks
    for: the project sends the confirmation message and refuses sign-in until
    the address is proven. Set true only where a mailbox is not available, such
    as a demo environment.
    """

    SUPABASE_MIGRATE_EXISTING_ON_LOGIN: bool = True
    """
    Provision a Supabase identity for a pre-existing account the first time its
    owner signs in, after their password has been checked against the local
    hash. Bcrypt hashes cannot be imported into Supabase, so without this every
    established user would have to reset their password to keep using the
    platform.
    """


    # ── Emergency communication (Phase 3) ────────────────────────────────
    #
    # Every one of these is optional. A missing credential disables the channel
    # it belongs to and nothing else: the SOS workflow keeps recording
    # emergencies, the dashboards keep updating, and the attempt is written to
    # the communication log as `skipped` with the reason. An emergency system
    # that refuses to record an emergency because a telephony vendor is
    # unconfigured would be worse than one that cannot telephone.

    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_PHONE_NUMBER: str = ""
    """The caller ID for voice and the sender for SMS."""

    TWILIO_WHATSAPP_NUMBER: str = ""
    """
    The WhatsApp sender, in `whatsapp:+…` form or as a bare number.

    Separate from `TWILIO_PHONE_NUMBER` because a WhatsApp sender is a
    different, separately-approved identity — the sandbox number during
    development, an approved business sender in production. Absent means the
    WhatsApp channel is skipped, not that the alert fails.
    """

    GOOGLE_MAPS_API_KEY: str = ""
    """
    Enables reverse geocoding, nearby-hospital search and distance/ETA.

    Deliberately optional and read at call time, not at import. Adding the key
    to the environment and restarting turns every Maps-backed feature on with
    no code change; while it is absent the platform falls back to the stored
    coordinates and a plain map link, and never invents an address, a hospital
    or an ETA.
    """

    PUBLIC_BASE_URL: str = ""
    """
    The externally reachable origin of this API, e.g. `https://api.example.com`.

    Twilio delivery callbacks are only requested when this is set: a callback
    URL Twilio cannot reach is worse than none, because the provider retries it
    and the log fills with failures that say nothing about the message. Absent
    means delivery tracking stops at "accepted by the provider", which is
    recorded honestly rather than guessed at. Adding it later turns full
    lifecycle tracking on with no code change.
    """

    EMERGENCY_COMMS_ENABLED: bool = True
    """
    Master switch for outbound emergency communication.

    Exists so a staging or demo deployment can run the whole SOS workflow —
    records, timeline, realtime — without placing real telephone calls to real
    people. Attempts are logged as `skipped`, so the audit trail still shows
    what would have been sent.
    """

    EMERGENCY_COMMS_MAX_ATTEMPTS: int = 4
    EMERGENCY_COMMS_BACKOFF_SECONDS: int = 30
    """
    First retry delay. Each further attempt doubles it — 30s, 60s, 120s — so a
    vendor outage of a few minutes is ridden out without a operator, and a
    permanently bad number stops being retried instead of looping forever.
    """

    EMERGENCY_RETRY_SWEEP_SECONDS: int = 30
    """
    How often the in-process sweeper looks for due retries.

    The sweep also runs as a Celery beat task. Both are safe together: rows are
    claimed with an atomic conditional update, so whichever gets there first
    owns the attempt. The in-process loop exists because Celery needs a broker,
    and an emergency notification must not depend on one being reachable.
    """

    # CORS
    BACKEND_CORS_ORIGINS: Annotated[
        List[str], BeforeValidator(parse_cors_origins)
    ] = []

    @field_validator("DATABASE_URL", mode="after")
    @classmethod
    def _require_async_driver(cls, v: str) -> str:
        """
        Normalise a managed provider's connection string to the asyncpg driver.

        Railway, Heroku and most managed PostgreSQL services hand out URLs in
        the `postgresql://` (or legacy `postgres://`) form, which SQLAlchemy
        resolves to psycopg2 — a driver this application does not install and
        cannot use, because the engine is created with `create_async_engine`.
        The result is a `ModuleNotFoundError: psycopg2` at import time, before
        any request is served.

        Rewriting the scheme here means the provider's URL can be used exactly
        as issued. Nothing has to be hand-edited into the environment file, so
        there is no second copy of a production credential to keep in step, and
        a URL that already names a driver is left untouched.
        """
        if not v:
            return v
        for prefix in ("postgresql://", "postgres://"):
            if v.startswith(prefix):
                return "postgresql+asyncpg://" + v[len(prefix):]
        return v

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            return json.loads(v)
        return v

    @model_validator(mode="after")
    def _validate_identity_provider(self) -> "Settings":
        """
        A misconfigured identity provider must not start silently.

        Selecting `supabase` without the project URL and anon key would leave
        every login failing at runtime with an opaque error; failing here says
        exactly what is missing.
        """
        provider = self.AUTH_PROVIDER.strip().lower()
        if provider not in ("local", "supabase"):
            raise ValueError(
                f"AUTH_PROVIDER must be 'local' or 'supabase', got '{self.AUTH_PROVIDER}'."
            )
        self.AUTH_PROVIDER = provider

        if provider == "supabase":
            missing = [
                name for name, value in (
                    ("SUPABASE_URL", self.SUPABASE_URL),
                    ("SUPABASE_ANON_KEY", self.SUPABASE_ANON_KEY),
                    ("SUPABASE_SERVICE_ROLE_KEY", self.SUPABASE_SERVICE_ROLE_KEY),
                ) if not value.strip()
            ]
            if missing:
                raise ValueError(
                    "AUTH_PROVIDER=supabase requires: " + ", ".join(missing)
                )
            self.SUPABASE_URL = self.SUPABASE_URL.rstrip("/")
        return self

    @model_validator(mode="after")
    def _reject_insecure_production_config(self) -> "Settings":
        """
        Fail closed on an insecure production configuration.

        Starting production with the committed default signing key silently
        produces a system whose tokens anyone can forge. Refusing to boot is far
        safer than logging a warning nobody reads.
        """
        if self.ENVIRONMENT.lower() != "production":
            if self.JWT_SECRET == INSECURE_DEFAULT_JWT_SECRET:
                logger.warning(
                    "[CONFIG] Using the built-in development JWT_SECRET. "
                    "Set JWT_SECRET before deploying to production."
                )
            return self

        if self.JWT_SECRET == INSECURE_DEFAULT_JWT_SECRET:
            raise ValueError(
                "JWT_SECRET is still the built-in development default. "
                "Set a unique JWT_SECRET before starting in production."
            )
        if len(self.JWT_SECRET) < MIN_PRODUCTION_SECRET_LENGTH:
            raise ValueError(
                f"JWT_SECRET must be at least {MIN_PRODUCTION_SECRET_LENGTH} "
                "characters in production."
            )
        return self

settings = Settings()
