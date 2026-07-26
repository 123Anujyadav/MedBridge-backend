import logging
import secrets
from typing import Any
from datetime import datetime, timedelta, timezone
import jwt
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
)
from app.models.user import User
from app.repositories.user import user_repository
from app.services.doctor_access import assert_doctor_may_practise
from app.schemas.auth import (
    DoctorSignup,
    LoginRequest,
    PatientSignup,
    ResetPasswordRequest,
    Token,
)

logger = logging.getLogger(__name__)

def _supabase_enabled() -> bool:
    """Whether Supabase is the active identity provider."""
    return settings.AUTH_PROVIDER == "supabase"


class AuthService:
    """
    Credential handling for the platform.

    Under `AUTH_PROVIDER=supabase` every credential operation — registration,
    sign-in, session rotation, sign-out, recovery and email verification — is
    performed by Supabase. This service still owns everything that decides what
    an authenticated person may *do*: the local account, its role, and the
    administrator approval a clinician needs. Request and response shapes are
    identical under both providers, so no API contract changes.
    """

    async def _provision_identity(
        self, email: str, password: str, role: str
    ) -> str | None:
        """
        Create the Supabase identity for a new account.

        Returns the Supabase user id, or None when the built-in provider is
        active. An address that already has an identity but no local account —
        left behind by an interrupted signup — is reused rather than rejected.
        """
        if not _supabase_enabled():
            return None

        from app.core.supabase import (
            SupabaseAuthError, get_supabase_auth_client, translate_auth_error,
        )

        client = get_supabase_auth_client()
        try:
            created = await client.admin_create_user(
                email=email, password=password,
                email_confirm=settings.SUPABASE_AUTOCONFIRM_SIGNUP,
                user_metadata={"role": role},
            )
            return created.get("id")
        except SupabaseAuthError as exc:
            existing = await client.admin_get_user_by_email(email)
            if existing and existing.get("id"):
                logger.info(
                    "[AUTH_SIGNUP_REUSING_IDENTITY] email=%s already had an "
                    "identity with no local account", email,
                )
                return existing["id"]
            raise translate_auth_error(exc)

    async def _discard_identity(self, supabase_user_id: str | None) -> None:
        """Roll back a provisioned identity when the local write fails."""
        if not supabase_user_id or not _supabase_enabled():
            return
        from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

        try:
            await get_supabase_auth_client().admin_delete_user(supabase_user_id)
        except SupabaseAuthError as exc:
            # Logged, not raised: the caller is already handling a failure, and
            # a stranded identity is recoverable while a confusing 500 is not.
            logger.error("[AUTH_SIGNUP_ROLLBACK_FAILED] %s: %s",
                         supabase_user_id, exc)

    async def signup_patient(
        self, db: AsyncSession, signup_data: PatientSignup, redis: Any
    ) -> User:
        """
        Registers a patient account atomically.

        Public registration: a patient is active immediately.
        """
        logger.info(f"[AUTH_SIGNUP_ATTEMPT] Role: patient | Email: {signup_data.email}")
        existing_user = await user_repository.get_by_email(db, signup_data.email)
        if existing_user:
            logger.warning(f"[AUTH_SIGNUP_FAILED] Email already registered: {signup_data.email}")
            raise BusinessRuleValidationException("This email address is already registered.")

        supabase_user_id = await self._provision_identity(
            signup_data.email, signup_data.password, "patient"
        )
        try:
            hashed_password = get_password_hash(signup_data.password)
            user = await user_repository.create_patient_user(
                db, signup_data.email, hashed_password, signup_data.profile,
                supabase_user_id=supabase_user_id,
            )
        except Exception:
            await self._discard_identity(supabase_user_id)
            raise

        await self._issue_verification(user, signup_data.email, redis)
        logger.info(f"[AUTH_SIGNUP_SUCCESS] User ID: {user.id} | Email: {signup_data.email}")
        return user

    async def signup_doctor(
        self, db: AsyncSession, signup_data: DoctorSignup, redis: Any
    ) -> User:
        """
        Registers a doctor account atomically.

        The clinical profile is created `pending`; the account cannot sign in
        until an administrator approves it.
        """
        logger.info(f"[AUTH_SIGNUP_ATTEMPT] Role: doctor | Email: {signup_data.email}")
        existing_user = await user_repository.get_by_email(db, signup_data.email)
        if existing_user:
            logger.warning(f"[AUTH_SIGNUP_FAILED] Email already registered: {signup_data.email}")
            raise BusinessRuleValidationException("This email address is already registered.")

        supabase_user_id = await self._provision_identity(
            signup_data.email, signup_data.password, "doctor"
        )
        try:
            hashed_password = get_password_hash(signup_data.password)
            user = await user_repository.create_doctor_user(
                db, signup_data.email, hashed_password, signup_data.profile,
                supabase_user_id=supabase_user_id,
            )
        except Exception:
            await self._discard_identity(supabase_user_id)
            raise

        await self._issue_verification(user, signup_data.email, redis)
        logger.info(f"[AUTH_SIGNUP_SUCCESS] User ID: {user.id} | Email: {signup_data.email}")
        return user

    async def _issue_verification(self, user: User, email: str, redis: Any) -> None:
        """
        Start email verification.

        Supabase owns this when it is the provider — it sends the message and
        records confirmation. The built-in provider keeps its cached token,
        which is the behaviour the existing verify endpoint expects.
        """
        if _supabase_enabled():
            from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

            try:
                await get_supabase_auth_client().resend_verification(email)
            except SupabaseAuthError as exc:
                # Registration has already succeeded; a mail failure must not
                # undo it. The user can request the email again.
                logger.warning("[AUTH_VERIFICATION_EMAIL_FAILED] %s: %s", email, exc)
            return

        verify_token = secrets.token_hex(32)
        await redis.set(f"verify_token:{verify_token}", str(user.id), ex=86400)

    async def login(
        self, db: AsyncSession, login_data: LoginRequest, redis: Any
    ) -> Token:
        """
        Verifies user credentials and issues Access/Refresh JWT tokens.
        Ensures 100% resilience against Redis downtime using in-memory fallback.
        """
        logger.info(f"[AUTH_LOGIN_ATTEMPT] Email: {login_data.email}")

        if _supabase_enabled():
            return await self._login_via_supabase(db, login_data, redis)

        user = await user_repository.get_by_email(db, login_data.email)
        if not user or not verify_password(login_data.password, user.hashed_password):
            logger.warning(f"[AUTH_LOGIN_FAILED] Invalid credentials for email: {login_data.email}")
            raise AuthenticationException("Incorrect email or password.")

        if not user.is_active:
            logger.warning(f"[AUTH_LOGIN_FAILED] User account deactivated: {user.id}")
            raise AuthorizationException("This user account has been deactivated.")

        # A doctor must be approved by an administrator before they hold a
        # session at all. Refusing here — rather than letting the token issue
        # and blocking at the dashboard — means an unapproved clinician never
        # holds a credential for this platform.
        if user.role == "doctor":
            await assert_doctor_may_practise(db, user)

        # Generate tokens
        access_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = create_refresh_token(
            subject=user.id,
            expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )

        # Store session/active token registry in Redis or In-Memory Fallback Cache
        await redis.set(
            f"active_session:{user.id}",
            refresh_token,
            ex=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
        )

        logger.info(
            f"[AUTH_LOGIN_SUCCESS] User ID: {user.id} | Role: {user.role} | "
            f"Access Token Issued (exp: {settings.ACCESS_TOKEN_EXPIRE_MINUTES}m) | "
            f"Redis Status: {'ONLINE' if getattr(redis, 'is_redis_available', False) else 'FALLBACK_MODE'}"
        )

        return Token(access_token=access_token, refresh_token=refresh_token)

    async def _migrate_existing_account(
        self, db: AsyncSession, login_data: LoginRequest, failure: Any
    ):
        """
        Give a pre-existing account its Supabase identity, at first sign-in.

        Bcrypt hashes cannot be imported into Supabase, so an account that
        predates the identity provider has no credential there. Rather than
        force every established clinician and patient through a password reset,
        the identity is created the first time they sign in successfully —
        *after* the password they typed is verified against the local hash, so
        this grants nothing that the old login would not have granted.

        Returns a Supabase session on success, or None to let the original
        failure stand.
        """
        from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

        if not settings.SUPABASE_MIGRATE_EXISTING_ON_LOGIN:
            return None

        # Only for "no such credential" style failures — never for a wrong
        # password against an identity that already exists.
        message = (getattr(failure, "message", "") or "").lower()
        if "invalid" not in message and "credential" not in message:
            return None

        user = await user_repository.get_by_email(db, login_data.email)
        if user is None or user.supabase_user_id:
            return None
        if not user.hashed_password or not verify_password(
            login_data.password, user.hashed_password
        ):
            return None

        client = get_supabase_auth_client()
        try:
            # Already established here, so the address is treated as proven.
            identity = await client.admin_create_user(
                email=user.email, password=login_data.password,
                email_confirm=True,
                user_metadata={"role": user.role, "migrated": True},
            )
            supabase_user_id = identity.get("id")
        except SupabaseAuthError as exc:
            existing = await client.admin_get_user_by_email(user.email)
            supabase_user_id = (existing or {}).get("id")
            if not supabase_user_id:
                logger.error(
                    "[AUTH_MIGRATION_FAILED] %s: %s", user.email, exc.message
                )
                return None

        user.supabase_user_id = supabase_user_id
        await db.flush()
        logger.info(
            "[AUTH_MIGRATED] local user=%s linked to a new supabase identity "
            "on first sign-in", user.id,
        )

        try:
            return await client.sign_in(login_data.email, login_data.password)
        except SupabaseAuthError as exc:
            logger.error("[AUTH_MIGRATION_SIGNIN_FAILED] %s", exc.message)
            return None

    async def _login_via_supabase(
        self, db: AsyncSession, login_data: LoginRequest, redis: Any
    ) -> Token:
        """
        Authenticate against Supabase, then authorise against this database.

        Supabase proves the credential. Everything after that — does this person
        have an account here, is it active, is the clinician approved — is
        decided locally, because the identity provider knows nothing about
        roles or clinical approval.
        """
        from app.core.identity import VerifiedIdentity
        from app.core.supabase import (
            SupabaseAuthError, get_supabase_auth_client, translate_auth_error,
        )
        from app.services.identity_link import resolve_local_user

        client = get_supabase_auth_client()
        try:
            session = await client.sign_in(login_data.email, login_data.password)
        except SupabaseAuthError as exc:
            session = await self._migrate_existing_account(db, login_data, exc)
            if session is None:
                logger.warning(
                    "[AUTH_LOGIN_FAILED] Supabase rejected %s: %s",
                    login_data.email, exc.message,
                )
                raise translate_auth_error(exc)

        access_token = session.get("access_token")
        refresh_token = session.get("refresh_token")
        supabase_user = session.get("user") or {}
        supabase_user_id = supabase_user.get("id")

        if not access_token or not refresh_token or not supabase_user_id:
            raise AuthenticationException("The authentication service returned an incomplete session.")

        user = await resolve_local_user(db, VerifiedIdentity(
            subject=str(supabase_user_id),
            token_type="access",
            email=supabase_user.get("email") or login_data.email,
            provider="supabase",
        ))
        if user is None:
            # Authenticated with the provider, but not a user of this platform.
            logger.warning(
                "[AUTH_LOGIN_NO_LOCAL_ACCOUNT] email=%s", login_data.email
            )
            raise AuthenticationException("Incorrect email or password.")

        if not user.is_active:
            logger.warning(f"[AUTH_LOGIN_FAILED] User account deactivated: {user.id}")
            raise AuthorizationException("This user account has been deactivated.")

        if user.role == "doctor":
            await assert_doctor_may_practise(db, user)

        await redis.set(
            f"active_session:{user.id}", refresh_token,
            ex=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        )
        logger.info(
            "[AUTH_LOGIN_SUCCESS] User ID: %s | Role: %s | Provider: supabase",
            user.id, user.role,
        )
        return Token(access_token=access_token, refresh_token=refresh_token)

    async def refresh_token(
        self, db: AsyncSession, refresh_token: str, redis: Any
    ) -> Token:
        """
        Validates refresh token, executes token rotation, and issues new tokens.
        Blacklists the old refresh token to prevent replay/abuse.
        """
        logger.info("[AUTH_REFRESH_ATTEMPT]")

        if _supabase_enabled():
            return await self._refresh_via_supabase(db, refresh_token, redis)

        try:
            payload = decode_token(refresh_token)
            user_id = payload.get("sub")
            token_type = payload.get("type")
            exp_timestamp = payload.get("exp")
            
            
            if not user_id or token_type != "refresh":
                raise AuthenticationException("Invalid token type.")
        except jwt.PyJWTError:
            raise AuthenticationException("Invalid or expired refresh token.")

        # Check if the refresh token is blacklisted in Redis or Fallback
        is_revoked = await redis.get(f"revoked_token:{refresh_token}")
        if is_revoked:
            logger.warning(f"[AUTH_REFRESH_FAILED] Revoked token reuse attempted for user {user_id}!")
            raise AuthenticationException("Session expired or token revoked.")

        # Retrieve user account
        user = await user_repository.get(db, user_id)
        if not user or not user.is_active:
            raise AuthenticationException("User account is inactive or not found.")

        # Compute remaining TTL for blacklisting old token
        exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        remaining_ttl = int((exp_time - now).total_seconds())

        if remaining_ttl > 0:
            await redis.set(f"revoked_token:{refresh_token}", "revoked", ex=remaining_ttl)

        # Issue new token pair (Rotation)
        new_access_token = create_access_token(subject=user.id)
        new_refresh_token = create_refresh_token(subject=user.id)

        await redis.set(
            f"active_session:{user.id}",
            new_refresh_token,
            ex=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
        )

        logger.info(f"[AUTH_REFRESH_SUCCESS] User ID: {user.id}")
        return Token(access_token=new_access_token, refresh_token=new_refresh_token)

    async def _refresh_via_supabase(
        self, db: AsyncSession, refresh_token: str, redis: Any
    ) -> Token:
        """
        Rotate the session at Supabase, then re-check local authorisation.

        The re-check matters: an account suspended or a clinician's approval
        withdrawn since the last refresh must not be handed a fresh token.
        """
        from app.core.identity import VerifiedIdentity
        from app.core.supabase import (
            SupabaseAuthError, get_supabase_auth_client, translate_auth_error,
        )
        from app.services.identity_link import resolve_local_user

        try:
            session = await get_supabase_auth_client().refresh_session(refresh_token)
        except SupabaseAuthError as exc:
            logger.warning("[AUTH_REFRESH_FAILED] %s", exc.message)
            raise translate_auth_error(exc)

        access_token = session.get("access_token")
        new_refresh = session.get("refresh_token")
        supabase_user = session.get("user") or {}
        if not access_token or not new_refresh:
            raise AuthenticationException("Session expired or token revoked.")

        user = await resolve_local_user(db, VerifiedIdentity(
            subject=str(supabase_user.get("id")),
            token_type="access",
            email=supabase_user.get("email"),
            provider="supabase",
        ))
        if user is None or not user.is_active:
            raise AuthenticationException("User account is inactive or not found.")

        if user.role == "doctor":
            await assert_doctor_may_practise(db, user)

        await redis.set(
            f"active_session:{user.id}", new_refresh,
            ex=settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        )
        logger.info(f"[AUTH_REFRESH_SUCCESS] User ID: {user.id}")
        return Token(access_token=access_token, refresh_token=new_refresh)

    async def logout(self, refresh_token: str, redis: Any) -> None:
        """
        Revokes a session by blacklisting the associated refresh token.
        """
        logger.info("[AUTH_LOGOUT_ATTEMPT]")

        if _supabase_enabled():
            await self._logout_via_supabase(refresh_token, redis)
            return

        try:
            payload = decode_token(refresh_token)
            token_type = payload.get("type")
            exp_timestamp = payload.get("exp")
            user_id = payload.get("sub")
            
            if token_type != "refresh":
                raise AuthenticationException("Invalid token format for logout.")
                
            exp_time = datetime.fromtimestamp(exp_timestamp, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            remaining_ttl = int((exp_time - now).total_seconds())

            if remaining_ttl > 0:
                await redis.set(f"revoked_token:{refresh_token}", "logged_out", ex=remaining_ttl)
                await redis.delete(f"active_session:{user_id}")
                
            logger.info(f"[AUTH_LOGOUT_SUCCESS] User ID: {user_id}")
        except jwt.PyJWTError:
            raise AuthenticationException("Cannot decode refresh token for logout.")

    async def _logout_via_supabase(self, refresh_token: str, redis: Any) -> None:
        """
        End the Supabase session.

        The refresh token is exchanged for its session first: GoTrue revokes by
        access token, and revoking at the provider is what actually stops the
        session being resumed from another tab or device.
        """
        from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

        client = get_supabase_auth_client()
        try:
            session = await client.refresh_session(refresh_token)
            access_token = session.get("access_token")
            user_id = (session.get("user") or {}).get("id")
            if access_token:
                await client.sign_out(access_token)
            if user_id:
                await redis.delete(f"active_session:{user_id}")
            logger.info("[AUTH_LOGOUT_SUCCESS] Supabase session revoked.")
        except SupabaseAuthError as exc:
            # An already-expired or already-revoked session is a successful
            # logout from the caller's point of view.
            logger.info("[AUTH_LOGOUT_NOOP] %s", exc.message)

    async def request_password_reset(
        self, db: AsyncSession, email: str, redis: Any
    ) -> None:
        """
        Begin password recovery.

        Supabase sends the recovery email when it is the provider. Either way
        the endpoint's response never reveals whether the address is registered.
        """
        if _supabase_enabled():
            from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

            try:
                await get_supabase_auth_client().send_password_reset(email)
                logger.info("[AUTH_RESET_REQUESTED] Recovery email dispatched.")
            except SupabaseAuthError as exc:
                logger.warning("[AUTH_RESET_FAILED] %s", exc.message)
            return

        user = await user_repository.get_by_email(db, email)
        if not user:
            logger.warning(f"Password reset requested for non-existent email: {email}")
            return

        reset_token = secrets.token_hex(32)
        redis_key = f"reset_token:{reset_token}"
        await redis.set(redis_key, str(user.id), ex=900)  # 15 minutes TTL

        # The token itself is never logged: anything with log access could
        # otherwise take over the account it belongs to.
        logger.info(f"Password reset token generated for User {user.id}.")

    async def reset_password(
        self, db: AsyncSession, reset_data: ResetPasswordRequest, redis: Any
    ) -> None:
        """
        Complete a password reset.

        Under Supabase the `token` is the access token the recovery link
        produces: it is verified with the provider, which is what proves the
        person controls the mailbox, and the password is then changed there.
        """
        if _supabase_enabled():
            await self._reset_password_via_supabase(db, reset_data)
            return

        redis_key = f"reset_token:{reset_data.token}"
        user_id = await redis.get(redis_key)
        
        if not user_id:
            raise BusinessRuleValidationException("Invalid or expired password reset token.")

        hashed_password = get_password_hash(reset_data.new_password)
        user = await user_repository.get(db, user_id)
        if not user:
            raise EntityNotFoundException("User", user_id)

        await user_repository.update(db, db_obj=user, obj_in={"hashed_password": hashed_password})
        
        await redis.delete(redis_key)
        await redis.delete(f"active_session:{user_id}")
        
        logger.info(f"Password successfully reset for User {user_id}.")

    async def _reset_password_via_supabase(
        self, db: AsyncSession, reset_data: ResetPasswordRequest
    ) -> None:
        from app.core.supabase import (
            SupabaseAuthError, get_supabase_auth_client, translate_auth_error,
        )

        client = get_supabase_auth_client()
        try:
            identity = await client.get_user(reset_data.token)
        except SupabaseAuthError as exc:
            raise BusinessRuleValidationException(
                "Invalid or expired password reset link."
            ) from exc

        supabase_user_id = identity.get("id")
        if not supabase_user_id:
            raise BusinessRuleValidationException(
                "Invalid or expired password reset link."
            )

        try:
            await client.admin_update_user(
                supabase_user_id, password=reset_data.new_password
            )
        except SupabaseAuthError as exc:
            raise translate_auth_error(exc)

        # Keep the local mirror consistent so a later switch back to the
        # built-in provider does not resurrect the old password.
        user = await user_repository.get_by_supabase_id(db, supabase_user_id)
        if user is None and identity.get("email"):
            user = await user_repository.get_by_email(db, identity["email"])
        if user is not None:
            user.hashed_password = get_password_hash(reset_data.new_password)
            await db.flush()

        logger.info("[AUTH_RESET_SUCCESS] Password updated via Supabase.")

    async def verify_account(self, db: AsyncSession, token: str, redis: Any) -> None:
        """
        Mark an account's email address as verified.

        Supabase owns confirmation when it is the provider: the token is one of
        its access tokens, and the local flag is mirrored from what it reports.
        """
        if _supabase_enabled():
            await self._verify_account_via_supabase(db, token)
            return

        redis_key = f"verify_token:{token}"
        user_id = await redis.get(redis_key)
        
        if not user_id:
            raise BusinessRuleValidationException("Invalid or expired account verification token.")

        user = await user_repository.get(db, user_id)
        if not user:
            raise EntityNotFoundException("User", user_id)

        await user_repository.update(db, db_obj=user, obj_in={"is_verified": True})
        await redis.delete(redis_key)
        
        logger.info(f"User account {user_id} successfully verified.")

    async def _verify_account_via_supabase(self, db: AsyncSession, token: str) -> None:
        from app.core.supabase import SupabaseAuthError, get_supabase_auth_client

        try:
            identity = await get_supabase_auth_client().get_user(token)
        except SupabaseAuthError as exc:
            raise BusinessRuleValidationException(
                "Invalid or expired account verification token."
            ) from exc

        supabase_user_id = identity.get("id")
        confirmed = bool(
            identity.get("email_confirmed_at") or identity.get("confirmed_at")
        )
        if not supabase_user_id or not confirmed:
            raise BusinessRuleValidationException(
                "Invalid or expired account verification token."
            )

        user = await user_repository.get_by_supabase_id(db, supabase_user_id)
        if user is None and identity.get("email"):
            user = await user_repository.get_by_email(db, identity["email"])
        if user is None:
            raise EntityNotFoundException("User", supabase_user_id)

        await user_repository.update(db, db_obj=user, obj_in={"is_verified": True})
        logger.info("[AUTH_VERIFY_SUCCESS] User %s confirmed via Supabase.", user.id)


auth_service = AuthService()
