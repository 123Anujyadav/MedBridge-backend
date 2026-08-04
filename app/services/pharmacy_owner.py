"""
Pharmacy owner provisioning.

The last link between a verified pharmacy and a person who can operate it.
Administrators create or attach a `role='pharmacy'` account and point its
`pharmacy_id` at a store; from then on the Phase 5 portal derives everything
from that user row.

Reuses the existing authentication end to end — `get_password_hash`, the same
`users` table, the same tokens, the same `RoleChecker`. Nothing here mints a
credential, issues a token, or introduces a second identity system.

Four invariants the service enforces rather than trusting callers with:

1. **One active owner per pharmacy.** Two people holding the same store's
   inventory is an accountability hole, not a convenience.
2. **Only an approved, active pharmacy may be staffed.** Handing out portal
   access to a store that has not cleared document review would let it dispense
   before anyone checked its licence.
3. **An owner is never removed while their store has orders in flight.** The
   patient waiting on those orders needs somebody able to act on them.
4. **Every action is audited** through the same trail the admin console reads.
"""

from __future__ import annotations

import logging
import secrets
import string
import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    BusinessRuleValidationException,
    EntityNotFoundException,
)
from app.core.security import get_password_hash
from app.models.medicine_order import MedicineOrder
from app.models.pharmacy import Pharmacy, VERIFICATION_APPROVED
from app.models.user import User
from app.services.notifications import notification_service
from app.services.pharmacy_admin import pharmacy_admin_service

logger = logging.getLogger(__name__)

OWNER_ROLE = "pharmacy"

# Orders nobody could act on if the store lost its operator.
IN_FLIGHT_STATUSES = ("received", "preparing", "packed", "out_for_delivery")

TEMP_PASSWORD_LENGTH = 14
# Excludes characters that are misread when a credential is dictated over the
# phone or copied off a screen — O/0, l/1/I.
_PASSWORD_ALPHABET = (
    "".join(c for c in string.ascii_letters if c not in "lIO")
    + "".join(c for c in string.digits if c not in "01")
    + "!@#$%^&*"
)


def generate_temporary_password() -> str:
    """A one-time credential, shown to the administrator exactly once."""
    return "".join(secrets.choice(_PASSWORD_ALPHABET) for _ in range(TEMP_PASSWORD_LENGTH))


class PharmacyOwnerService:
    # ── lookups ──────────────────────────────────────────────────────────

    async def _pharmacy(self, db: AsyncSession, pharmacy_id: uuid.UUID) -> Pharmacy:
        pharmacy = await db.get(Pharmacy, pharmacy_id)
        if not pharmacy or pharmacy.deleted_at is not None:
            raise EntityNotFoundException("Pharmacy", str(pharmacy_id))
        return pharmacy

    async def _assert_staffable(self, pharmacy: Pharmacy) -> None:
        """
        A store must be approved and active before anyone can be given its keys.

        Checked here rather than at the endpoint so every path — assign, create,
        change — is covered by one rule.
        """
        if pharmacy.verification_status != VERIFICATION_APPROVED:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} is '{pharmacy.verification_status}' and has not been "
                "approved. Complete verification before assigning an owner."
            )
        if not pharmacy.is_active:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} is suspended. Reactivate it before assigning an owner."
            )

    async def list_owners(
        self, db: AsyncSession, pharmacy_id: uuid.UUID
    ) -> list[User]:
        """Everyone linked to this store, active or not."""
        result = await db.execute(
            select(User)
            .where(User.pharmacy_id == pharmacy_id, User.role == OWNER_ROLE)
            .order_by(User.created_at)
        )
        return list(result.scalars().all())

    async def current_owner(
        self, db: AsyncSession, pharmacy_id: uuid.UUID
    ) -> User | None:
        result = await db.execute(
            select(User).where(
                User.pharmacy_id == pharmacy_id,
                User.role == OWNER_ROLE,
                User.is_active.is_(True),
            )
        )
        return result.scalars().first()

    async def _assert_no_active_owner(
        self, db: AsyncSession, pharmacy: Pharmacy
    ) -> None:
        existing = await self.current_owner(db, pharmacy.id)
        if existing:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} already has an active owner ({existing.email}). "
                "Change or remove them first."
            )

    # ── assignment ───────────────────────────────────────────────────────

    async def assign_existing_user(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        actor: User,
        ip: str = "",
    ) -> User:
        """
        Promote an existing account to operate this pharmacy.

        Refuses to convert a patient, doctor or administrator. Their existing
        records reference them under that role — silently changing it would
        orphan a doctor's cases or an admin's audit trail, and is never what an
        operator meant by "link this account".
        """
        pharmacy = await self._pharmacy(db, pharmacy_id)
        await self._assert_staffable(pharmacy)
        await self._assert_no_active_owner(db, pharmacy)

        user = await db.get(User, user_id)
        if not user:
            raise EntityNotFoundException("User", str(user_id))

        if user.role not in (OWNER_ROLE, None, ""):
            raise BusinessRuleValidationException(
                f"{user.email} is a '{user.role}' account and cannot be converted. "
                "Create a dedicated pharmacy account instead."
            )

        if user.pharmacy_id and user.pharmacy_id != pharmacy_id:
            raise BusinessRuleValidationException(
                f"{user.email} already operates another pharmacy."
            )

        previous_role = user.role
        user.role = OWNER_ROLE
        user.pharmacy_id = pharmacy_id
        user.is_active = True

        pharmacy_admin_service._audit(
            db, actor=actor, action="PHARMACY_OWNER_ASSIGNED",
            resource_id=str(pharmacy_id),
            details=f"Linked {user.email} to {pharmacy.name}.",
            field="pharmacy_id", previous=None, new=str(pharmacy_id), ip=ip,
        )
        pharmacy_admin_service._audit(
            db, actor=actor, action="PHARMACY_OWNER_ASSIGNED",
            resource="User", resource_id=str(user.id),
            details=f"Granted pharmacy access for {pharmacy.name}.",
            field="role", previous=previous_role, new=OWNER_ROLE, ip=ip,
        )

        await db.flush()
        logger.info(
            "[PHARMACY_OWNER_ASSIGNED] user=%s pharmacy=%s by=%s",
            user.id, pharmacy_id, actor.id,
        )
        return user

    async def create_owner(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        email: str,
        actor: User,
        password: str | None = None,
        ip: str = "",
    ) -> tuple[User, str]:
        """
        Create a dedicated pharmacy account.

        Returns `(user, temporary_password)`. The password is returned exactly
        once, in the response to this call, and only its hash is stored — there
        is no path that can read it back afterwards, which is why the endpoint
        surfaces it to the administrator rather than logging or emailing it.
        """
        pharmacy = await self._pharmacy(db, pharmacy_id)
        await self._assert_staffable(pharmacy)
        await self._assert_no_active_owner(db, pharmacy)

        normalised = (email or "").strip().lower()
        if not normalised:
            raise BusinessRuleValidationException("An email address is required.")

        clash = await db.scalar(
            select(func.count()).select_from(User).where(
                func.lower(User.email) == normalised
            )
        )
        if clash:
            raise BusinessRuleValidationException(
                f"An account already exists for {normalised}. Link it instead of "
                "creating a second one."
            )

        temporary = password or generate_temporary_password()
        user = User(
            id=uuid.uuid4(),
            email=normalised,
            hashed_password=get_password_hash(temporary),
            role=OWNER_ROLE,
            is_active=True,
            # Not `is_verified`: the account exists but nobody has proved they
            # control the mailbox. Login does not depend on it, and leaving it
            # false keeps that distinction honest.
            is_verified=False,
            pharmacy_id=pharmacy_id,
        )
        db.add(user)
        await db.flush()

        pharmacy_admin_service._audit(
            db, actor=actor, action="PHARMACY_OWNER_CREATED",
            resource_id=str(pharmacy_id),
            details=f"Created owner account {normalised} for {pharmacy.name}.",
            field="pharmacy_id", previous=None, new=str(pharmacy_id), ip=ip,
        )
        logger.info(
            "[PHARMACY_OWNER_CREATED] user=%s pharmacy=%s by=%s",
            user.id, pharmacy_id, actor.id,
        )
        return user, temporary

    async def change_owner(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        new_user_id: uuid.UUID,
        *,
        actor: User,
        reason: str = "",
        ip: str = "",
    ) -> User:
        """
        Hand a store to a different operator.

        Done as remove-then-assign in one transaction so the store is never
        left with two active owners, and never with none if the second half
        fails.
        """
        pharmacy = await self._pharmacy(db, pharmacy_id)
        await self._assert_staffable(pharmacy)

        outgoing = await self.current_owner(db, pharmacy_id)
        if outgoing and str(outgoing.id) == str(new_user_id):
            raise BusinessRuleValidationException(
                f"{outgoing.email} already operates {pharmacy.name}."
            )

        if outgoing:
            await self._detach(
                db, outgoing, pharmacy, actor=actor,
                reason=reason or "Replaced by a new owner", ip=ip,
                action="PHARMACY_OWNER_REPLACED",
            )
            # Flushed before the assignment below, which re-queries for an
            # active owner. Without this the detach is still pending and that
            # query would find the outgoing owner, refusing the very handover
            # this method exists to perform.
            await db.flush()

        return await self.assign_existing_user(
            db, pharmacy_id, new_user_id, actor=actor, ip=ip
        )

    # ── removal & lifecycle ──────────────────────────────────────────────

    async def _detach(
        self,
        db: AsyncSession,
        user: User,
        pharmacy: Pharmacy,
        *,
        actor: User,
        reason: str,
        ip: str,
        action: str = "PHARMACY_OWNER_REMOVED",
    ) -> None:
        """
        Revoke access.

        The account is deactivated and unlinked rather than deleted: it is
        referenced by audit rows and by the order events it recorded, and those
        must stay readable. `require_verified_pharmacy` rejects it on the very
        next request because both `is_active` and `pharmacy_id` now fail.
        """
        previous_pharmacy = str(user.pharmacy_id) if user.pharmacy_id else None
        user.pharmacy_id = None
        user.is_active = False

        pharmacy_admin_service._audit(
            db, actor=actor, action=action, resource_id=str(pharmacy.id),
            details=f"Revoked {user.email} from {pharmacy.name}. {reason}".strip(),
            field="pharmacy_id", previous=previous_pharmacy, new=None, ip=ip,
        )
        pharmacy_admin_service._audit(
            db, actor=actor, action=action, resource="User",
            resource_id=str(user.id),
            details=reason or "Pharmacy access revoked.",
            field="is_active", previous=True, new=False, ip=ip,
        )

    async def remove_owner(
        self,
        db: AsyncSession,
        pharmacy_id: uuid.UUID,
        *,
        actor: User,
        reason: str = "",
        ip: str = "",
    ) -> None:
        """Revoke the current owner, refusing while orders are in flight."""
        pharmacy = await self._pharmacy(db, pharmacy_id)
        owner = await self.current_owner(db, pharmacy_id)
        if not owner:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} has no active owner to remove."
            )

        in_flight = await db.scalar(
            select(func.count()).select_from(MedicineOrder).where(
                MedicineOrder.pharmacy_id == pharmacy_id,
                MedicineOrder.status.in_(IN_FLIGHT_STATUSES),
            )
        )
        if in_flight:
            raise BusinessRuleValidationException(
                f"{pharmacy.name} has {in_flight} order(s) in progress. Removing the "
                "owner now would leave nobody able to dispatch them. Complete or "
                "cancel them first."
            )

        await self._detach(
            db, owner, pharmacy, actor=actor,
            reason=reason or "Removed by administrator", ip=ip,
        )
        await db.flush()
        logger.info(
            "[PHARMACY_OWNER_REMOVED] user=%s pharmacy=%s by=%s",
            owner.id, pharmacy_id, actor.id,
        )

    async def set_owner_active(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        active: bool,
        actor: User,
        reason: str = "",
        ip: str = "",
    ) -> User:
        """
        Suspend or reactivate an owner without unlinking them.

        Distinct from removal: the link survives, so reactivating restores
        access without re-assigning the store. Suspension takes effect on the
        next request — `require_verified_pharmacy` reads `is_active` live.
        """
        user = await db.get(User, user_id)
        if not user or user.role != OWNER_ROLE:
            raise EntityNotFoundException("Pharmacy owner", str(user_id))

        if active and not user.pharmacy_id:
            raise BusinessRuleValidationException(
                f"{user.email} is not linked to a pharmacy. Assign them to one first."
            )

        if active:
            # Reactivating must not create a second active owner.
            pharmacy = await self._pharmacy(db, user.pharmacy_id)
            existing = await self.current_owner(db, pharmacy.id)
            if existing and existing.id != user.id:
                raise BusinessRuleValidationException(
                    f"{pharmacy.name} already has an active owner ({existing.email})."
                )

        previous = user.is_active
        user.is_active = active

        pharmacy_admin_service._audit(
            db, actor=actor,
            action="PHARMACY_OWNER_ACTIVATED" if active else "PHARMACY_OWNER_SUSPENDED",
            resource="User", resource_id=str(user.id),
            details=reason or ("Reactivated." if active else "Suspended."),
            field="is_active", previous=previous, new=active, ip=ip,
        )
        await db.flush()
        logger.info(
            "[PHARMACY_OWNER_%s] user=%s by=%s",
            "ACTIVATED" if active else "SUSPENDED", user.id, actor.id,
        )
        return user

    # ── credentials ──────────────────────────────────────────────────────

    async def reset_password(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        actor: User,
        ip: str = "",
    ) -> tuple[User, str]:
        """
        Issue a new temporary password.

        Returns it once. Only the hash is written, and the audit row records
        that a reset happened without ever carrying the value — an audit trail
        readable by other administrators must not contain live credentials.
        """
        user = await db.get(User, user_id)
        if not user or user.role != OWNER_ROLE:
            raise EntityNotFoundException("Pharmacy owner", str(user_id))

        temporary = generate_temporary_password()
        user.hashed_password = get_password_hash(temporary)

        pharmacy_admin_service._audit(
            db, actor=actor, action="PHARMACY_OWNER_PASSWORD_RESET",
            resource="User", resource_id=str(user.id),
            details=f"Temporary password issued for {user.email}.", ip=ip,
        )
        await db.flush()
        logger.info("[PHARMACY_OWNER_PASSWORD_RESET] user=%s by=%s", user.id, actor.id)
        return user, temporary

    async def send_invitation(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        *,
        actor: User,
        ip: str = "",
        redis=None,
    ) -> dict:
        """
        Invite an owner into the portal.

        Delivered as an in-app notification, which is the only channel that
        actually works: `send_email_task` is a stub that logs and returns True
        without contacting a mail server, so reporting "invitation emailed"
        would be false. A fresh temporary password is returned to the
        administrator to pass on out of band.
        """
        user = await db.get(User, user_id)
        if not user or user.role != OWNER_ROLE:
            raise EntityNotFoundException("Pharmacy owner", str(user_id))
        if not user.pharmacy_id:
            raise BusinessRuleValidationException(
                f"{user.email} is not linked to a pharmacy yet."
            )

        pharmacy = await self._pharmacy(db, user.pharmacy_id)
        _, temporary = await self.reset_password(db, user_id, actor=actor, ip=ip)

        await notification_service.notify(
            db,
            user_id=user.id,
            category="system",
            type="pharmacy_invitation",
            title=f"You now manage {pharmacy.name} on MedBridge",
            message=(
                "Your pharmacy portal account is ready. Sign in with the temporary "
                "password your administrator has shared and change it immediately."
            ),
            priority="high",
            action_url="/pharmacy/dashboard",
            action_label="Open pharmacy portal",
            dedupe_key=f"pharmacy-invite-{user.id}",
            redis=redis,
        )

        pharmacy_admin_service._audit(
            db, actor=actor, action="PHARMACY_OWNER_INVITED",
            resource="User", resource_id=str(user.id),
            details=f"Invitation issued for {pharmacy.name}.", ip=ip,
        )
        await db.flush()
        logger.info("[PHARMACY_OWNER_INVITED] user=%s by=%s", user.id, actor.id)

        return {
            "user_id": str(user.id),
            "email": user.email,
            "pharmacy_name": pharmacy.name,
            "temporary_password": temporary,
            "portal_url": "/pharmacy/dashboard",
            # Stated plainly so the caller does not assume delivery happened.
            "delivery": "in_app_notification",
            "email_sent": False,
        }


pharmacy_owner_service = PharmacyOwnerService()
