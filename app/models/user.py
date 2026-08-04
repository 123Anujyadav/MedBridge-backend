import uuid

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base_class import Base

class User(Base):
    """
    Core authentication and role authorization model.
    """
    __tablename__ = "users"  # Explicitly override base table naming
    __table_args__ = (
        # `pharmacy` joins the existing roles rather than starting a second
        # identity system: the same login, the same token, the same RoleChecker.
        CheckConstraint(
            "role IN ('patient', 'doctor', 'admin', 'pharmacy', 'delivery_partner')",
            name="user_role_check",
        ),
    )


    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )
    
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    supabase_user_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=True
    )
    """
    The Supabase Auth user this account is linked to.

    Null for accounts created before the identity provider was introduced, and
    for any account that has never signed in through it. The link is made on
    first Supabase sign-in by matching the verified email, so existing users
    keep their id, their role and every record that references them — a second
    account is never created for someone who already exists here.
    """
    
    role: Mapped[str] = mapped_column(
        String(50),
        default="patient",
        nullable=False
    )

    pharmacy_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("pharmacies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    """
    The store this account operates, for `role='pharmacy'` accounts.

    Null for every other role, and the reason the portal needs no store id in
    any request: the pharmacy an owner may act on is read from their own user
    row, so there is no path parameter to tamper with to reach someone else's
    inventory or orders.

    SET NULL rather than CASCADE — retiring a pharmacy must not delete the
    person who ran it, since their audit trail references that account.
    """
    
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )
    
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    # Relationships
    patient = relationship(
        "Patient",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    doctor = relationship(
        "Doctor",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    audit_logs = relationship(
        "AuditLog",
        back_populates="user"
    )
    
    notifications = relationship(
        "NotificationItem",
        back_populates="user",
        cascade="all, delete-orphan"
    )
