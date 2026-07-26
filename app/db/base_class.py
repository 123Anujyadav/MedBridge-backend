import re
import uuid
from datetime import datetime
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.types import DateTime

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy declarative database models.
    Enforces UUID v4 primary keys and audit timestamps automatically.
    """
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        default=None,
        nullable=True
    )

    @property
    def is_deleted(self) -> bool:
        """Returns True if the record has been soft-deleted."""
        return self.deleted_at is not None

    def soft_delete(self) -> None:
        """Sets the deleted_at timestamp to mark this record as deleted."""
        from datetime import timezone
        self.deleted_at = datetime.now(timezone.utc)


    @declared_attr.directive
    def __tablename__(cls) -> str:
        """
        Converts ClassName to class_name for table naming conventions.
        """
        name = cls.__name__
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        return pattern.sub('_', name).lower()
