import uuid
from datetime import datetime
from enum import StrEnum
from typing import List, Optional
from pydantic import BaseModel, Field


class AuditAction(StrEnum):
    """
    Closed vocabulary of client-reportable audit actions.

    Restricting this to an enum stops a caller writing arbitrary text into the
    compliance trail (e.g. logging a fabricated "ADMIN_APPROVED_TRANSFER").
    """

    VIEW_DASHBOARD = "VIEW_DASHBOARD"
    VIEW_REPORT = "VIEW_REPORT"
    DOWNLOAD_REPORT = "DOWNLOAD_REPORT"
    VIEW_PRESCRIPTION = "VIEW_PRESCRIPTION"
    VIEW_CASE = "VIEW_CASE"
    VIEW_PATIENT_RECORD = "VIEW_PATIENT_RECORD"
    VIEW_APPOINTMENT = "VIEW_APPOINTMENT"
    VIEW_MEDICAL_HISTORY = "VIEW_MEDICAL_HISTORY"
    VIEW_NOTIFICATION = "VIEW_NOTIFICATION"
    EXPORT_DATA = "EXPORT_DATA"

    # Deliberately read-only. Write events (prescriptions issued, consent
    # changed, records modified) are logged server-side where the write happens
    # — a client must never be able to assert that a mutation occurred.

class UploadResponse(BaseModel):
    filename: str
    file_url: str
    content_type: str
    size_bytes: int

class NotificationResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    type: str
    title: str
    message: str
    timestamp: str
    read: bool
    priority: str
    action_url: Optional[str] = None
    action_label: Optional[str] = None

    class Config:
        from_attributes = True

class NotificationCard(BaseModel):
    """One actionable notification, with the context its action link needs."""

    id: str
    type: str
    category: str
    title: str
    message: str
    priority: str
    timestamp: str
    read: bool
    archived: bool
    action_url: Optional[str] = None
    action_label: Optional[str] = None
    case_id: Optional[str] = None
    case_short_id: Optional[str] = None
    """First segment of the case UUID — enough to identify it on a card."""
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    group_key: Optional[str] = None
    read_at: Optional[str] = None
    delivered_at: Optional[str] = None


class NotificationGroup(BaseModel):
    """Two or more similar unread notifications, collapsed."""

    group_key: str
    category: str
    label: str
    count: int
    highest_priority: str


class NotificationCenterResponse(BaseModel):
    total: int
    returned: int
    skip: int
    limit: int
    has_more: bool
    unread_count: int
    critical_count: int
    groups: List[NotificationGroup] = []
    notifications: List[NotificationCard] = []


class MarkNotificationsRequest(BaseModel):
    notification_ids: List[uuid.UUID] = Field(min_length=1, max_length=500)


class CalendarEventResponse(BaseModel):
    id: uuid.UUID
    title: str
    date: str
    time: str
    duration: int
    type: str  # e.g., "appointment"
    status: str
    description: Optional[str] = None

class TimelineEventResponse(BaseModel):
    id: uuid.UUID
    type: str  # e.g., "case_created", "notes_added", "prescription_written", "report_filed"
    title: str
    description: str
    timestamp: datetime


class CaseTimelineEvent(BaseModel):
    """
    One entry in a case's history.

    `source` distinguishes an event recorded when it happened — which can prove
    its actor and before/after values — from a milestone derived from a clinical
    row's own timestamp. Both are real; only the first has an attributable actor.
    """

    id: str
    event_type: str
    category: str
    title: str
    description: str = ""
    timestamp: str
    actor_type: str
    actor_label: str
    """Display role: Doctor, Patient, AI Assistant, Administrator, System."""
    actor_name: str
    field_changed: Optional[str] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    reason: Optional[str] = None
    source: str
    """`recorded` or `derived`."""


class CaseTimelineResponse(BaseModel):
    case_id: uuid.UUID
    total: int
    returned: int
    skip: int
    limit: int
    has_more: bool
    events: List[CaseTimelineEvent] = []

class ClientAuditLogRequest(BaseModel):
    """
    Client-reported *context* for an audit event.

    The client may only say WHICH resource it viewed. It cannot set the actor,
    the IP address, the timestamp, or a free-text action string — those are
    derived server-side, because an audit trail a client can forge is not an
    audit trail.
    """

    action: AuditAction
    resource: str = Field(min_length=1, max_length=100)
    resource_id: str = Field(min_length=1, max_length=100)
    details: str = Field(default="", max_length=500)

class FeedbackRequest(BaseModel):
    category: str = Field(min_length=1, max_length=100)
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)

class SettingsResponse(BaseModel):
    theme: str
    notifications_enabled: bool
    email_notifications: bool
    marketing_emails: bool

class SettingsUpdateRequest(BaseModel):
    theme: Optional[str] = Field(None, pattern="^(light|dark)$")
    notifications_enabled: Optional[bool] = None
    email_notifications: Optional[bool] = None
    marketing_emails: Optional[bool] = None

class SearchResult(BaseModel):
    id: uuid.UUID
    type: str  # "doctor" or "hospital"
    name: str
    details: str  # specialty or address
    city: str
    state: str
