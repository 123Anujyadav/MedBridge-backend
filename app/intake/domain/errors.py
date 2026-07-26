"""
Domain exceptions for the Medical Case Intake Agent.

Each one extends the project exception whose registered handler already carries
the right HTTP semantics, so intake errors map to correct status codes through
`app/middleware/exceptions.py` with no new handlers:

    SessionNotFoundError      -> EntityNotFoundException          -> 404
    InvalidSessionStateError  -> BusinessRuleValidationException  -> 422
    EvidenceNotGroundedError  -> AronofyException                 -> 500

`EvidenceNotGroundedError` stays a plain domain error deliberately: it signals an
internal safety-invariant breach, not a client mistake, and is never raised on
the request path (grounding failures are filtered and logged, not thrown).
"""

from app.core.exceptions import (
    AronofyException,
    BusinessRuleValidationException,
    EntityNotFoundException,
)


class DomainError(AronofyException):
    """Base class for intake domain rule violations."""


class SessionNotFoundError(EntityNotFoundException):
    """Raised when an intake session id cannot be resolved (or has expired)."""

    def __init__(self, session_id: str) -> None:
        super().__init__("IntakeSession", session_id)
        self.session_id = session_id


class InvalidSessionStateError(BusinessRuleValidationException):
    """Raised when an operation is illegal for the session's current status."""

    def __init__(self, session_id: str, current: str, expected: str) -> None:
        super().__init__(
            f"Intake session is in state '{current}'; "
            f"this operation requires state '{expected}'."
        )
        self.session_id = session_id
        self.current = current
        self.expected = expected


class EvidenceNotGroundedError(DomainError):
    """
    Raised when an extracted entity cites evidence absent from the transcript.

    The hard stop against fabricated clinical data.
    """

    def __init__(self, value: str, quote: str) -> None:
        super().__init__(
            f"Extracted value '{value}' cited evidence not present in the "
            f"patient transcript: '{quote}'."
        )
        self.value = value
        self.quote = quote
