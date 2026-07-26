"""
Immutable value objects for the Medical Case Intake Agent.

Frozen dataclasses rather than Pydantic models: the domain layer stays free of
framework imports. Pydantic is used at the boundaries (API schemas in
`app/schemas/intake_api.py`, LLM output parsing in `app/intake/workflow`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.intake.domain.enums import ConfidenceBand

# Score at or above which we treat an extraction as clinically dependable.
_HIGH_BAND_FLOOR = 0.80
# Score at or above which an extraction is usable but worth confirming.
_MEDIUM_BAND_FLOOR = 0.55


@dataclass(frozen=True, slots=True)
class Confidence:
    """
    A model's self-reported certainty about one extracted value, clamped to
    [0.0, 1.0].

    A score of exactly 0.0 means "not stated by the patient" and pairs with the
    UNKNOWN band. We never let an unstated field masquerade as a low-confidence
    guess.
    """

    score: float

    def __post_init__(self) -> None:
        # Clamp defensively: the score originates from an LLM and cannot be trusted
        # to respect its own schema.
        clamped = min(1.0, max(0.0, float(self.score)))
        object.__setattr__(self, "score", round(clamped, 4))

    @property
    def band(self) -> ConfidenceBand:
        if self.score <= 0.0:
            return ConfidenceBand.UNKNOWN
        if self.score >= _HIGH_BAND_FLOOR:
            return ConfidenceBand.HIGH
        if self.score >= _MEDIUM_BAND_FLOOR:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW

    @property
    def is_unknown(self) -> bool:
        return self.score <= 0.0

    def meets(self, threshold: float) -> bool:
        """True when this score satisfies a minimum confidence requirement."""
        return self.score >= threshold

    def penalised(self, factor: float) -> Confidence:
        """
        Return a downgraded copy.

        Used when an extraction survives grounding checks only partially — we
        keep the data but reduce trust in it rather than discarding silently.
        """
        return Confidence(self.score * factor)

    @classmethod
    def unknown(cls) -> Confidence:
        return cls(0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "band": self.band.value}

    @classmethod
    def from_dict(cls, raw: Any) -> Confidence:
        if isinstance(raw, (int, float)):
            return cls(float(raw))
        if isinstance(raw, dict):
            return cls(float(raw.get("score", 0.0)))
        return cls.unknown()


@dataclass(frozen=True, slots=True)
class Evidence:
    """
    Provenance for an extracted value: the verbatim span of patient text the
    extraction was drawn from, plus which conversation turn it came from.

    Every clinical entity must carry one. An entity whose evidence cannot be
    located in the transcript is treated as fabricated and dropped — see
    `policies.is_evidence_grounded`.
    """

    quote: str
    turn_index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "quote", (self.quote or "").strip())
        object.__setattr__(self, "turn_index", max(0, int(self.turn_index)))

    @property
    def is_empty(self) -> bool:
        return not self.quote

    def to_dict(self) -> dict[str, Any]:
        return {"quote": self.quote, "turn_index": self.turn_index}

    @classmethod
    def from_dict(cls, raw: Any) -> Evidence:
        if isinstance(raw, str):
            return cls(quote=raw)
        if isinstance(raw, dict):
            return cls(
                quote=str(raw.get("quote", "")),
                turn_index=int(raw.get("turn_index", 0) or 0),
            )
        return cls(quote="")
