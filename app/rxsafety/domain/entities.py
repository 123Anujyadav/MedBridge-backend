"""Value objects shared across the rxsafety context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from app.models.rx_verification import (
    VERDICT_CRITICAL,
    VERDICT_SAFE,
    VERDICT_SEVERITY,
    VERDICT_UNKNOWN,
    VERDICT_WARNING,
)

Severity = Literal["safe", "warning", "critical", "unknown"]


@dataclass(frozen=True)
class DrugConcept:
    """
    A prescribed drug after normalisation.

    `rxcui` is None when the name could not be resolved. Callers must branch on
    that rather than silently continuing: an unresolved drug has not been
    checked, which is not the same as having been checked and found safe.
    """

    original_name: str
    rxcui: str | None = None
    normalised_name: str | None = None
    ingredients: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return self.rxcui is not None

    @property
    def display_name(self) -> str:
        return self.normalised_name or self.original_name


@dataclass(frozen=True)
class Evidence:
    """A citable excerpt from a source document."""

    source: str
    section: str
    excerpt: str
    reference: str = ""

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "section": self.section,
            "excerpt": self.excerpt,
            "reference": self.reference,
        }


@dataclass
class Finding:
    """One safety observation."""

    category: str
    severity: Severity
    title: str
    detail: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    medications_involved: list[str] = field(default_factory=list)
    source: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """True when the finding cites a source document rather than a model."""
        return bool(self.evidence)


@dataclass
class DrugLabel:
    """
    The subset of an openFDA structured product label the checks care about.

    Every field is a list of free-text sections as published; openFDA returns
    them that way and the wording is what gets cited back to the reader.
    """

    rxcui: str | None
    brand_name: str = ""
    generic_name: str = ""
    drug_interactions: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pregnancy: list[str] = field(default_factory=list)
    geriatric_use: list[str] = field(default_factory=list)
    renal_notes: list[str] = field(default_factory=list)
    hepatic_notes: list[str] = field(default_factory=list)
    dosage_and_administration: list[str] = field(default_factory=list)
    food_effect: list[str] = field(default_factory=list)
    reference: str = ""

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.drug_interactions,
                self.contraindications,
                self.warnings,
                self.pregnancy,
                self.geriatric_use,
                self.renal_notes,
                self.hepatic_notes,
                self.dosage_and_administration,
                self.food_effect,
            )
        )


def worst_verdict(severities: Sequence[str]) -> str:
    """
    The overall verdict is the worst individual severity.

    An empty sequence means nothing was found *and* nothing was checked, which
    is `unknown`, not `safe`. Callers that genuinely checked and found nothing
    pass an explicit `safe`.
    """
    if not severities:
        return VERDICT_UNKNOWN
    return max(severities, key=lambda s: VERDICT_SEVERITY.get(s, 0))


__all__ = [
    "DrugConcept",
    "DrugLabel",
    "Evidence",
    "Finding",
    "Severity",
    "worst_verdict",
    "VERDICT_CRITICAL",
    "VERDICT_WARNING",
    "VERDICT_SAFE",
    "VERDICT_UNKNOWN",
]
