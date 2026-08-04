"""
Provider ports for clinical drug data.

The application layer depends on these protocols, never on a concrete vendor.
Swapping openFDA for a licensed source (DrugBank, First Databank, Medi-Span)
means writing one adapter and changing one factory — no business logic moves.

Every method is specified to degrade rather than raise: an unreachable source
returns None or an empty collection, and the caller records the drug as
*unchecked*. Raising here would abort a review that could still have produced
useful findings from the sources that did answer.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from app.rxsafety.domain.entities import DrugConcept, DrugLabel


@runtime_checkable
class DrugNormaliser(Protocol):
    """Resolves free-text drug names onto stable concept identifiers."""

    name: str

    async def normalise(self, drug_name: str) -> DrugConcept:
        """
        Resolve one name. Returns an unresolved `DrugConcept` (rxcui=None) when
        the name is unknown or the service is unreachable — never raises.
        """
        ...

    async def normalise_many(self, drug_names: Sequence[str]) -> list[DrugConcept]:
        ...


@runtime_checkable
class DrugLabelSource(Protocol):
    """Supplies published label content for a resolved drug."""

    name: str

    async def fetch_label(self, concept: DrugConcept) -> DrugLabel | None:
        """None means "could not retrieve", not "no warnings exist"."""
        ...


@runtime_checkable
class InteractionSource(Protocol):
    """
    Reports known interactions between a set of drugs.

    Returns pairs of (rxcui_a, rxcui_b, description, severity). An empty list
    from a source that answered means no known interaction; a source that could
    not answer returns None so the caller can mark the check as incomplete.
    """

    name: str

    async def find_interactions(
        self, concepts: Sequence[DrugConcept]
    ) -> list[tuple[str, str, str, str]] | None:
        ...


@runtime_checkable
class SafetyExplainer(Protocol):
    """
    Turns grounded findings into plain language for patients and clinicians.

    Implementations must not invent findings. They receive the evidence that
    was already gathered and rephrase it; anything they add beyond that is
    recorded with `source='groq'` and no evidence, so the reader can see it is
    model-generated.
    """

    name: str

    async def summarise(
        self,
        *,
        medications: Sequence[str],
        findings_payload: Sequence[dict],
        patient_context: dict,
    ) -> tuple[str, str | None]:
        """Returns (summary_text, model_used). Empty summary on failure."""
        ...
