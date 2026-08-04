"""
Deterministic safety checks.

These run without any network call and produce the same answer every time, so
they stay useful when RxNorm and openFDA are both unreachable. They are also
the only checks in this context that are *complete* — a duplicate ingredient is
either present in the prescription or it is not, with no source to be down and
no model to hedge.

Label-derived checks live in `verifier.py`; this module is pure.
"""

from __future__ import annotations

import re
from typing import Sequence

from app.rxsafety.domain.entities import (
    DrugConcept,
    Evidence,
    Finding,
    VERDICT_CRITICAL,
    VERDICT_WARNING,
)

# Terms that mark a label section as relevant to a given patient factor. Matched
# case-insensitively against label text that has already been retrieved, so a
# miss here understates risk rather than inventing it.
RENAL_TERMS = ("renal impairment", "kidney disease", "creatinine clearance", "nephrotox")
HEPATIC_TERMS = ("hepatic impairment", "liver disease", "hepatotox", "cirrhosis")
PREGNANCY_TERMS = ("pregnan", "teratogen", "nursing mothers", "lactation")
ELDERLY_TERMS = ("geriatric", "elderly", "65 years of age and older")

FOOD_INSTRUCTIONS = {
    "before_food": "before food",
    "after_food": "after food",
    "with_food": "with food",
    "empty_stomach": "on an empty stomach",
    "anytime": "at any time",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def check_duplicate_therapy(
    concepts: Sequence[DrugConcept], medication_names: Sequence[str]
) -> list[Finding]:
    """
    Two prescription lines sharing an active ingredient.

    This is the check that catches a brand and its generic ordered together —
    "Crocin" plus "Paracetamol" is a double dose of acetaminophen, and it reads
    as two different drugs to anyone reading the names alone.

    Only resolved concepts participate: without an ingredient list there is
    nothing to compare, and guessing from name similarity would flag
    "Amoxicillin" against "Amoxicillin-Clavulanate" as duplicates when they are
    a legitimate pairing.
    """
    findings: list[Finding] = []
    by_ingredient: dict[str, list[str]] = {}

    for concept in concepts:
        if not concept.resolved:
            continue
        for ingredient in concept.ingredients:
            by_ingredient.setdefault(ingredient, []).append(concept.display_name)

    for ingredient, drugs in by_ingredient.items():
        unique = sorted(set(drugs))
        if len(unique) < 2:
            continue
        findings.append(
            Finding(
                category="duplicate_therapy",
                severity=VERDICT_CRITICAL,
                title=f"Duplicate active ingredient: {ingredient}",
                detail=(
                    f"{' and '.join(unique)} each contain {ingredient}. Taken together "
                    "they deliver more of that ingredient than either line states on "
                    "its own, which can push the daily total past its safe maximum."
                ),
                recommendation=(
                    "Confirm both lines are intended. If they are, the combined daily "
                    "dose of the shared ingredient should be checked against its limit."
                ),
                confidence=0.95,
                medications_involved=unique,
                source="rxnorm",
                evidence=[
                    Evidence(
                        source="rxnorm",
                        section="ingredients",
                        excerpt=(
                            f"RxNorm lists '{ingredient}' as an active ingredient of "
                            f"{', '.join(unique)}."
                        ),
                        reference="https://rxnav.nlm.nih.gov/REST/rxcui/",
                    )
                ],
            )
        )

    return findings


def check_allergies(
    concepts: Sequence[DrugConcept], allergies: Sequence[str]
) -> list[Finding]:
    """
    A prescribed drug matching a recorded patient allergy.

    Matches on ingredient as well as name, so an allergy recorded as
    "penicillin" catches a prescription written for a branded penicillin.
    """
    findings: list[Finding] = []
    recorded = [_norm(a) for a in allergies if _norm(a)]
    if not recorded:
        return findings

    for concept in concepts:
        haystack = {_norm(concept.display_name), _norm(concept.original_name)}
        haystack.update(_norm(i) for i in concept.ingredients)
        haystack.discard("")

        for allergy in recorded:
            hit = any(allergy in term or term in allergy for term in haystack)
            if not hit:
                continue
            findings.append(
                Finding(
                    category="allergy",
                    severity=VERDICT_CRITICAL,
                    title=f"Recorded allergy: {allergy}",
                    detail=(
                        f"{concept.display_name} matches an allergy recorded on this "
                        f"patient's profile ('{allergy}')."
                    ),
                    recommendation=(
                        "Verify the recorded allergy with the patient before dispensing."
                    ),
                    confidence=0.9,
                    medications_involved=[concept.display_name],
                    source="rules",
                    evidence=[
                        Evidence(
                            source="patient_record",
                            section="allergies",
                            excerpt=f"Patient profile lists an allergy to '{allergy}'.",
                        )
                    ],
                )
            )
            break

    return findings


def check_unresolved(concepts: Sequence[DrugConcept]) -> list[Finding]:
    """
    Report drugs no source could identify.

    Without this the reader sees a short list of findings and reasonably infers
    the rest of the prescription was checked and cleared. An unresolved drug was
    not checked at all, and saying so is the difference between a partial review
    and a misleading one.
    """
    unresolved = [c.original_name for c in concepts if not c.resolved]
    if not unresolved:
        return []

    return [
        Finding(
            category="contraindication",
            severity="unknown",
            title="Some medicines could not be identified",
            detail=(
                f"{', '.join(unresolved)} could not be matched to a known drug record, "
                "so no interaction, contraindication or dosage check was run against "
                f"{'them' if len(unresolved) > 1 else 'it'}."
            ),
            recommendation=(
                "Check the spelling against the prescription. These medicines still "
                "need a manual review."
            ),
            confidence=1.0,
            medications_involved=unresolved,
            source="rules",
        )
    ]


def check_food_instructions(medications: Sequence[dict]) -> list[Finding]:
    """
    Flag lines carrying no food instruction.

    Low severity by design: a missing instruction is an incompleteness in the
    order, not a hazard. It is surfaced because the patient-facing card has a
    slot for it and an empty slot reads as "no restriction" when the truth is
    "not stated".
    """
    missing = [
        m.get("name", "")
        for m in medications
        if not (m.get("food_instruction") or "").strip()
    ]
    if not missing:
        return []

    return [
        Finding(
            category="food_interaction",
            severity=VERDICT_WARNING,
            title="Food instructions not stated",
            detail=(
                f"No food timing was recorded for {', '.join(n for n in missing if n)}. "
                "Absorption of some drugs changes substantially with food."
            ),
            recommendation="Ask the prescriber to confirm food timing for these lines.",
            confidence=1.0,
            medications_involved=[n for n in missing if n],
            source="rules",
        )
    ]


def label_mentions(sections: Sequence[str], terms: Sequence[str]) -> str | None:
    """Return the first section mentioning any term, else None."""
    for section in sections:
        lowered = _norm(section)
        if any(term in lowered for term in terms):
            return section
    return None
