"""
Canonical medical specialties and mapping rules.

Free-text specialty names from an LLM are unreliable ("heart doctor",
"Cardiology Dept", "cardiac"). Everything is funnelled through
`canonicalise_specialty` so the value stored on a case and the value used to
query the doctor directory are always drawn from one closed vocabulary.
"""

from __future__ import annotations

import re
import unicodedata

GENERAL_MEDICINE = "General Medicine"
EMERGENCY_MEDICINE = "Emergency Medicine"

CANONICAL_SPECIALTIES: tuple[str, ...] = (
    "Cardiology",
    "Neurology",
    "Gastroenterology",
    "Pulmonology",
    "Dermatology",
    "Orthopedics",
    "Endocrinology",
    "Nephrology",
    "Psychiatry",
    "Otolaryngology",
    "Ophthalmology",
    "Gynecology",
    "Urology",
    "Pediatrics",
    "Oncology",
    "Rheumatology",
    EMERGENCY_MEDICINE,
    GENERAL_MEDICINE,
)

# Lay phrasing and common variants -> canonical name.
_ALIASES: dict[str, str] = {
    "cardiac": "Cardiology",
    "cardiologist": "Cardiology",
    "heart": "Cardiology",
    "heart doctor": "Cardiology",
    "cardiovascular": "Cardiology",
    "cardiothoracic": "Cardiology",
    "neurologist": "Neurology",
    "brain": "Neurology",
    "nerve": "Neurology",
    "neuro": "Neurology",
    "gastro": "Gastroenterology",
    "gastroenterologist": "Gastroenterology",
    "stomach": "Gastroenterology",
    "digestive": "Gastroenterology",
    "gi": "Gastroenterology",
    "pulmonologist": "Pulmonology",
    "lung": "Pulmonology",
    "respiratory": "Pulmonology",
    "chest medicine": "Pulmonology",
    "dermatologist": "Dermatology",
    "skin": "Dermatology",
    "orthopedic": "Orthopedics",
    "orthopaedics": "Orthopedics",
    "orthopaedic": "Orthopedics",
    "bone": "Orthopedics",
    "joint": "Orthopedics",
    "endocrinologist": "Endocrinology",
    "diabetes": "Endocrinology",
    "thyroid": "Endocrinology",
    "hormone": "Endocrinology",
    "nephrologist": "Nephrology",
    "kidney": "Nephrology",
    "renal": "Nephrology",
    "psychiatrist": "Psychiatry",
    "mental health": "Psychiatry",
    "psychological": "Psychiatry",
    "psychiatric": "Psychiatry",
    "ent": "Otolaryngology",
    "ear nose throat": "Otolaryngology",
    "ear nose and throat": "Otolaryngology",
    "otorhinolaryngology": "Otolaryngology",
    "ophthalmologist": "Ophthalmology",
    "eye": "Ophthalmology",
    "vision": "Ophthalmology",
    "gynaecology": "Gynecology",
    "gynecologist": "Gynecology",
    "obstetrics": "Gynecology",
    "obgyn": "Gynecology",
    "womens health": "Gynecology",
    "urologist": "Urology",
    "bladder": "Urology",
    "pediatrician": "Pediatrics",
    "paediatrics": "Pediatrics",
    "paediatric": "Pediatrics",
    "child": "Pediatrics",
    "oncologist": "Oncology",
    "cancer": "Oncology",
    "tumour": "Oncology",
    "rheumatologist": "Rheumatology",
    "arthritis": "Rheumatology",
    "autoimmune": "Rheumatology",
    "emergency": EMERGENCY_MEDICINE,
    "casualty": EMERGENCY_MEDICINE,
    "er": EMERGENCY_MEDICINE,
    "a and e": EMERGENCY_MEDICINE,
    "internal medicine": GENERAL_MEDICINE,
    "general physician": GENERAL_MEDICINE,
    "general practice": GENERAL_MEDICINE,
    "gp": GENERAL_MEDICINE,
    "family medicine": GENERAL_MEDICINE,
    "physician": GENERAL_MEDICINE,
    "medicine": GENERAL_MEDICINE,
}

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)

_CANONICAL_BY_LOWER = {s.casefold(): s for s in CANONICAL_SPECIALTIES}


def _normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    stripped = _NON_WORD_RE.sub(" ", folded)
    return _WS_RE.sub(" ", stripped).strip()


def canonicalise_specialty(raw: str | None, *, default: str = GENERAL_MEDICINE) -> str:
    """
    Map arbitrary specialty text onto the closed vocabulary.

    Resolution order: exact match, then alias table, then containment in either
    direction (so "Department of Cardiology" and "cardio" both land on
    "Cardiology"). Falls back to `default` rather than passing unknown text
    through, so a hallucinated specialty can never reach the doctor query.
    """
    normalised = _normalise(raw or "")
    if not normalised:
        return default

    exact = _CANONICAL_BY_LOWER.get(normalised)
    if exact:
        return exact

    alias = _ALIASES.get(normalised)
    if alias:
        return alias

    for canonical_lower, canonical in _CANONICAL_BY_LOWER.items():
        if canonical_lower in normalised or normalised in canonical_lower:
            return canonical

    for alias_key, canonical in _ALIASES.items():
        if re.search(rf"\b{re.escape(alias_key)}\b", normalised):
            return canonical

    return default


def is_canonical(specialty: str) -> bool:
    return specialty in CANONICAL_SPECIALTIES
