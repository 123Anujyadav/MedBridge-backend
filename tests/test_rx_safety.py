"""
Prescription safety verification.

The property under test throughout is the one that makes this feature safe to
show a patient: **the verifier never reports an unchecked medicine as safe.**
Every provider here is a stub, because the point is to pin behaviour when the
real providers are slow, partial, or down — which is exactly when a naive
implementation says "no problems found".
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.rxsafety.application import rules
from app.rxsafety.application.verifier import PrescriptionVerifier
from app.rxsafety.domain.entities import DrugConcept, DrugLabel, worst_verdict


# ── stubs ────────────────────────────────────────────────────────────────


class StubNormaliser:
    name = "rxnorm"

    def __init__(self, mapping: dict[str, DrugConcept] | None = None) -> None:
        self._mapping = mapping or {}

    async def normalise(self, drug_name: str) -> DrugConcept:
        return self._mapping.get(drug_name, DrugConcept(original_name=drug_name))

    async def normalise_many(self, drug_names):
        return [await self.normalise(n) for n in drug_names]


class StubLabelSource:
    name = "openfda"

    def __init__(self, labels: dict[str, DrugLabel | None] | None = None) -> None:
        self._labels = labels or {}

    async def fetch_label(self, concept: DrugConcept):
        return self._labels.get(concept.rxcui)


class DeadLabelSource:
    """Every lookup fails, as during an openFDA outage."""

    name = "openfda"

    async def fetch_label(self, concept: DrugConcept):
        return None


class RecordingSession:
    """Captures what would be persisted without touching a database."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj) -> None:
        self.added.append(obj)
        if not getattr(obj, "id", None):
            obj.id = uuid.uuid4()

    async def flush(self) -> None:
        return None


def _medication(name: str, **overrides):
    base = dict(
        name=name,
        generic_name=None,
        dosage="500 mg",
        frequency="twice daily",
        duration="5 days",
        food_instruction="after_food",
        rxcui=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _prescription():
    return SimpleNamespace(id=uuid.uuid4(), patient_id=uuid.uuid4())


def _findings(session: RecordingSession):
    return [o for o in session.added if hasattr(o, "category")]


def _record(session: RecordingSession):
    return next(o for o in session.added if hasattr(o, "verdict"))


# ── the core safety property ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unresolvable_drug_is_never_reported_safe():
    """
    A drug no source can identify must not produce a clean verdict.

    This is the failure that matters most: a patient shown "Safe" for a
    prescription nothing was actually checked against.
    """
    session = RecordingSession()
    verifier = PrescriptionVerifier(StubNormaliser(), StubLabelSource())

    record = await verifier.verify(
        session, _prescription(), [_medication("Nonexistentazole")]
    )

    assert record.verdict != "safe"
    assert record.status == "degraded"
    assert "Nonexistentazole" in record.unchecked_medications
    assert record.checked_medication_count == 0


@pytest.mark.asyncio
async def test_label_source_outage_marks_drug_unchecked_not_safe():
    """A resolved drug whose label could not be fetched is still unchecked."""
    session = RecordingSession()
    verifier = PrescriptionVerifier(
        StubNormaliser(
            {"Metformin": DrugConcept("Metformin", "6809", "metformin", ("metformin",))}
        ),
        DeadLabelSource(),
    )

    record = await verifier.verify(session, _prescription(), [_medication("Metformin")])

    assert record.verdict != "safe"
    assert record.unchecked_medications == ["Metformin"]
    assert record.status == "degraded"


@pytest.mark.asyncio
async def test_clean_label_yields_safe_with_full_confidence():
    """When a drug genuinely is checked and clean, say so."""
    session = RecordingSession()
    concept = DrugConcept("Metformin", "6809", "metformin", ("metformin",))
    verifier = PrescriptionVerifier(
        StubNormaliser({"Metformin": concept}),
        StubLabelSource({"6809": DrugLabel(rxcui="6809")}),
    )

    record = await verifier.verify(session, _prescription(), [_medication("Metformin")])

    assert record.verdict == "safe"
    assert record.status == "completed"
    assert record.unchecked_medications == []
    assert record.confidence == 1.0


# ── deterministic rules ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_duplicate_ingredient_across_brand_and_generic_is_critical():
    """
    Crocin and Paracetamol read as two drugs and are one ingredient.

    Ingredient overlap is the only thing that catches this; name comparison
    never would.
    """
    session = RecordingSession()
    crocin = DrugConcept("Crocin", "161", "Crocin", ("acetaminophen",))
    paracetamol = DrugConcept("Paracetamol", "162", "Paracetamol", ("acetaminophen",))

    verifier = PrescriptionVerifier(
        StubNormaliser({"Crocin": crocin, "Paracetamol": paracetamol}),
        StubLabelSource({"161": DrugLabel(rxcui="161"), "162": DrugLabel(rxcui="162")}),
    )

    record = await verifier.verify(
        session,
        _prescription(),
        [_medication("Crocin"), _medication("Paracetamol")],
    )

    duplicates = [f for f in _findings(session) if f.category == "duplicate_therapy"]
    assert len(duplicates) == 1
    assert duplicates[0].severity == "critical"
    assert record.verdict == "critical"


def test_duplicate_check_ignores_unresolved_drugs():
    """Without ingredients there is nothing to compare; do not guess by name."""
    concepts = [DrugConcept("Amoxicillin"), DrugConcept("Amoxicillin-Clavulanate")]
    assert rules.check_duplicate_therapy(concepts, [c.original_name for c in concepts]) == []


def test_allergy_match_works_through_ingredient_not_just_name():
    concepts = [DrugConcept("Augmentin", "111", "Augmentin", ("amoxicillin",))]
    findings = rules.check_allergies(concepts, ["Amoxicillin"])

    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert findings[0].category == "allergy"


def test_allergy_check_is_silent_when_none_recorded():
    concepts = [DrugConcept("Augmentin", "111", "Augmentin", ("amoxicillin",))]
    assert rules.check_allergies(concepts, []) == []


def test_missing_food_instruction_is_reported():
    findings = rules.check_food_instructions(
        [{"name": "Metformin", "food_instruction": ""}]
    )
    assert len(findings) == 1
    assert findings[0].category == "food_interaction"


def test_present_food_instruction_is_not_reported():
    findings = rules.check_food_instructions(
        [{"name": "Metformin", "food_instruction": "after_food"}]
    )
    assert findings == []


# ── verdict aggregation ──────────────────────────────────────────────────


def test_worst_verdict_picks_the_most_severe():
    assert worst_verdict(["safe", "warning", "critical"]) == "critical"
    assert worst_verdict(["safe", "warning"]) == "warning"
    assert worst_verdict(["safe"]) == "safe"


def test_worst_verdict_of_nothing_is_unknown_not_safe():
    """Nothing checked is not the same as nothing wrong."""
    assert worst_verdict([]) == "unknown"


# ── patient-factor filtering ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pregnancy_warning_does_not_fire_without_the_factor():
    """
    Label warnings are filtered to the patient.

    Showing every pregnancy section to every patient buries the findings that
    do apply, so an unset factor must not raise one.
    """
    session = RecordingSession()
    concept = DrugConcept("Isotretinoin", "999", "Isotretinoin", ("isotretinoin",))
    label = DrugLabel(rxcui="999", pregnancy=["May cause severe fetal harm in pregnancy."])

    verifier = PrescriptionVerifier(
        StubNormaliser({"Isotretinoin": concept}), StubLabelSource({"999": label})
    )

    await verifier.verify(session, _prescription(), [_medication("Isotretinoin")], {})
    assert [f for f in _findings(session) if f.category == "pregnancy"] == []


@pytest.mark.asyncio
async def test_pregnancy_warning_fires_when_the_patient_is_pregnant():
    session = RecordingSession()
    concept = DrugConcept("Isotretinoin", "999", "Isotretinoin", ("isotretinoin",))
    label = DrugLabel(rxcui="999", pregnancy=["May cause severe fetal harm in pregnancy."])

    verifier = PrescriptionVerifier(
        StubNormaliser({"Isotretinoin": concept}), StubLabelSource({"999": label})
    )

    record = await verifier.verify(
        session, _prescription(), [_medication("Isotretinoin")], {"is_pregnant": True}
    )

    pregnancy = [f for f in _findings(session) if f.category == "pregnancy"]
    assert len(pregnancy) == 1
    assert pregnancy[0].severity == "critical"
    assert record.verdict == "critical"


@pytest.mark.asyncio
async def test_elderly_warning_respects_age_threshold():
    session = RecordingSession()
    concept = DrugConcept("Diazepam", "3322", "Diazepam", ("diazepam",))
    label = DrugLabel(
        rxcui="3322", geriatric_use=["In elderly patients, start at the low end."]
    )
    verifier = PrescriptionVerifier(
        StubNormaliser({"Diazepam": concept}), StubLabelSource({"3322": label})
    )

    await verifier.verify(session, _prescription(), [_medication("Diazepam")], {"age": 40})
    assert [f for f in _findings(session) if f.category == "elderly"] == []

    session_old = RecordingSession()
    await verifier.verify(
        session_old, _prescription(), [_medication("Diazepam")], {"age": 72}
    )
    assert [f for f in _findings(session_old) if f.category == "elderly"]


# ── evidence and provenance ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_label_findings_carry_citable_evidence():
    """
    A finding drawn from a label must quote it.

    Evidence is what separates a grounded finding from a model assertion, and
    the UI badges them differently.
    """
    session = RecordingSession()
    concept = DrugConcept("Warfarin", "11289", "Warfarin", ("warfarin",))
    label = DrugLabel(
        rxcui="11289",
        contraindications=["Contraindicated in patients with active bleeding."],
        reference="https://api.fda.gov/drug/label.json?search=id:abc",
    )
    verifier = PrescriptionVerifier(
        StubNormaliser({"Warfarin": concept}), StubLabelSource({"11289": label})
    )

    await verifier.verify(session, _prescription(), [_medication("Warfarin")])

    finding = next(f for f in _findings(session) if f.category == "contraindication")
    assert finding.source == "openfda"
    assert finding.evidence
    assert "active bleeding" in finding.evidence[0]["excerpt"]
    assert finding.evidence[0]["reference"].startswith("https://api.fda.gov")


@pytest.mark.asyncio
async def test_verifier_never_mutates_the_prescription():
    """
    The whole feature is advisory. The only write permitted onto a medication
    row is the rxcui lookup key.
    """
    session = RecordingSession()
    concept = DrugConcept("Warfarin", "11289", "Warfarin", ("warfarin",))
    label = DrugLabel(rxcui="11289", contraindications=["Do not use."])
    medication = _medication("Warfarin")
    before = {
        "name": medication.name,
        "dosage": medication.dosage,
        "frequency": medication.frequency,
        "duration": medication.duration,
    }

    verifier = PrescriptionVerifier(
        StubNormaliser({"Warfarin": concept}), StubLabelSource({"11289": label})
    )
    await verifier.verify(session, _prescription(), [medication])

    assert medication.name == before["name"]
    assert medication.dosage == before["dosage"]
    assert medication.frequency == before["frequency"]
    assert medication.duration == before["duration"]
    # rxcui is the one permitted write, and it is a lookup key.
    assert medication.rxcui == "11289"


# ── degradation and coverage ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_partial_coverage_lowers_confidence():
    """Two of four drugs checked is not a confident review."""
    session = RecordingSession()
    known = {
        "A": DrugConcept("A", "1", "A", ("a",)),
        "B": DrugConcept("B", "2", "B", ("b",)),
    }
    verifier = PrescriptionVerifier(
        StubNormaliser(known),
        StubLabelSource({"1": DrugLabel(rxcui="1"), "2": DrugLabel(rxcui="2")}),
    )

    record = await verifier.verify(
        session,
        _prescription(),
        [_medication("A"), _medication("B"), _medication("C"), _medication("D")],
    )

    assert record.confidence == 0.5
    assert sorted(record.unchecked_medications) == ["C", "D"]


@pytest.mark.asyncio
async def test_empty_prescription_is_safe_and_complete():
    session = RecordingSession()
    verifier = PrescriptionVerifier(StubNormaliser(), StubLabelSource())

    record = await verifier.verify(session, _prescription(), [])

    assert record.verdict == "safe"
    assert record.status == "completed"
    assert record.checked_medication_count == 0


@pytest.mark.asyncio
async def test_fallback_summary_states_coverage_when_explainer_absent():
    """
    With no language model the summary still has to be honest about what was
    not checked, because that text is what the patient reads.
    """
    session = RecordingSession()
    verifier = PrescriptionVerifier(StubNormaliser(), StubLabelSource(), explainer=None)

    record = await verifier.verify(session, _prescription(), [_medication("Unknownium")])

    assert "Unknownium" in record.summary
    assert "manual review" in record.summary.lower()
    assert "does not change your prescription" in record.summary.lower()
