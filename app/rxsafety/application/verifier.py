"""
The prescription safety verifier.

Composes the normaliser, the label source, the deterministic rules and the
explainer into one review, then persists it.

Two invariants hold throughout:

1. **Nothing here writes to `prescriptions` or `medications`.** The review is
   stored beside the prescription, never into it. `rxcui` on a medication row is
   the single exception and it is a normalisation cache, not clinical content.

2. **Absence of evidence is never reported as safety.** Every drug that could
   not be resolved or whose label could not be fetched is named in
   `unchecked_medications`, and the presence of any such drug caps the run at
   `degraded` rather than `completed`.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prescription import Medication, Prescription
from app.models.rx_verification import (
    PrescriptionVerification,
    VerificationFinding,
    VERDICT_SAFE,
    VERDICT_UNKNOWN,
    VERDICT_WARNING,
)
from app.rxsafety.application import rules
from app.rxsafety.domain.entities import (
    DrugConcept,
    DrugLabel,
    Evidence,
    Finding,
    worst_verdict,
)
from app.rxsafety.domain.ports import (
    DrugLabelSource,
    DrugNormaliser,
    SafetyExplainer,
)

logger = logging.getLogger(__name__)

ENGINE_VERSION = "rxsafety-1.0"

# Label sections mapped onto the finding categories the product surfaces.
# Each entry: (category, label attribute, matching terms, severity, title)
LABEL_CHECKS = (
    ("contraindication", "contraindications", None, "critical", "Contraindication on the label"),
    ("drug_interaction", "drug_interactions", None, "warning", "Interaction noted on the label"),
    ("max_dosage", "dosage_and_administration", None, "warning", "Dosage guidance on the label"),
)

# Patient-factor checks: only raised when the patient actually has the factor.
FACTOR_CHECKS = (
    ("pregnancy", "pregnancy", rules.PREGNANCY_TERMS, "critical", "Pregnancy warning"),
    ("renal", "renal_notes", rules.RENAL_TERMS, "warning", "Kidney precaution"),
    ("hepatic", "hepatic_notes", rules.HEPATIC_TERMS, "warning", "Liver precaution"),
    ("elderly", "geriatric_use", rules.ELDERLY_TERMS, "warning", "Guidance for older adults"),
)

ELDERLY_AGE = 65


class PrescriptionVerifier:
    """Runs one safety review and stores the result."""

    def __init__(
        self,
        normaliser: DrugNormaliser,
        label_source: DrugLabelSource,
        explainer: SafetyExplainer | None = None,
    ) -> None:
        self._normaliser = normaliser
        self._labels = label_source
        self._explainer = explainer

    # ── public API ───────────────────────────────────────────────────────

    async def verify(
        self,
        db: AsyncSession,
        prescription: Prescription,
        medications: Sequence[Medication],
        patient_context: dict | None = None,
    ) -> PrescriptionVerification:
        started = time.monotonic()
        context = patient_context or {}
        names = [m.name for m in medications if (m.name or "").strip()]

        record = PrescriptionVerification(
            prescription_id=prescription.id,
            status="pending",
            engine_version=ENGINE_VERSION,
        )

        if not names:
            record.status = "completed"
            record.verdict = VERDICT_SAFE
            record.confidence = 1.0
            record.summary = "This prescription contains no medicines to check."
            # Set explicitly rather than relying on the column defaults: those
            # only materialise on INSERT, so a caller reading this record before
            # it is flushed would otherwise see None where it expects a count.
            record.checked_medication_count = 0
            record.unchecked_medications = []
            record.sources_used = []
            record.completed_at = datetime.now(timezone.utc)
            record.duration_ms = int((time.monotonic() - started) * 1000)
            db.add(record)
            return record

        sources: list[str] = []

        # ── 1. normalise ─────────────────────────────────────────────────
        concepts = await self._normaliser.normalise_many(names)
        if any(c.resolved for c in concepts):
            sources.append(self._normaliser.name)

        # Cache the RxCUI back onto the medication rows. This is the only write
        # this context makes outside its own tables, and it is a lookup key
        # rather than clinical content — it changes no dose, drug or instruction.
        by_name = {c.original_name: c for c in concepts}
        for medication in medications:
            concept = by_name.get(medication.name)
            if concept and concept.resolved and not medication.rxcui:
                medication.rxcui = concept.rxcui

        # ── 2. fetch labels ──────────────────────────────────────────────
        labels: dict[str, DrugLabel | None] = {}
        fetch_labels = getattr(self._labels, "fetch_labels", None)
        if callable(fetch_labels):
            labels = await fetch_labels(concepts)
        else:
            for concept in concepts:
                if concept.resolved:
                    labels[concept.rxcui] = await self._labels.fetch_label(concept)

        if any(label is not None for label in labels.values()):
            sources.append(self._labels.name)

        # ── 3. what could not be checked ─────────────────────────────────
        unchecked: list[str] = [c.original_name for c in concepts if not c.resolved]
        for concept in concepts:
            if concept.resolved and labels.get(concept.rxcui) is None:
                unchecked.append(concept.original_name)

        # ── 4. findings ──────────────────────────────────────────────────
        findings: list[Finding] = []
        findings += rules.check_duplicate_therapy(concepts, names)
        findings += rules.check_allergies(concepts, context.get("allergies") or [])
        findings += rules.check_unresolved(concepts)
        findings += rules.check_food_instructions(
            [
                {"name": m.name, "food_instruction": m.food_instruction}
                for m in medications
            ]
        )
        findings += self._label_findings(concepts, labels, context)

        # ── 5. verdict ───────────────────────────────────────────────────
        checked_count = sum(
            1 for c in concepts if c.resolved and labels.get(c.rxcui) is not None
        )

        if findings:
            verdict = worst_verdict([f.severity for f in findings])
        elif checked_count:
            # Genuinely checked and nothing found.
            verdict = VERDICT_SAFE
        else:
            verdict = VERDICT_UNKNOWN

        record.verdict = verdict
        record.confidence = self._confidence(concepts, labels, findings)
        record.checked_medication_count = checked_count
        record.unchecked_medications = sorted(set(unchecked))
        record.status = "degraded" if unchecked else "completed"

        # ── 6. summary ───────────────────────────────────────────────────
        if self._explainer:
            summary, model = await self._explainer.summarise(
                medications=names,
                findings_payload=[
                    {
                        "category": f.category,
                        "severity": f.severity,
                        "title": f.title,
                        "detail": f.detail,
                        "medications_involved": f.medications_involved,
                    }
                    for f in findings
                ],
                patient_context={**context, "unchecked": record.unchecked_medications},
            )
            if summary:
                record.summary = summary
                record.model_used = model
                sources.append(self._explainer.name)

        if not record.summary:
            record.summary = self._fallback_summary(record, findings)

        record.sources_used = sources
        record.completed_at = datetime.now(timezone.utc)
        record.duration_ms = int((time.monotonic() - started) * 1000)

        db.add(record)
        await db.flush()

        for finding in findings:
            db.add(
                VerificationFinding(
                    verification_id=record.id,
                    category=finding.category,
                    severity=finding.severity,
                    confidence=finding.confidence,
                    title=finding.title[:255],
                    detail=finding.detail,
                    recommendation=finding.recommendation,
                    medications_involved=finding.medications_involved,
                    source=finding.source,
                    evidence=[e.as_dict() for e in finding.evidence],
                )
            )

        logger.info(
            "[RXSAFETY_VERIFIED] rx=%s verdict=%s findings=%d checked=%d/%d "
            "unchecked=%s status=%s %dms",
            prescription.id,
            record.verdict,
            len(findings),
            checked_count,
            len(names),
            record.unchecked_medications,
            record.status,
            record.duration_ms,
        )
        return record

    # ── internals ────────────────────────────────────────────────────────

    def _label_findings(
        self,
        concepts: Sequence[DrugConcept],
        labels: dict[str, DrugLabel | None],
        context: dict,
    ) -> list[Finding]:
        findings: list[Finding] = []
        age = context.get("age")
        is_pregnant = bool(context.get("is_pregnant"))
        conditions = " ".join(str(c).lower() for c in (context.get("conditions") or []))

        for concept in concepts:
            label = labels.get(concept.rxcui) if concept.resolved else None
            if label is None or label.is_empty:
                continue

            for category, attribute, _terms, severity, title in LABEL_CHECKS:
                sections = getattr(label, attribute, [])
                if not sections:
                    continue
                findings.append(
                    self._from_label(
                        concept, label, category, attribute, sections[0], severity, title
                    )
                )

            for category, attribute, terms, severity, title in FACTOR_CHECKS:
                # Only surface a patient-factor warning when the patient has the
                # factor. Showing every pregnancy section to every patient turns
                # the panel into noise and buries the findings that apply.
                applies = (
                    (category == "pregnancy" and is_pregnant)
                    or (category == "elderly" and isinstance(age, int) and age >= ELDERLY_AGE)
                    or (category == "renal" and ("kidney" in conditions or "renal" in conditions))
                    or (category == "hepatic" and ("liver" in conditions or "hepat" in conditions))
                )
                if not applies:
                    continue

                sections = getattr(label, attribute, [])
                excerpt = rules.label_mentions(sections, terms or ())
                if not excerpt:
                    continue
                findings.append(
                    self._from_label(
                        concept, label, category, attribute, excerpt, severity, title
                    )
                )

        return findings

    @staticmethod
    def _from_label(
        concept: DrugConcept,
        label: DrugLabel,
        category: str,
        section: str,
        excerpt: str,
        severity: str,
        title: str,
    ) -> Finding:
        return Finding(
            category=category,
            severity=severity,
            title=f"{title}: {concept.display_name}",
            detail=excerpt[:800],
            recommendation="Discuss with the prescribing doctor before making any change.",
            # Label text is authoritative about what the label says, which is
            # not the same as being tailored to this patient. High but not
            # absolute.
            confidence=0.8,
            medications_involved=[concept.display_name],
            source="openfda",
            evidence=[
                Evidence(
                    source="openfda",
                    section=section,
                    excerpt=excerpt[:1200],
                    reference=label.reference,
                )
            ],
        )

    @staticmethod
    def _confidence(
        concepts: Sequence[DrugConcept],
        labels: dict[str, DrugLabel | None],
        findings: Sequence[Finding],
    ) -> float:
        """
        Confidence in the *review*, not in any one finding.

        Driven by coverage: a review that could only check two of five drugs is
        not a confident review however clean those two came back.
        """
        total = len(concepts) or 1
        checked = sum(
            1 for c in concepts if c.resolved and labels.get(c.rxcui) is not None
        )
        coverage = checked / total

        grounded = [f for f in findings if f.is_grounded]
        if findings and not grounded:
            # Only ungrounded findings — cap low.
            return round(min(coverage, 0.5), 2)

        return round(coverage, 2)

    @staticmethod
    def _fallback_summary(
        record: PrescriptionVerification, findings: Sequence[Finding]
    ) -> str:
        """
        Used when the explainer is unavailable.

        Deliberately blunt about coverage — this is the text a patient reads
        when the language model is down, and it must not read as reassurance
        the data does not support.
        """
        parts: list[str] = []
        if findings:
            critical = sum(1 for f in findings if f.severity == "critical")
            warning = sum(1 for f in findings if f.severity == VERDICT_WARNING)
            parts.append(
                f"{len(findings)} point(s) were found for review "
                f"({critical} critical, {warning} warning)."
            )
        elif record.checked_medication_count:
            parts.append(
                "No known issues were found in the drug labels that were checked."
            )
        else:
            parts.append("No medicines could be checked against a drug label.")

        if record.unchecked_medications:
            parts.append(
                f"{', '.join(record.unchecked_medications)} could not be checked and "
                "still need a manual review."
            )

        parts.append("This is guidance only and does not change your prescription.")
        return " ".join(parts)
