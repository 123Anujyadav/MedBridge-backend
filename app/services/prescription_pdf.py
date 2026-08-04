"""
Printable prescription PDFs.

Separate from `report_generator` on purpose. A prescription and a medical report
are different documents with different content, different legal weight and
different layouts; folding them into one generator would mean a growing pile of
`if is_prescription` branches through a module that already renders reports
correctly. The two share ReportLab and the house palette, nothing else.

The rendered document deliberately reproduces the *snapshot* fields on the
prescription rather than reading the live doctor profile, so reprinting an old
prescription years later produces the same page it produced on the day it was
signed.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Iterable, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.core.upload import PRESCRIPTIONS_DIR

logger = logging.getLogger(__name__)

BRAND = colors.HexColor("#00685F")
INK = colors.HexColor("#191C1C")
MUTED = colors.HexColor("#6D7A77")
RULE = colors.HexColor("#C0D0CE")
TINT = colors.HexColor("#F4F9F8")
ALERT = colors.HexColor("#B3261E")

FOOD_LABELS = {
    "before_food": "Before food",
    "after_food": "After food",
    "with_food": "With food",
    "empty_stomach": "Empty stomach",
    "anytime": "Any time",
}


def _text(value: Any, fallback: str = "—") -> str:
    """Render a value for the page, never the string 'None'."""
    if value is None:
        return fallback
    rendered = str(value).strip()
    return rendered or fallback


class PrescriptionPDFGenerator:
    """Renders a prescription to a PDF on disk and returns its location."""

    def __init__(self, output_dir: str = PRESCRIPTIONS_DIR) -> None:
        self._output_dir = output_dir

    # ── styles ───────────────────────────────────────────────────────────

    def _styles(self) -> dict[str, ParagraphStyle]:
        base = getSampleStyleSheet()
        return {
            "brand": ParagraphStyle(
                "Brand", parent=base["Heading1"], fontSize=20, leading=24,
                textColor=BRAND, spaceAfter=0,
            ),
            "brandSub": ParagraphStyle(
                "BrandSub", parent=base["Normal"], fontSize=7.5, leading=10,
                textColor=MUTED, spaceAfter=0,
            ),
            "docType": ParagraphStyle(
                "DocType", parent=base["Heading2"], fontSize=13, leading=16,
                textColor=BRAND, alignment=2, spaceAfter=0,
            ),
            "docMeta": ParagraphStyle(
                "DocMeta", parent=base["Normal"], fontSize=8, leading=11,
                textColor=MUTED, alignment=2,
            ),
            "section": ParagraphStyle(
                "Section", parent=base["Heading3"], fontSize=10, leading=13,
                textColor=BRAND, spaceBefore=10, spaceAfter=5,
            ),
            "body": ParagraphStyle(
                "Body", parent=base["Normal"], fontSize=9, leading=12.5, textColor=INK,
            ),
            "small": ParagraphStyle(
                "Small", parent=base["Normal"], fontSize=7.5, leading=10, textColor=MUTED,
            ),
            "cell": ParagraphStyle(
                "Cell", parent=base["Normal"], fontSize=8, leading=11, textColor=INK,
            ),
            "cellHead": ParagraphStyle(
                "CellHead", parent=base["Normal"], fontSize=8, leading=11,
                textColor=colors.white,
            ),
            "advisory": ParagraphStyle(
                "Advisory", parent=base["Normal"], fontSize=7.5, leading=10,
                textColor=ALERT,
            ),
        }

    # ── building blocks ──────────────────────────────────────────────────

    def _header(self, rx: Any, s: dict) -> Iterable:
        issued = rx.consultation_date or rx.created_at
        issued_text = (
            issued.strftime("%d %b %Y, %H:%M UTC") if isinstance(issued, datetime) else "—"
        )
        left = [
            Paragraph("MedBridge", s["brand"]),
            Paragraph("AI HEALTHCARE PLATFORM", s["brandSub"]),
        ]
        right = [
            Paragraph("<b>PRESCRIPTION</b>", s["docType"]),
            Paragraph(f"Rx ID: {str(rx.id)[:8].upper()}", s["docMeta"]),
            Paragraph(f"Issued: {issued_text}", s["docMeta"]),
        ]
        table = Table([[left, right]], colWidths=[95 * mm, 75 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return [table, Spacer(1, 6), HRFlowable(width="100%", thickness=1.5, color=BRAND), Spacer(1, 8)]

    def _parties(self, rx: Any, s: dict) -> Iterable:
        """Prescriber and patient, side by side — the hospital-slip layout."""
        prescriber = [
            Paragraph("<b>PRESCRIBER</b>", s["small"]),
            Paragraph(f"<b>{_text(rx.doctor_name)}</b>", s["body"]),
            Paragraph(_text(rx.doctor_qualification, ""), s["small"]),
            Paragraph(_text(rx.doctor_specialty, ""), s["small"]),
            Paragraph(_text(rx.doctor_hospital, ""), s["small"]),
            Paragraph(f"Reg. No: {_text(rx.doctor_registration_number)}", s["small"]),
        ]
        if rx.doctor_experience_years:
            prescriber.append(
                Paragraph(f"{rx.doctor_experience_years} years' experience", s["small"])
            )

        patient = [
            Paragraph("<b>PATIENT</b>", s["small"]),
            Paragraph(f"<b>{_text(rx.patient_name)}</b>", s["body"]),
            Paragraph(f"Patient ID: {str(rx.patient_id)[:8].upper()}", s["small"]),
            Paragraph(f"Diagnosis: {_text(rx.diagnosis)}", s["small"]),
        ]

        table = Table([[prescriber, patient]], colWidths=[85 * mm, 85 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (-1, -1), TINT),
            ("BOX", (0, 0), (-1, -1), 0.5, RULE),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, RULE),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        return [table, Spacer(1, 4)]

    def _medications(self, medications: Sequence[Any], s: dict) -> Iterable:
        story: list = [Paragraph("℞  MEDICATIONS", s["section"])]

        if not medications:
            story.append(Paragraph("No medicines were prescribed.", s["body"]))
            return story

        header = [
            Paragraph("<b>#</b>", s["cellHead"]),
            Paragraph("<b>Medicine</b>", s["cellHead"]),
            Paragraph("<b>Strength</b>", s["cellHead"]),
            Paragraph("<b>Dosage</b>", s["cellHead"]),
            Paragraph("<b>Frequency</b>", s["cellHead"]),
            Paragraph("<b>Duration</b>", s["cellHead"]),
            Paragraph("<b>Food</b>", s["cellHead"]),
            Paragraph("<b>Qty</b>", s["cellHead"]),
        ]
        rows = [header]

        for index, med in enumerate(medications, start=1):
            name = f"<b>{_text(med.name)}</b>"
            # The generic is printed under the brand because that is what a
            # pharmacist substitutes against, and what makes a cheaper
            # equivalent findable.
            if getattr(med, "generic_name", None):
                name += f"<br/><font size=7 color='#6D7A77'>{med.generic_name}</font>"
            if getattr(med, "brand_name", None):
                name += f"<br/><font size=7 color='#6D7A77'>Brand: {med.brand_name}</font>"

            rows.append([
                Paragraph(str(index), s["cell"]),
                Paragraph(name, s["cell"]),
                Paragraph(_text(getattr(med, "strength", None)), s["cell"]),
                Paragraph(_text(med.dosage), s["cell"]),
                Paragraph(_text(med.frequency), s["cell"]),
                Paragraph(_text(med.duration), s["cell"]),
                Paragraph(
                    FOOD_LABELS.get(getattr(med, "food_instruction", None) or "", "—"),
                    s["cell"],
                ),
                Paragraph(_text(getattr(med, "quantity", None)), s["cell"]),
            ])

        table = Table(
            rows,
            colWidths=[8 * mm, 44 * mm, 20 * mm, 22 * mm, 26 * mm, 20 * mm, 20 * mm, 10 * mm],
            repeatRows=1,
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, RULE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

        instructions = [
            (m.name, m.special_instructions)
            for m in medications
            if (getattr(m, "special_instructions", "") or "").strip()
        ]
        if instructions:
            story.append(Spacer(1, 7))
            story.append(Paragraph("INSTRUCTIONS", s["section"]))
            for name, note in instructions:
                story.append(Paragraph(f"<b>{_text(name)}:</b> {note}", s["body"]))

        return story

    def _verification(self, verification: Any | None, s: dict) -> Iterable:
        """
        The AI safety review, printed as an advisory block.

        Included because a patient reading a paper prescription should see the
        same warnings the app showed them. Worded to make it unmistakable that
        the review annotates the prescription and does not alter it.
        """
        if verification is None:
            return []

        verdict = (verification.verdict or "unknown").upper()
        story = [
            Paragraph("AI SAFETY REVIEW (ADVISORY)", s["section"]),
            Paragraph(
                f"<b>Result: {verdict}</b> &nbsp;·&nbsp; "
                f"Confidence {verification.confidence:.0%} &nbsp;·&nbsp; "
                f"{verification.checked_medication_count} medicine(s) checked",
                s["body"],
            ),
        ]
        if verification.summary:
            story.append(Paragraph(verification.summary, s["body"]))

        if verification.unchecked_medications:
            story.append(
                Paragraph(
                    "Not checked: "
                    + ", ".join(verification.unchecked_medications)
                    + ". These still require manual review.",
                    s["advisory"],
                )
            )

        story.append(
            Paragraph(
                "This review is generated from published drug labels and is guidance "
                "only. It does not modify the prescription above, which stands exactly "
                "as issued by the prescriber.",
                s["small"],
            )
        )
        return [KeepTogether(story), Spacer(1, 4)]

    def _signature(self, rx: Any, s: dict) -> Iterable:
        signed = rx.signed_at.strftime("%d %b %Y, %H:%M UTC") if rx.signed_at else None
        block = [
            Paragraph("_" * 34, s["body"]),
            Paragraph(f"<b>{_text(rx.doctor_name)}</b>", s["small"]),
            Paragraph(_text(rx.doctor_specialty, ""), s["small"]),
            Paragraph(f"Reg. No: {_text(rx.doctor_registration_number)}", s["small"]),
            Paragraph(
                f"Digitally signed {signed}" if signed else "Not digitally signed",
                s["small"],
            ),
        ]
        table = Table([["", block]], colWidths=[100 * mm, 70 * mm])
        table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        return [Spacer(1, 14), table]

    # ── entry point ──────────────────────────────────────────────────────

    def generate(
        self,
        rx: Any,
        medications: Sequence[Any],
        verification: Any | None = None,
    ) -> dict:
        """
        Render and persist. Returns filename, absolute path and download URL.

        Raises on failure rather than returning a broken path: a caller that
        thinks it has a PDF and does not would surface a dead download link.
        """
        os.makedirs(self._output_dir, exist_ok=True)

        filename = f"prescription_{str(rx.id)[:8]}.pdf"
        dest_path = os.path.join(self._output_dir, filename)
        s = self._styles()

        document = SimpleDocTemplate(
            dest_path,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=16 * mm,
            bottomMargin=20 * mm,
            title=f"Prescription {str(rx.id)[:8].upper()}",
            author="MedBridge",
        )

        story: list = []
        story += self._header(rx, s)
        story += self._parties(rx, s)
        story += self._medications(medications, s)
        story += self._verification(verification, s)

        if (rx.notes or "").strip():
            story.append(Paragraph("CLINICAL NOTES", s["section"]))
            story.append(Paragraph(rx.notes, s["body"]))

        if rx.follow_up_date:
            story.append(Spacer(1, 6))
            story.append(
                Paragraph(f"<b>Follow-up:</b> {_text(rx.follow_up_date)}", s["body"])
            )

        story += self._signature(rx, s)

        def _footer(canvas, doc) -> None:
            canvas.saveState()
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(MUTED)
            canvas.drawString(
                20 * mm, 12 * mm,
                "MedBridge · Computer-generated prescription · Not valid without the prescriber's signature",
            )
            canvas.drawRightString(190 * mm, 12 * mm, f"Page {doc.page}")
            canvas.restoreState()

        document.build(story, onFirstPage=_footer, onLaterPages=_footer)

        logger.info("[RX_PDF_GENERATED] rx=%s file=%s", rx.id, filename)
        return {
            "filename": filename,
            "file_path": dest_path,
            # Must match the route in api/v1/endpoints/prescriptions.py. The PDF
            # is served through an authenticated endpoint rather than a static
            # mount, because a prescription is a clinical document and a
            # guessable /uploads path would be readable by anyone.
            "file_url": f"/api/v1/prescriptions/{rx.id}/pdf",
        }


prescription_pdf_generator = PrescriptionPDFGenerator()
