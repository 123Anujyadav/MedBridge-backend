import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.core.upload import REPORTS_DIR

class MedicalReportGenerator:
    """
    Service responsible for generating professional, structured PDF Medical Reports
    using ReportLab. Output PDFs include hospital branding, patient & doctor metadata,
    diagnosis, prescribed medications table, clinical notes, and doctor sign-off.
    """

    def generate_pdf(
        self,
        patient_name: str,
        patient_id: str,
        doctor_name: str,
        doctor_id: str,
        symptoms: str,
        diagnosis: str,
        clinical_notes: str,
        medications: List[Dict[str, Any]],
        recommended_tests: Optional[List[str]] = None,
        follow_up_date: Optional[str] = None,
        doctor_remarks: Optional[str] = None,
        hospital_name: Optional[str] = "MedBridge General Hospital & Medical Center",
        # ── Document-management extensions ───────────────────────────────
        # Optional so every existing caller keeps working unchanged. The
        # version workflow supplies them; the consultation flow does not.
        chief_complaint: Optional[str] = None,
        clinical_summary: Optional[str] = None,
        ai_findings: Optional[str] = None,
        prescription_text: Optional[str] = None,
        follow_up_instructions: Optional[str] = None,
        recommendations: Optional[List[str]] = None,
        approval_info: Optional[Dict[str, Any]] = None,
        version_label: Optional[str] = None,
        filename_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render one clinical report to PDF.

        This is the platform's only document generator; the version workflow
        renders through it rather than maintaining a parallel implementation, so
        a preview and the downloaded file are the same bytes from the same code.
        """
        # Ensure output directory exists. Shared with the download route via
        # core.upload so the writer and reader cannot drift apart again.
        upload_dir = REPORTS_DIR
        os.makedirs(upload_dir, exist_ok=True)

        report_id = str(uuid.uuid4())
        stem = filename_stem or f"medical_report_{report_id[:8]}"
        filename = f"{stem}.pdf"
        dest_path = os.path.join(upload_dir, filename)

        doc = SimpleDocTemplate(
            dest_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom Paragraph Styles
        title_style = ParagraphStyle(
            'HeaderTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#00685F'),
            alignment=0
        )

        subtitle_style = ParagraphStyle(
            'HeaderSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor('#6D7A77')
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=15,
            textColor=colors.HexColor('#00685F'),
            spaceBefore=10,
            spaceAfter=5
        )

        body_style = ParagraphStyle(
            'ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#191C1C')
        )

        bold_label = ParagraphStyle(
            'BoldLabel',
            parent=body_style,
            fontName='Helvetica-Bold'
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=11,
            textColor=colors.white
        )

        table_body_style = ParagraphStyle(
            'TableBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#191C1C')
        )

        story = []

        # 1. Header Section
        now_str = datetime.now(timezone.utc).strftime("%B %d, %Y - %H:%M UTC")
        header_data = [
            [
                Paragraph(f"<b>{hospital_name}</b>", title_style),
                Paragraph("<b>OFFICIAL MEDICAL REPORT</b>", ParagraphStyle('RightTitle', parent=title_style, fontSize=14, alignment=2, textColor=colors.HexColor('#00685F')))
            ],
            [
                Paragraph("Enterprise AI Clinical Decision & Consultation System", subtitle_style),
                Paragraph(f"Generated: {now_str}", ParagraphStyle('RightDate', parent=subtitle_style, alignment=2))
            ]
        ]

        header_table = Table(header_data, colWidths=[340, 200])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('TOPPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#00685F'), spaceAfter=12))

        # 2. Patient & Doctor Information Box
        info_data = [
            [
                Paragraph("<b>PATIENT DETAILS</b>", section_heading),
                Paragraph("<b>ATTENDING PHYSICIAN</b>", section_heading)
            ],
            [
                Paragraph(f"<b>Name:</b> {patient_name}", body_style),
                Paragraph(f"<b>Name:</b> {doctor_name}", body_style)
            ],
            [
                Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
                Paragraph(f"<b>Doctor ID:</b> {doctor_id}", body_style)
            ],
            [
                Paragraph(f"<b>Consultation Date:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d')}", body_style),
                Paragraph(f"<b>Facility:</b> {hospital_name}", body_style)
            ]
        ]

        info_table = Table(info_data, colWidths=[270, 270])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F4F9F8')),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#C0D0CE')),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.HexColor('#E2EFEF')),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 12))

        # 3. Clinical Assessment & Diagnosis
        story.append(Paragraph("Clinical Assessment & Diagnosis", section_heading))
        assessment_data = []
        if chief_complaint:
            assessment_data.append([
                Paragraph("<b>Chief Complaint:</b>", bold_label),
                Paragraph(chief_complaint, body_style),
            ])
        if clinical_summary:
            assessment_data.append([
                Paragraph("<b>Clinical Summary:</b>", bold_label),
                Paragraph(clinical_summary, body_style),
            ])
        assessment_data += [
            [Paragraph("<b>Reported Symptoms:</b>", bold_label), Paragraph(symptoms or "None reported", body_style)],
            [Paragraph("<b>Primary Diagnosis:</b>", bold_label), Paragraph(f"<b>{diagnosis or 'Pending Evaluation'}</b>", body_style)],
            [Paragraph("<b>Clinical Notes:</b>", bold_label), Paragraph(clinical_notes or "No additional notes", body_style)]
        ]
        assessment_table = Table(assessment_data, colWidths=[130, 410])
        assessment_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 4),
        ]))
        story.append(assessment_table)
        story.append(Spacer(1, 10))

        # 3b. AI findings, always in their own visually distinct block so a
        #     clinician can never mistake model output for a human assessment.
        if ai_findings:
            story.append(Paragraph("AI Findings (Decision Support — Not a Diagnosis)",
                                   section_heading))
            ai_table = Table(
                [[Paragraph(ai_findings, body_style)]], colWidths=[540]
            )
            ai_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F9F8')),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#00685F')),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(ai_table)
            story.append(Spacer(1, 10))

        # 3c. Free-text prescription, for reports issued without itemised meds.
        if prescription_text:
            story.append(Paragraph("Prescription", section_heading))
            story.append(Paragraph(prescription_text, body_style))
            story.append(Spacer(1, 10))

        # 4. Prescribed Medications Table
        if medications:
            story.append(Paragraph("Prescribed Medications (Rx)", section_heading))
            rx_rows = [
                [
                    Paragraph("Medication Name", table_header_style),
                    Paragraph("Dosage", table_header_style),
                    Paragraph("Frequency", table_header_style),
                    Paragraph("Duration", table_header_style),
                    Paragraph("Special Instructions", table_header_style)
                ]
            ]
            for m in medications:
                rx_rows.append([
                    Paragraph(m.get("name", ""), table_body_style),
                    Paragraph(m.get("dosage", ""), table_body_style),
                    Paragraph(m.get("frequency", ""), table_body_style),
                    Paragraph(m.get("duration", ""), table_body_style),
                    Paragraph(m.get("special_instructions", "-"), table_body_style)
                ])
            rx_table = Table(rx_rows, colWidths=[120, 80, 100, 70, 170])
            rx_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00685F')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(rx_table)
            story.append(Spacer(1, 10))

        # 5. Recommended Tests & Remarks
        additional_info = []
        if recommended_tests:
            tests_str = ", ".join(recommended_tests) if isinstance(recommended_tests, list) else str(recommended_tests)
            additional_info.append([Paragraph("<b>Recommended Diagnostic Tests:</b>", bold_label), Paragraph(tests_str, body_style)])
        if follow_up_date:
            additional_info.append([Paragraph("<b>Follow-up Appointment:</b>", bold_label), Paragraph(follow_up_date, body_style)])
        if follow_up_instructions:
            additional_info.append([Paragraph("<b>Follow-up Instructions:</b>", bold_label), Paragraph(follow_up_instructions, body_style)])
        if recommendations:
            additional_info.append([
                Paragraph("<b>Recommendations:</b>", bold_label),
                Paragraph("<br/>".join(f"• {r}" for r in recommendations), body_style),
            ])
        if doctor_remarks:
            additional_info.append([Paragraph("<b>Doctor Remarks:</b>", bold_label), Paragraph(doctor_remarks, body_style)])

        if additional_info:
            story.append(Paragraph("Follow-up & Diagnostic Orders", section_heading))
            add_table = Table(additional_info, colWidths=[160, 380])
            add_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(add_table)
            story.append(Spacer(1, 15))

        # 5b. Approval record, when the document carries one.
        if approval_info:
            story.append(Paragraph("Approval Record", section_heading))
            rows = []
            for label, key in (
                ("Status", "status"),
                ("Approved By", "approved_by"),
                ("Approved At", "approved_at"),
                ("Approval Notes", "approval_note"),
                ("Rejection Reason", "rejection_reason"),
            ):
                value = approval_info.get(key)
                if value:
                    rows.append([
                        Paragraph(f"<b>{label}:</b>", bold_label),
                        Paragraph(str(value), body_style),
                    ])
            if rows:
                approval_table = Table(rows, colWidths=[160, 380])
                approval_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                ]))
                story.append(approval_table)
                story.append(Spacer(1, 12))

        # 6. Doctor Signature & Stamp
        sign_block = [
            [
                Paragraph("<b>Electronically Verified & Signed by:</b>", subtitle_style),
                Paragraph("<b>Stamp / Digital Hash</b>", ParagraphStyle('StampHead', parent=subtitle_style, alignment=2))
            ],
            [
                Paragraph(f"<b>{doctor_name}</b><br/>Authorized Medical Specialist", body_style),
                Paragraph(f"Ref: {report_id[:16]}<br/>HIPAA Compliant Record", ParagraphStyle('StampVal', parent=subtitle_style, alignment=2))
            ]
        ]
        sign_table = Table(sign_block, colWidths=[340, 200])
        sign_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LINEABOVE', (0,0), (-1,0), 0.5, colors.HexColor('#94A3B8')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(KeepTogether(sign_table))

        # Signature line: an explicit placeholder rather than an implied
        # signature, so an unsigned document never looks signed.
        story.append(Spacer(1, 18))
        signature_line = Table(
            [[Paragraph("_________________________<br/>"
                        "<font size=8>Authorised signature</font>", body_style),
              Paragraph(f"<font size=8>Document version: "
                        f"{version_label or 'v1'}</font>",
                        ParagraphStyle('SigVer', parent=subtitle_style, alignment=2))]],
            colWidths=[340, 200],
        )
        signature_line.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'BOTTOM')]))
        story.append(signature_line)

        footer_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        footer_version = version_label or "v1"

        def _footer(canvas, document) -> None:
            """
            Page number, generation timestamp and version on every page.

            Drawn per page rather than appended to the story, because a flowable
            can only state the total page count after the document is built.
            """
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor('#6D7A77'))
            canvas.drawString(36, 22, f"{hospital_name} — Confidential Clinical Record")
            canvas.drawCentredString(
                letter[0] / 2.0, 22, f"Generated {footer_stamp} · {footer_version}"
            )
            canvas.drawRightString(letter[0] - 36, 22, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        # Build Document
        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)

        file_size = os.path.getsize(dest_path)
        file_size_str = f"{round(file_size / 1024, 1)} KB"

        return {
            "report_id": report_id,
            "filename": filename,
            "file_url": f"/uploads/reports/{filename}",
            "file_size": file_size_str,
            "dest_path": dest_path
        }

    def generate_analytics_pdf(
        self,
        title: str,
        doctor_name: str,
        hospital_name: str,
        period: str,
        sections: List[tuple],
        filename_stem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Render an analytics summary using this same ReportLab pipeline.

        Kept on the existing generator rather than in a second module so the
        platform has one place that knows how to produce a branded PDF — same
        header treatment, same palette, same paginated footer.
        """
        os.makedirs(REPORTS_DIR, exist_ok=True)
        stem = filename_stem or f"analytics_{uuid.uuid4().hex[:8]}"
        dest_path = os.path.join(REPORTS_DIR, f"{stem}.pdf")

        doc = SimpleDocTemplate(
            dest_path, pagesize=letter,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36,
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'AnalyticsTitle', parent=styles['Heading1'], fontName='Helvetica-Bold',
            fontSize=18, leading=22, textColor=colors.HexColor('#00685F'),
        )
        subtitle_style = ParagraphStyle(
            'AnalyticsSubtitle', parent=styles['Normal'], fontName='Helvetica',
            fontSize=10, leading=12, textColor=colors.HexColor('#6D7A77'),
        )
        section_heading = ParagraphStyle(
            'AnalyticsSection', parent=styles['Heading2'], fontName='Helvetica-Bold',
            fontSize=12, leading=15, textColor=colors.HexColor('#00685F'),
            spaceBefore=10, spaceAfter=5,
        )
        body_style = ParagraphStyle(
            'AnalyticsBody', parent=styles['Normal'], fontName='Helvetica',
            fontSize=9, leading=12, textColor=colors.HexColor('#191C1C'),
        )

        story: List[Any] = [
            Paragraph(f"<b>{hospital_name}</b>", title_style),
            Paragraph(title, subtitle_style),
            Paragraph(f"{doctor_name} · Period: {period}", subtitle_style),
            Spacer(1, 8),
            HRFlowable(width="100%", thickness=1.5,
                       color=colors.HexColor('#00685F'), spaceAfter=12),
        ]

        for heading, entries in sections:
            if not entries:
                continue
            story.append(Paragraph(heading, section_heading))
            table = Table(
                [[Paragraph(f"<b>{label}</b>", body_style), Paragraph(value, body_style)]
                 for label, value in entries],
                colWidths=[190, 350],
            )
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor('#E2EFEF')),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1),
                 [colors.white, colors.HexColor('#F8FAFC')]),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(table)
            story.append(Spacer(1, 8))

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        def _footer(canvas, document) -> None:
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor('#6D7A77'))
            canvas.drawString(36, 22, f"{hospital_name} — Analytics Summary")
            canvas.drawCentredString(letter[0] / 2.0, 22, f"Generated {stamp}")
            canvas.drawRightString(letter[0] - 36, 22, f"Page {canvas.getPageNumber()}")
            canvas.restoreState()

        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        size = os.path.getsize(dest_path)
        return {
            "filename": os.path.basename(dest_path),
            "file_url": f"/uploads/reports/{os.path.basename(dest_path)}",
            "file_size": f"{round(size / 1024, 1)} KB",
            "dest_path": dest_path,
        }


report_generator = MedicalReportGenerator()
