import logging
import uuid
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.ai_core import AnalysisAgent, ChatAgent, SPECIALIST_PROMPTS
from app.core.ai_provider import get_groq_api_key
from app.intake.domain.specialties import (
    CANONICAL_SPECIALTIES,
    canonicalise_specialty,
)
from app.models.report import Report

logger = logging.getLogger(__name__)

class AIService:
    """
    Internal Service Wrapper exposing the AI Agent Core (app.ai_core)
    to FastAPI handlers and background workers using clean modular architecture.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Same centralised Groq credential as the intake and assistant agents.
        self.api_key = api_key or get_groq_api_key()
        self.analysis_agent = AnalysisAgent(api_key=self.api_key)
        self.chat_agent = ChatAgent(api_key=self.api_key)

    @staticmethod
    def _parse_report_sections(content: str) -> Dict[str, str]:
        """
        Split the clinical report into its numbered sections.

        Keyed by the section number as a string, so a renamed heading does not
        silently drop the field the way a title match would.
        """
        sections: Dict[str, str] = {}
        current: Optional[str] = None
        buffer: List[str] = []
        for line in (content or "").splitlines():
            heading = re.match(r"^\s*#{1,6}\s*(\d{1,2})\s*[.)]\s*(.*)$", line)
            if heading:
                if current is not None:
                    sections[current] = "\n".join(buffer).strip()
                current = heading.group(1)
                buffer = []
            elif current is not None:
                buffer.append(line)
        if current is not None:
            sections[current] = "\n".join(buffer).strip()
        return sections

    @staticmethod
    def _parse_urgency(risk_section: str) -> str:
        """
        Map the Clinical Risk Level section onto the stored urgency.

        Checked most severe first: "moderate to high" is a high-risk statement,
        and testing "low" first would file it as low.
        """
        text = (risk_section or "").lower()
        for needle, level in (
            ("critical", "critical"), ("emergency", "critical"),
            ("high", "high"), ("severe", "high"), ("urgent", "high"),
            ("moderate", "medium"), ("medium", "medium"),
            ("low", "low"), ("mild", "low"), ("minimal", "low"),
        ):
            if needle in text:
                return level
        return "medium"

    @staticmethod
    def _parse_symptoms(symptom_section: str) -> List[str]:
        """Pull the bullet list out of the model's Symptom Summary section."""
        found: List[str] = []
        for line in (symptom_section or "").splitlines():
            item = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", line).strip()
            item = item.strip("*_` ")
            if 2 < len(item) < 120:
                found.append(item)
        return found[:10]

    async def process_symptom_intake(
        self,
        db: AsyncSession,
        user_id: uuid.UUID,
        user_name: str,
        symptom_text: str,
        age: str = "30",
        gender: str = "unspecified"
    ) -> Dict[str, Any]:
        """
        Executes AI Symptom Intake analysis:
        1. Runs AnalysisAgent reasoning on symptom intake text.
        2. Computes urgency, recommended specialty, confidence, and extracted symptoms.
        3. Persists structured report results into PostgreSQL database (reports table).
        """
        data_payload = {
            "patient_name": user_name,
            "age": age,
            "gender": gender,
            "report": f"PATIENT SYMPTOM INTAKE REPORT:\n{symptom_text}"
        }

        clinical_prompt = (
            "You are an expert Clinical Decision Support AI Assistant. Analyze the patient symptom intake and output a comprehensive structured medical report containing EXACTLY the following Markdown headers:\n\n"
            "### 1. Chief Complaint\n"
            "### 2. Present Illness Summary\n"
            "### 3. Symptom Summary\n"
            "### 4. Possible Differential Diagnosis\n"
            "### 5. Clinical Risk Level\n"
            "### 6. Red Flag Symptoms\n"
            "### 7. Suggested Laboratory Tests\n"
            "### 8. Suggested Imaging\n"
            "### 9. Suggested Medicines (Recommendation Only)\n"
            "### 10. Lifestyle Advice\n"
            "### 11. Follow-up Recommendation\n"
            "### 12. Emergency Warning\n"
            "### 13. AI Confidence Score\n"
            "### 14. Timestamp\n"
            "### 15. Recommended Specialty\n"
            "\n"
            "Section 5 must contain exactly one of: Low, Medium, High, Critical.\n"
            "Section 15 must contain exactly one specialty name and nothing else, "
            "chosen from: " + ", ".join(CANONICAL_SPECIALTIES) + ".\n"
        )

        analysis_result = self.analysis_agent.analyze_report(
            data=data_payload,
            system_prompt=clinical_prompt
        )

        content_text = analysis_result.get("content", "") if isinstance(analysis_result, dict) else str(analysis_result)

        # Read the structured fields out of the numbered sections the prompt
        # mandates, rather than scanning the whole document for keywords.
        #
        # Scanning was not a weaker version of this — it was a constant. The
        # prompt requires a "12. Emergency Warning" heading, so the substring
        # "emergency" is present in every reply and urgency was unconditionally
        # "high". The specialty search looked for department nouns ("Neurology")
        # in prose that says "migraine" or "neurologist", so nothing ever
        # matched and every case fell through to General Medicine — which is
        # also what picks the doctor a case is routed to, below.
        sections = self._parse_report_sections(content_text)

        urgency_level = self._parse_urgency(sections.get("5", ""))
        recommended_specialty = canonicalise_specialty(sections.get("15", ""))

        # Symptom tags come from the model's own "Symptom Summary" section.
        # Splitting the patient's raw sentence on "." only ever echoed their
        # words back, and any sentence of 60 characters or more produced the
        # literal placeholder "Symptom intake analyzed".
        extracted_symptoms = self._parse_symptoms(sections.get("3", ""))
        if not extracted_symptoms:
            extracted_symptoms = [
                s.strip() for s in symptom_text.split(".") if 3 < len(s.strip()) < 120
            ]
        if not extracted_symptoms:
            extracted_symptoms = ["Symptom intake analyzed"]

        # This pipeline does not measure a confidence score. It previously
        # recorded a hardcoded 94.0, which surfaced downstream as a measured
        # "94% — High Confidence" badge on report cards and in the clinical
        # review. 0.0 is read everywhere as "never scored" and suppresses the
        # badge entirely, which is the honest representation.
        confidence_score = 0.0

        # Save results in PostgreSQL database using Report SQLAlchemy model (status=pending for Doctor approval)
        new_report = Report(
            patient_id=user_id,
            patient_name=user_name,
            type="ai_symptom_intake",
            title="AI Clinical Decision Support Report",
            summary=f"Symptom intake: {symptom_text[:120]}...",
            content=content_text,
            date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            status="pending",
            ai_generated=True,
            ai_confidence_score=confidence_score,
            tags=extracted_symptoms[:5],
            vitals={
                "urgency": urgency_level,
                "recommended_specialty": recommended_specialty,
                "confidence": confidence_score
            }
        )


        db.add(new_report)

        # Route AI triage case to doctor consultation queue if patient profile exists
        from sqlalchemy import select
        from app.models.doctor import Doctor
        from app.models.patient import Patient
        from app.models.case import Case

        pat_res = await db.execute(select(Patient).where(Patient.id == user_id).limit(1))
        patient_obj = pat_res.scalars().first()
        if not patient_obj:
            patient_obj = Patient(
                id=user_id,
                first_name=user_name,
                last_name="Patient",
                date_of_birth="1995-01-01",
                gender=gender if gender != "unspecified" else "male"
            )
            db.add(patient_obj)
            await db.flush()

        # Match on the AI-recommended specialty rather than taking whichever
        # doctor row sorts first. Arbitrary assignment is not just poor routing:
        # ownership across this platform is derived from "has a case with this
        # patient", so assigning an unrelated clinician hands them access to
        # that patient's full record, reports and prescriptions.
        doc_res = await db.execute(
            select(Doctor)
            .where(Doctor.specialty.ilike(recommended_specialty))
            .where(Doctor.verification_status == "verified")
            .order_by(Doctor.total_cases.asc())
            .limit(1)
        )
        assigned_doc = doc_res.scalars().first()

        if assigned_doc is None:
            # No verified doctor in that specialty. The case is still created so
            # nothing is lost, but it stays unassigned for triage rather than
            # being routed to someone unqualified for it.
            logger.warning(
                "[AI_TRIAGE_UNROUTED] No verified %s doctor available; "
                "case left unassigned for manual triage.",
                recommended_specialty,
            )

        if patient_obj:
            new_case = Case(
                patient_id=patient_obj.id,
                patient_name=f"{patient_obj.first_name} {patient_obj.last_name}",
                patient_age=int(age) if age.isdigit() else 30,
                patient_gender=gender,
                doctor_id=assigned_doc.id if assigned_doc else None,
                doctor_name=(
                    f"{assigned_doc.first_name} {assigned_doc.last_name}"
                    if assigned_doc
                    else None
                ),
                specialty=recommended_specialty,
                symptom_summary=symptom_text[:200],
                urgency_level=urgency_level,
                status="routed" if assigned_doc else "ai_processing",
                ai_confidence_score=confidence_score / 100.0,
                notes=f"AI Clinical Decision Support Triage: {recommended_specialty}"
            )
            db.add(new_case)
            await db.flush()
            # Link the intake report to the case it produced. They were created
            # as unrelated siblings, so the report carried a NULL case_id and no
            # consumer could tell which case it belonged to.
            new_report.case_id = new_case.id

            # Notify the assigned clinician. `notify_case_doctor` returns
            # without sending when the case is unassigned, so an unrouted case
            # never lands in an unrelated doctor's inbox.
            from app.services.notifications import notification_service

            await notification_service.safe_notify_case_doctor(
                db, case=new_case,
                category="case", type="case_assigned",
                title="New Patient Case Assigned",
                message=(
                    f"{new_case.patient_name} — {recommended_specialty}. "
                    f"{symptom_text[:120]}"
                ),
                priority="high" if urgency_level in ("high", "critical") else "medium",
                action_url=f"/doctor/cases?case={new_case.id}",
                action_label="Open Case",
                group_key="case_assigned",
                dedupe_key=f"case_assigned:{new_case.id}",
            )
            await notification_service.safe_notify_case_doctor(
                db, case=new_case,
                category="ai", type="ai_analysis_ready",
                title="New AI Analysis Ready",
                message=f"AI triage completed for {new_case.patient_name}.",
                priority="medium",
                action_url=f"/doctor/ai-reports?case={new_case.id}",
                action_label="Review AI Analysis",
                group_key="ai_analysis_ready",
                dedupe_key=f"ai_analysis_ready:{new_case.id}",
            )
            if urgency_level == "critical":
                await notification_service.safe_notify_case_doctor(
                    db, case=new_case,
                    category="ai", type="critical_urgency",
                    title="Critical AI Urgency Detected",
                    message=(
                        f"AI triage flagged {new_case.patient_name} as critical. "
                        "Immediate review recommended."
                    ),
                    priority="critical",
                    action_url=f"/doctor/cases?case={new_case.id}",
                    action_label="Open Case",
                    group_key="critical_urgency",
                    dedupe_key=f"critical_urgency:{new_case.id}",
                )

        await db.commit()
        await db.refresh(new_report)

        return {
            "success": True,
            "report_id": str(new_report.id),
            "status": "pending_review",
            "extractedSymptoms": extracted_symptoms[:5],
            "urgency": urgency_level,
            "specialty": recommended_specialty,
            "confidence": confidence_score,
            "analysis": content_text
        }

    def analyze_report(
        self,
        report_text: str,
        patient_name: str = "Patient",
        age: str = "30",
        gender: str = "unspecified",
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes medical report analysis using AnalysisAgent.
        """
        data_payload = {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "report": report_text
        }
        
        return self.analysis_agent.analyze_report(
            data=data_payload,
            system_prompt=SPECIALIST_PROMPTS.get("comprehensive_analyst", ""),
            chat_history=chat_history
        )

    def ask_chat_agent(
        self,
        query: str,
        report_context: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes RAG conversational retrieval using ChatAgent and FAISS.
        """
        vectorstore = None
        if report_context and report_context.strip():
            vectorstore = self.chat_agent.initialize_vector_store(report_context)

        response_text = self.chat_agent.get_response(
            query=query,
            vectorstore=vectorstore,
            chat_history=chat_history or []
        )
        
        return {
            "success": True,
            "query": query,
            "response": response_text
        }


# Global singleton instance for FastAPI dependency injection
_ai_service_instance: Optional[AIService] = None

def get_ai_service() -> AIService:
    """
    FastAPI Dependency Injection Provider for AIService.
    """
    global _ai_service_instance
    if _ai_service_instance is None:
        _ai_service_instance = AIService()
    return _ai_service_instance
