from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db, get_current_active_user
from app.models.user import User
from app.ai_core import extract_text_from_pdf
from app.services.ai_service import AIService, get_ai_service

router = APIRouter()

class SymptomIntakeRequest(BaseModel):
    symptoms: str
    age: Optional[str] = "30"
    gender: Optional[str] = "unspecified"

class ChatRequest(BaseModel):
    query: str
    report_context: Optional[str] = ""
    chat_history: Optional[List[Dict[str, str]]] = []

class TextAnalysisRequest(BaseModel):
    patient_name: Optional[str] = "Patient"
    age: Optional[str] = "30"
    gender: Optional[str] = "unspecified"
    report_text: str

@router.get("/health", response_model=Dict[str, Any])
async def ai_core_health_check(
    ai_service: AIService = Depends(get_ai_service)
) -> Any:
    """
    Dedicated AI Core Health Check Endpoint.
    Verifies LLM model manager connectivity and provider status.
    """
    return ai_service.analysis_agent.model_manager.check_health()


@router.post("/symptom-intake", response_model=Dict[str, Any])
async def process_symptom_intake(
    request: SymptomIntakeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
    ai_service: AIService = Depends(get_ai_service)
) -> Any:
    """
    AI Symptom Intake Endpoint:
    Receives patient symptom description via JSON payload, processes it using AI Core reasoning engine,
    computes urgency & specialty recommendation, and stores structured findings in PostgreSQL database.
    Protected by existing JWT Authentication.
    """
    if not request.symptoms or not request.symptoms.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Symptom text description is required."
        )

    user_display = getattr(current_user, "email", "Patient").split("@")[0].capitalize()
    return await ai_service.process_symptom_intake(
        db=db,
        user_id=current_user.id,
        user_name=user_display,
        symptom_text=request.symptoms,
        age=request.age or "30",
        gender=request.gender or "unspecified"
    )


@router.post("/analyze-report", response_model=Dict[str, Any])
async def analyze_medical_report(
    patient_name: Optional[str] = Form("Patient"),
    age: Optional[str] = Form("30"),
    gender: Optional[str] = Form("unspecified"),
    file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_active_user),
    ai_service: AIService = Depends(get_ai_service)
) -> Any:
    """
    Analyzes an uploaded medical report PDF file using the internal AI Service.
    Protected by existing JWT Authentication.
    """
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="PDF report file is required."
        )
    
    extracted_text = extract_text_from_pdf(file.file)
    if extracted_text.startswith("Error") or "exceeds" in extracted_text or "too short" in extracted_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=extracted_text)

    return ai_service.analyze_report(
        report_text=extracted_text,
        patient_name=patient_name or current_user.full_name or "Patient",
        age=age,
        gender=gender
    )

@router.post("/analyze-text", response_model=Dict[str, Any])
async def analyze_medical_text(
    request: TextAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    ai_service: AIService = Depends(get_ai_service)
) -> Any:
    """
    Analyzes raw text blood work or symptoms via JSON payload.
    Protected by existing JWT Authentication.
    """
    return ai_service.analyze_report(
        report_text=request.report_text,
        patient_name=request.patient_name or current_user.full_name or "Patient",
        age=request.age,
        gender=request.gender
    )

@router.post("/chat", response_model=Dict[str, Any])
async def chat_with_ai(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    ai_service: AIService = Depends(get_ai_service)
) -> Any:
    """
    Executes RAG conversational question answering over report context.
    Protected by existing JWT Authentication.
    """
    return ai_service.ask_chat_agent(
        query=request.query,
        report_context=request.report_context,
        chat_history=request.chat_history
    )
