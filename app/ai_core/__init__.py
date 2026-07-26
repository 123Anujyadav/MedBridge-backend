"""
Reusable AI Agent Core Module (Backend/app/ai_core)
Pure Python framework-agnostic AI Brain exports.
"""

from app.ai_core.analysis_agent import AnalysisAgent
from app.ai_core.chat_agent import ChatAgent
from app.ai_core.model_manager import ModelManager
from app.ai_core.prompts import SPECIALIST_PROMPTS
from app.ai_core.utils.pdf_extractor import extract_text_from_pdf
from app.ai_core.utils.validators import validate_pdf_content, validate_pdf_file

__all__ = [
    "AnalysisAgent",
    "ChatAgent",
    "ModelManager",
    "SPECIALIST_PROMPTS",
    "extract_text_from_pdf",
    "validate_pdf_content",
    "validate_pdf_file",
]
