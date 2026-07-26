import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from app.ai_core.model_manager import ModelManager
from app.ai_core.utils.validators import sanitize_prompt_input, validate_llm_output

logger = logging.getLogger("app.ai_core.analysis_agent")

class AnalysisAgent:
    """
    Agent responsible for managing report analysis, rate limiting, input sanitization,
    output validation, and implementing in-context learning from previous analyses.
    Framework-agnostic (zero dependency on Streamlit / UI state).
    """
    
    def __init__(self, api_key: Optional[str] = None, daily_limit: int = 50):
        self.model_manager = ModelManager(api_key=api_key)
        self.analysis_count = 0
        self.last_analysis = datetime.now(timezone.utc)
        self.analysis_limit = daily_limit
        self.models_used = {}
        self.knowledge_base = {}

    def check_rate_limit(self):
        """Check if analysis daily limit is reached."""
        now = datetime.now(timezone.utc)
        time_until_reset = timedelta(days=1) - (now - self.last_analysis)
        hours, remainder = divmod(time_until_reset.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if time_until_reset.days < 0:
            self.analysis_count = 0
            self.last_analysis = now
            return True, None
        
        if self.analysis_count >= self.analysis_limit:
            error_msg = f"Daily analysis limit reached. Reset in {hours}h {minutes}m"
            return False, error_msg
        return True, None

    def analyze_report(
        self,
        data: Any,
        system_prompt: str,
        check_only: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze report data using input sanitization, dynamic context building, and output validation.
        """
        can_analyze, error_msg = self.check_rate_limit()
        if not can_analyze:
            return {"success": False, "error": error_msg}
        
        if check_only:
            return {"success": can_analyze, "error": error_msg}
        
        processed_data = self._preprocess_data(data)
        
        enhanced_prompt = (
            self._build_enhanced_prompt(system_prompt, processed_data, chat_history) 
            if chat_history else system_prompt
        )
        
        # Invoke multi-tier ModelManager with timeout safeguards
        result = self.model_manager.generate_analysis(processed_data, enhanced_prompt)
        
        if result.get("success"):
            # Output validation & safety disclaimer injection
            raw_content = result.get("content", "")
            validated_content = validate_llm_output(raw_content)
            result["content"] = validated_content

            self._update_analytics(result)
            self._update_knowledge_base(processed_data, validated_content)
        
        return result
    
    def _update_analytics(self, result: Dict[str, Any]):
        """Update internal analytics after successful analysis."""
        self.analysis_count += 1
        self.last_analysis = datetime.now(timezone.utc)
        
        model_used = result.get("model_used", "unknown")
        self.models_used[model_used] = self.models_used.get(model_used, 0) + 1
    
    def _update_knowledge_base(self, data: Dict[str, Any], analysis: str):
        """
        Update local memory knowledge base with new analysis results for in-context learning.
        """
        if not isinstance(data, dict) or 'report' not in data:
            return
            
        report_text = str(data['report']).lower()
        patient_profile = f"{data.get('age', 'unknown')}-{data.get('gender', 'unknown')}"
        
        key_indicators = [
            "hemoglobin", "glucose", "cholesterol", "triglycerides", 
            "hdl", "ldl", "wbc", "rbc", "platelet", "creatinine"
        ]
        
        for indicator in key_indicators:
            if indicator in report_text and indicator in analysis.lower():
                if indicator not in self.knowledge_base:
                    self.knowledge_base[indicator] = {}
                
                if patient_profile not in self.knowledge_base[indicator]:
                    self.knowledge_base[indicator][patient_profile] = []
                
                lines = analysis.split('\n')
                relevant_lines = [l for l in lines if indicator in l.lower()]
                if relevant_lines:
                    if len(self.knowledge_base[indicator][patient_profile]) >= 3:
                        self.knowledge_base[indicator][patient_profile].pop(0)
                    self.knowledge_base[indicator][patient_profile].append(relevant_lines[0])
    
    def _build_enhanced_prompt(self, system_prompt: str, data: Any, chat_history: Optional[List[Dict[str, str]]]):
        """Build enhanced system prompt using previous analyses and chat history context."""
        enhanced_prompt = system_prompt
        
        if isinstance(data, dict) and 'report' in data:
            kb_context = self._get_knowledge_base_context(data)
            if kb_context:
                enhanced_prompt += "\n\n## Relevant Learning From Previous Analyses\n" + kb_context
        
        if chat_history:
            session_context = self._get_session_context(chat_history)
            if session_context:
                enhanced_prompt += "\n\n## Current Session History\n" + session_context
        
        return enhanced_prompt
    
    def _get_knowledge_base_context(self, data: Dict[str, Any]) -> str:
        """Extract relevant context from knowledge base."""
        if not self.knowledge_base:
            return ""
            
        report_text = str(data.get('report', '')).lower()
        patient_profile = f"{data.get('age', 'unknown')}-{data.get('gender', 'unknown')}"
        
        context_items = []
        for indicator, profiles in self.knowledge_base.items():
            if indicator in report_text:
                if patient_profile in profiles:
                    for insight in profiles[patient_profile]:
                        context_items.append(f"- {indicator} (similar patient profile): {insight}")
                
                for profile, insights in profiles.items():
                    if profile != patient_profile:
                        for insight in insights:
                            context_items.append(f"- {indicator} (other patient profile): {insight}")
        
        if len(context_items) > 5:
            context_items = context_items[:5]
            
        return "\n".join(context_items) if context_items else ""
    
    def _get_session_context(self, chat_history: List[Dict[str, str]]) -> str:
        """Extract recent exchange pairs from chat history."""
        if not chat_history or len(chat_history) < 2:
            return ""
            
        context_items = []
        for i in range(len(chat_history) - 1, 0, -2):
            if i >= 1 and chat_history[i-1].get('role') == 'user' and chat_history[i].get('role') == 'assistant':
                user_msg = sanitize_prompt_input(str(chat_history[i-1].get('content', '')), max_length=500)
                ai_msg = sanitize_prompt_input(str(chat_history[i].get('content', '')), max_length=500)
                
                context_items.append(f"User: {user_msg}\nAssistant: {ai_msg}")
                
                if len(context_items) >= 2:
                    break
                    
        return "\n\n".join(reversed(context_items)) if context_items else ""
    
    def _preprocess_data(self, data: Any) -> Any:
        """Sanitize and pre-process data dictionary before submitting to LLM."""
        if isinstance(data, dict):
            raw_report = str(data.get("report", ""))
            sanitized_report = sanitize_prompt_input(raw_report)
            return {
                "patient_name": sanitize_prompt_input(str(data.get("patient_name", "")), max_length=100),
                "age": sanitize_prompt_input(str(data.get("age", "")), max_length=10),
                "gender": sanitize_prompt_input(str(data.get("gender", "")), max_length=20),
                "report": sanitized_report
            }
        return sanitize_prompt_input(str(data))
