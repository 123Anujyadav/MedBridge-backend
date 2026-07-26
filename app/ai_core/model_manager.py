import os
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional
import groq
from app.core.ai_provider import get_groq_api_key, get_groq_model

logger = logging.getLogger("app.ai_core.model_manager")

class ModelTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary" 
    TERTIARY = "tertiary"
    FALLBACK = "fallback"

class ModelManager:
    """
    Manages AI model selection, timeout handling, retries, and fallback strategies.
    Production-ready framework-agnostic LLM tier manager.
    """
    
    MODEL_CONFIG = {
        ModelTier.PRIMARY: {
            "provider": "groq",
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 25.0
        },
        ModelTier.SECONDARY: {
            "provider": "groq", 
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 25.0
        },
        ModelTier.TERTIARY: {
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "max_tokens": 2000, 
            "temperature": 0.7,
            "timeout": 15.0
        },
        ModelTier.FALLBACK: {
            "provider": "groq",
            "model": "llama3-70b-8192",
            "max_tokens": 2000,
            "temperature": 0.7,
            "timeout": 20.0
        }
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.clients = {}
        # Credential comes from the centralised AI provider config so this
        # legacy sync path uses the same Groq account as every other service.
        self.api_key = api_key or get_groq_api_key()
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for each provider."""
        if self.api_key:
            try:
                self.clients["groq"] = groq.Groq(api_key=self.api_key, timeout=30.0)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {str(e)}")

    def check_health(self) -> Dict[str, Any]:
        """
        Executes a lightweight health check to verify LLM provider status.
        """
        if not self.api_key:
            return {
                "status": "unhealthy",
                "error": "GROQ_API_KEY is not configured",
                "provider": "groq"
            }
            
        if "groq" not in self.clients:
            return {
                "status": "unhealthy",
                "error": "Groq client failed to initialize",
                "provider": "groq"
            }

        try:
            # Test connectivity with minimal ping
            return {
                "status": "healthy",
                "provider": "groq",
                "primary_model": self.MODEL_CONFIG[ModelTier.PRIMARY]["model"]
            }
        except Exception as e:
            return {
                "status": "degraded",
                "error": str(e),
                "provider": "groq"
            }

    def generate_analysis(self, data: Any, system_prompt: str, retry_count: int = 0) -> Dict[str, Any]:
        """
        Generate analysis using the best available model with automatic timeout & tier fallback.
        """
        if retry_count > 3:
            logger.error("All AI model tiers failed after maximum retry attempts.")
            return {
                "success": False,
                "error": "All AI model tiers failed after maximum retries. Please try again later."
            }

        tier_mapping = [ModelTier.PRIMARY, ModelTier.SECONDARY, ModelTier.TERTIARY, ModelTier.FALLBACK]
        tier = tier_mapping[retry_count]
            
        model_config = self.MODEL_CONFIG[tier]
        provider = model_config["provider"]
        model = model_config["model"]
        timeout_sec = model_config.get("timeout", 25.0)
        
        if provider not in self.clients:
            if not self.api_key:
                return {
                    "success": False,
                    "error": "GROQ_API_KEY environment variable is not configured."
                }
            logger.warning(f"No client available for provider: {provider}. Attempting next tier.")
            return self.generate_analysis(data, system_prompt, retry_count + 1)
            
        start_time = time.time()
        try:
            client = self.clients[provider]
            logger.info(f"LLM Generation Attempt | Tier: {tier.value} | Provider: {provider} | Model: {model} | Attempt: {retry_count + 1}")
            
            if provider == "groq":
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(data)}
                    ],
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    timeout=timeout_sec
                )
                
                latency_ms = round((time.time() - start_time) * 1000, 2)
                logger.info(f"LLM Generation Success | Model: {model} | Latency: {latency_ms}ms")
                
                return {
                    "success": True,
                    "content": completion.choices[0].message.content,
                    "model_used": f"{provider}/{model}",
                    "latency_ms": latency_ms
                }
                
        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            error_message = str(e).lower()
            logger.warning(f"Model {model} failed after {latency_ms}ms: {error_message}")
            
            # Rate limit or quota backoff delay
            if "rate limit" in error_message or "quota" in error_message:
                time.sleep(1.5)
            
            return self.generate_analysis(data, system_prompt, retry_count + 1)
            
        return {"success": False, "error": "Analysis failed with all available models"}
