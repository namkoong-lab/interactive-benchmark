"""Qwen Provider (OpenRouter Exclusive Version)."""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .utils import retry_with_backoff

load_dotenv()


class QwenProvider(BaseLLMProvider):
    """
    Provider for Qwen models via OpenRouter ONLY.
    Forces all requests through OpenRouter and handles model ID formatting automatically.
    """
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._debug_mode: bool = False
    
    def get_provider_name(self) -> str:
        return "qwen"
    
    def matches_model(self, model: str) -> bool:
        """Match qwen-* models."""
        return model.lower().startswith("qwen")
    
    def initialize(self) -> None:
        """Initialize OpenRouter client."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please add it to your .env file to use Qwen models."
            )
        
        if self._debug_mode:
            print(f"[DEBUG] QwenProvider initialized in OpenRouter-ONLY mode.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/your-project",
                "X-Title": "AIR Research Framework"
            }
        )
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 2048, 
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        def _make_request():
            current_model = model
            if not current_model.startswith("qwen/"):
                current_model = f"qwen/{current_model}"
                        
            kwargs: Dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            kwargs["extra_body"] = {}
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            if self._debug_mode:
                print(f"[DEBUG] OpenRouter Qwen Request: model={current_model}")
            
            resp = self.client.chat.completions.create(**kwargs)
            message = resp.choices[0].message
            content = message.content
            
            if not content:
                return "QUESTION: (System Error: Empty response from OpenRouter/Qwen)"
            
            return content.strip()
        
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def supports_json_mode(self) -> bool:
        return True
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug