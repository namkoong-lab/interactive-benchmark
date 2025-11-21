"""DeepSeek Provider (OpenRouter Exclusive Version)."""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .utils import retry_with_backoff

load_dotenv()


class DeepSeekProvider(BaseLLMProvider):
    """
    Provider for DeepSeek models via OpenRouter ONLY.
    Forces all requests through OpenRouter and handles model ID formatting automatically.
    """
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._debug_mode: bool = False
    
    def get_provider_name(self) -> str:
        return "deepseek"
    
    def matches_model(self, model: str) -> bool:
        """Match deepseek-* models."""
        return model.lower().startswith("deepseek")
    
    def initialize(self) -> None:
        """Initialize OpenRouter client."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please add it to your .env file to use DeepSeek models."
            )
        
        if self._debug_mode:
            print(f"[DEBUG] DeepSeekProvider initialized in OpenRouter-ONLY mode.")
        
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
        max_tokens: int = 4096, 
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        def _make_request():
            current_model = model
            
            if not current_model.startswith("deepseek/"):
                if current_model == "deepseek-reasoner":
                    current_model = "deepseek/deepseek-r1"
                elif current_model == "deepseek-chat":
                    current_model = "deepseek/deepseek-chat"
                else:
                    current_model = f"deepseek/{current_model}"
            
            is_reasoner = "r1" in current_model or "reasoner" in current_model
            
            kwargs: Dict[str, Any] = {
                "model": current_model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            
            if not is_reasoner:
                kwargs["temperature"] = temperature
            
            if json_mode and not is_reasoner:
                kwargs["response_format"] = {"type": "json_object"}
            
            kwargs["extra_body"] = {}
            if is_reasoner:
                kwargs["extra_body"]["include_reasoning"] = True

            if self._debug_mode:
                print(f"[DEBUG] OpenRouter DeepSeek Request: model={current_model}")
            
            resp = self.client.chat.completions.create(**kwargs)
            message = resp.choices[0].message
            content = message.content or ""
            
            reasoning = None
            if hasattr(message, 'reasoning') and message.reasoning:
                reasoning = message.reasoning
            
            if reasoning:
                if self._debug_mode:
                    print(f"[DEBUG] DeepSeek Reasoning captured ({len(reasoning)} chars)")
                return f"<think>\n{reasoning}\n</think>\n\n{content}"
            
            if not content:
                return "QUESTION: (System Error: Empty response from OpenRouter/DeepSeek)"
            
            return content.strip()
        
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def supports_json_mode(self) -> bool:
        return True
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug