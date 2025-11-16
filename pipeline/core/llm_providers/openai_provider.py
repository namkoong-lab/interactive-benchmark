"""OpenAI provider implementation."""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .utils import retry_with_backoff

load_dotenv()


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models (GPT-3.5, GPT-4, O1, etc.)."""
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self._debug_mode: bool = False
        self.total_usage_stats = {"input_tokens": 0, "output_tokens": 0}
        
    def get_usage_stats(self) -> Dict[str, int]:
        """Returns the cumulative token usage."""
        return self.total_usage_stats
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def matches_model(self, model: str) -> bool:
        """Match GPT models and other non-prefixed models (default provider)."""
        # OpenAI handles: gpt-*, o1-*, or anything not matching other providers
        return not (
            model.startswith("gemini-") or 
            model.startswith("claude-")
        )
    
    def initialize(self) -> None:
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        
        if self._debug_mode:
            print(f"[DEBUG] OpenAI API key found: {api_key[:8]}...")
        
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 256,
        json_mode: bool = False,
        response_schema: Optional[Dict[str, Any]] = None,
        system_prompt_override: Optional[str] = None,
    ) -> str:
        def _make_request():
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            
            # GPT-5 models don't support temperature parameter
            if not model.startswith("gpt-5"):
                kwargs["temperature"] = temperature
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            if self._debug_mode:
                print(f"[DEBUG] OpenAI request: model={model}, messages={len(messages)}, json_mode={json_mode}")
            
            resp = self.client.chat.completions.create(**kwargs)
            if resp.usage:
                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens
                self.total_usage_stats["input_tokens"] += input_tokens
                self.total_usage_stats["output_tokens"] += output_tokens
                
                if self._debug_mode:
                    print(f"[DEBUG] OpenAI Usage (Current): Input={input_tokens}, Output={output_tokens}")
                    print(f"[DEBUG] OpenAI Usage (Total): Input={self.total_usage_stats['input_tokens']}, Output={self.total_usage_stats['output_tokens']}")
            
            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI returned None for message content")
            return content.strip()
            
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def supports_json_mode(self) -> bool:
        return True
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug

