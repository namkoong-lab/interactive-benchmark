"""Anthropic Claude provider implementation."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .utils import retry_with_backoff

load_dotenv()

# Try importing Anthropic
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False


class ClaudeProvider(BaseLLMProvider):
    """Provider for Anthropic Claude models."""
    
    def __init__(self):
        self.client = None
        self._debug_mode = False
    
    def get_provider_name(self) -> str:
        return "claude"
    
    def matches_model(self, model: str) -> bool:
        return model.startswith("claude-")
    
    def initialize(self) -> None:
        """Initialize Anthropic client."""
        if not ANTHROPIC_AVAILABLE:
            raise RuntimeError(
                "anthropic library not installed. "
                "Install with: pip install anthropic"
            )
        
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY not set in environment")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        if self._debug_mode:
            print(f"[DEBUG] Claude client initialized with API key: {api_key[:8]}...")
    
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
            # Extract system prompt
            system_prompt = system_prompt_override
            if not system_prompt:
                for m in messages:
                    if m.get("role") == "system":
                        system_prompt = m.get("content")
                        break
            
            # Convert messages (skip system)
            claude_messages = []
            for m in messages:
                role = m.get("role")
                if role == "system":
                    continue
                mapped_role = "user" if role == "user" else "assistant"
                claude_messages.append({
                    "role": mapped_role,
                    "content": [{"type": "text", "text": m.get("content", "")}],
                })
            
            system_blocks = (
                [{"type": "text", "text": system_prompt}] 
                if system_prompt else []
            )
            
            if self._debug_mode:
                print(f"[DEBUG] Claude request: model={model}, messages={len(claude_messages)}")
            
            # Stream response
            resp = self.client.messages.create(
                model=model,
                system=system_blocks,
                messages=claude_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            
            # Accumulate streamed chunks
            response_text = ""
            for chunk in resp:
                if chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        response_text += chunk.delta.text
            
            result = response_text.strip()
            if not result:
                raise ValueError("Claude returned empty response")
            return result
        
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug

