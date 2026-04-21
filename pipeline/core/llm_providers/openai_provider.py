"""OpenAI provider implementation."""

import os
from typing import List, Dict, Any, Optional, cast
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
            model.startswith("claude-") or
            model.startswith("qwen-") or       
            model.startswith("deepseek-") or   
            model.startswith("moonshot-") or   
            model.startswith("kimi-")          
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
        count_usage: bool = True,
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
            
            # Cap completion length (callers e.g. scoring pass max_tokens; must be forwarded).
            # o-series / gpt-5+ use max_completion_tokens; older chat models accept max_tokens.
            if model.startswith(("gpt-5", "o1", "o3", "o4")):
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens
            
            if self._debug_mode:
                print(f"[DEBUG] OpenAI request: model={model}, messages={len(messages)}, json_mode={json_mode}")
            
            resp = self.client.chat.completions.create(**kwargs)
            if resp.usage:
                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens
                if count_usage:
                    self.total_usage_stats["input_tokens"] += input_tokens
                    self.total_usage_stats["output_tokens"] += output_tokens
                
                if self._debug_mode:
                    print(f"[DEBUG] OpenAI Usage (Current): Input={input_tokens}, Output={output_tokens}")
                    if count_usage:
                        print(f"[DEBUG] OpenAI Usage (Total): Input={self.total_usage_stats['input_tokens']}, Output={self.total_usage_stats['output_tokens']}")
            
            choice = resp.choices[0]
            msg = choice.message
            content = msg.content
            if content is None:
                refusal = getattr(msg, "refusal", None)
                fr = getattr(choice, "finish_reason", None)
                raise ValueError(
                    "OpenAI returned no message content "
                    f"(finish_reason={fr!r}, refusal={refusal!r}). "
                    "Often a policy refusal or empty completion — retry or change prompt/category."
                )
            return content.strip()
            
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def chat_completion_with_tools(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        count_usage: bool = True,
    ) -> Dict[str, Any]:
        """One completion turn; may include tool_calls on the assistant message."""
        
        def _make_request():
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": max_tokens,
            }
            if not model.startswith("gpt-5"):
                kwargs["temperature"] = temperature
            
            if self._debug_mode:
                print(
                    f"[DEBUG] OpenAI tools request: model={model}, messages={len(messages)}, "
                    f"tools={len(tools)}"
                )
            
            resp = self.client.chat.completions.create(**kwargs)
            if resp.usage and count_usage:
                self.total_usage_stats["input_tokens"] += resp.usage.prompt_tokens
                self.total_usage_stats["output_tokens"] += resp.usage.completion_tokens
            
            msg = resp.choices[0].message
            out: Dict[str, Any] = {
                "role": cast(str, msg.role),
                "content": (msg.content or "") or "",
            }
            if msg.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return out
        
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def supports_json_mode(self) -> bool:
        return True
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug

