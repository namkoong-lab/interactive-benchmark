"""Google Gemini provider implementation."""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from .base import BaseLLMProvider
from .utils import retry_with_backoff

load_dotenv()

# Try importing Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models."""
    
    def __init__(self):
        self._configured = False
        self._debug_mode = False
    
    def get_provider_name(self) -> str:
        return "gemini"
    
    def matches_model(self, model: str) -> bool:
        return model.startswith("gemini-")
    
    def initialize(self) -> None:
        """Configure Gemini API."""
        if not GEMINI_AVAILABLE:
            raise RuntimeError(
                "google-generativeai not installed. "
                "Install with: pip install google-generativeai"
            )
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in environment")
        
        genai.configure(api_key=api_key)
        self._configured = True
        
        if self._debug_mode:
            print(f"[DEBUG] Gemini configured with API key: {api_key[:8]}...")
    
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
            # Extract system instruction
            system_instruction = system_prompt_override
            if not system_instruction:
                for m in messages:
                    if m.get("role") == "system":
                        system_instruction = m.get("content")
                        break
            
            # Convert messages (skip system, map assistant->model)
            contents = []
            for m in messages:
                role = m.get("role")
                if role == "system":
                    continue
                parts_role = "model" if role == "assistant" else "user"
                contents.append({
                    "role": parts_role,
                    "parts": [m.get("content")],
                })
            
            # Generation config
            generation_config: Dict[str, Any] = {"temperature": temperature}
            if json_mode:
                generation_config["response_mime_type"] = "application/json"
                if response_schema:
                    generation_config["response_schema"] = response_schema
            
            if self._debug_mode:
                print(f"[DEBUG] Gemini request: model={model}, json_mode={json_mode}")
            
            # Create model and generate
            model_client = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_instruction,
            )
            resp = model_client.generate_content(
                contents,
                generation_config=generation_config,
            )
            
            # Extract text (handle various response formats)
            try:
                candidates = getattr(resp, "candidates", []) or []
                finish_reason = None
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)

                if finish_reason == 2:  # MAX_TOKENS
                    content_obj = getattr(candidates[0], "content", None)
                    parts = getattr(content_obj, "parts", None) if content_obj else None
                    if parts:
                        collected: list[str] = []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                collected.append(t)
                        partial = "\n".join(collected).strip()
                        if partial:
                            return partial
                    raise RuntimeError(
                        "Gemini stopped due to max_tokens; increase max_tokens in caller"
                    )

                text = getattr(resp, "text", None)
                if text is not None:
                    return (resp.text or "").strip()

                if candidates:
                    content_obj = getattr(candidates[0], "content", None)
                    parts = getattr(content_obj, "parts", None) if content_obj else None
                    if parts:
                        collected: list[str] = []
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                collected.append(t)
                        if collected:
                            return "\n".join(collected).strip()
            except Exception as _:
                pass

            result = (getattr(resp, "text", "") or "").strip()
            if not result:
                raise ValueError("Gemini returned empty response")
            return result
        
        return retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
    
    def supports_json_mode(self) -> bool:
        return True
    
    def supports_response_schema(self) -> bool:
        return True
    
    def set_debug_mode(self, debug: bool):
        self._debug_mode = debug

