import os
import json
import time
import random
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment.")
        print(f"[DEBUG] OpenAI API key found: {api_key[:8]}...")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

_gemini_available = True
try:
    import google.generativeai as genai
    # Test if GenerativeModel is available
    _ = genai.GenerativeModel
    # Check version
    try:
        import google.generativeai
    except:
        pass
except Exception as e:
    genai = None
    _gemini_available = False

_gemini_configured = False

def _ensure_gemini_configured() -> None:
    global _gemini_configured
    if not _gemini_available:
        raise RuntimeError("google-generativeai not installed. Please add it to requirements.txt and pip install.")
    if not _gemini_configured:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment.")
        genai.configure(api_key=api_key)
        _gemini_configured = True

try:
    import anthropic
    _claude_available = True
except ImportError:
    anthropic = None
    _claude_available = False

_anthropic_client: Optional[Any] = None

def _get_anthropic_client() -> Any:
    global _anthropic_client
    if not _claude_available:
        raise RuntimeError("anthropic library not installed. Please `pip install anthropic`.")
    if _anthropic_client is None:
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY is not set in environment.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini-")

def _is_claude_model(model: str) -> bool:
    return model.startswith("claude-")

def _retry_with_backoff(func, max_retries: int = 5, base_delay: float = 1.0, max_delay: float = 60.0):
    """
    Retry a function with exponential backoff and jitter.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                print(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise
            
            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
            total_delay = delay + jitter
            
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {total_delay:.2f}s...")
            time.sleep(total_delay)

def chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 256,
    json_mode: bool = False,
    response_schema: Optional[Dict[str, Any]] = None,
    system_prompt_override: Optional[str] = None,
) -> str:
    """
    Portable chat completion across OpenAI and Gemini.

    messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns text content (string). If json_mode=True, providers are instructed to return JSON only.
    """
    if _is_gemini_model(model):
        return _gemini_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            response_schema=response_schema,
            system_prompt_override=system_prompt_override,
        )
    elif _is_claude_model(model):
        return _claude_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            system_prompt_override=system_prompt_override,
        )
    else:
        return _openai_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )


def _openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> str:
    def _make_request():
        client = _get_openai_client()
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        # Add some debugging info
        print(f"[DEBUG] OpenAI request - Model: {model}, Messages: {len(messages)}, JSON mode: {json_mode}")
        
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()
    
    return _retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)


def _gemini_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema: Optional[Dict[str, Any]],
    system_prompt_override: Optional[str],
) -> str:
    def _make_request():
        _ensure_gemini_configured()
        system_instruction = None
        if system_prompt_override:
            system_instruction = system_prompt_override
        else:
            for m in messages:
                if m.get("role") == "system":
                    system_instruction = m.get("content")
                    break
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                continue
            if role == "assistant":
                parts_role = "model"
            else:
                parts_role = "user"
            contents.append({
                "role": parts_role,
                "parts": [content],
            })

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
            if response_schema is not None:
                generation_config["response_schema"] = response_schema

        model_client = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction,
        )
        resp = model_client.generate_content(
            contents,
            generation_config=generation_config,
        )

        # Handle finish_reason before using resp.text to avoid ValueError
        try:
            candidates = getattr(resp, "candidates", []) or []
            finish_reason = None
            if candidates:
                # Newer SDK: finish_reason is an enum/int on candidate
                finish_reason = getattr(candidates[0], "finish_reason", None)

            # If stopped due to reaching max tokens (finish_reason == 2),
            # assemble partial text from parts instead of using resp.text
            if finish_reason == 2:
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
                # No parts to salvage
                raise RuntimeError(
                    "Gemini stopped due to max_tokens; increase max_tokens in caller"
                )

            # Otherwise, prefer quick accessor
            text = getattr(resp, "text", None)
            if text is not None:
                return (resp.text or "").strip()

            # Fallback: parse first candidate parts
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
            # Fall through to generic accessor error if anything unexpected
            pass

        # Last resort
        return (getattr(resp, "text", "") or "").strip()
    
    return _retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)


def _claude_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    system_prompt_override: Optional[str],
) -> str:
    def _make_request():
        client = _get_anthropic_client()
        # Extract optional system prompt
        system_prompt = system_prompt_override
        if not system_prompt:
            for m in messages:
                if m.get("role") == "system":
                    system_prompt = m.get("content")
                    break

        # Convert messages to Anthropic format: content must be a list of blocks
        claude_messages: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            mapped_role = "user" if role == "user" else "assistant"
            claude_messages.append({
                "role": mapped_role,
                "content": [{"type": "text", "text": m.get("content", "")}],
            })

        # Anthropic expects system as list of content blocks or an empty list
        system_blocks = (
            [{"type": "text", "text": system_prompt}] if system_prompt else []
        )

        resp = client.messages.create(
            model=model,
            system=system_blocks,
            messages=claude_messages,
            temperature=temperature,
        )
        return (resp.content[0].text or "").strip()
    
    return _retry_with_backoff(_make_request, max_retries=5, base_delay=1.0, max_delay=60.0)
