import os
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

_openai_client: Optional[OpenAI] = None

def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client

_gemini_available = True
try:
    import google.generativeai as genai
    # Test if GenerativeModel is available
    _ = genai.GenerativeModel
    # Check version
    try:
        import google.generativeai
        print(f"[DEBUG] Gemini version: {google.generativeai.__version__}")
    except:
        pass
except Exception as e:
    genai = None
    _gemini_available = False
    print(f"[DEBUG] Gemini import failed: {e}")

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
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini-")

def _is_claude_model(model: str) -> bool:
    return model.startswith("claude-")

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
    client = _get_openai_client()
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def _gemini_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    response_schema: Optional[Dict[str, Any]],
    system_prompt_override: Optional[str],
) -> str:
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
        "max_output_tokens": max_tokens,
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
    return (resp.text or "").strip()


def _claude_chat_completion(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    system_prompt_override: Optional[str],
) -> str:
    client = _get_anthropic_client()
    system_prompt = system_prompt_override
    if not system_prompt:
        for m in messages:
            if m.get("role") == "system":
                system_prompt = m.get("content")
                break
            
    user_messages = [m for m in messages if m.get("role") != "system"]
    
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=user_messages,
        temperature=temperature,       
    )
    return resp.content[0].text.strip()
