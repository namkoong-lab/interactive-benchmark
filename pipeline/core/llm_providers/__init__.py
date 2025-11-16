"""
LLM Provider system with auto-discovery.

To add a new provider:
1. Create a file named `*_provider.py` in this directory
2. Define a class that inherits from BaseLLMProvider

Usage:
    from pipeline.core.llm_providers import chat_completion, set_debug_mode
    
    response = chat_completion(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4",
        temperature=0.7
    )
    
    set_debug_mode(True)
"""

import os
import importlib
import inspect
from pathlib import Path
from .base import BaseLLMProvider
from .registry import ProviderRegistry


def _auto_discover_providers():
    """Auto-discover and register all provider classes."""
    registry = ProviderRegistry()
    providers_dir = Path(__file__).parent
    
    # Scan for *_provider.py files (excluding base.py, utils.py, registry.py)
    provider_files = sorted(providers_dir.glob("*_provider.py"))
    
    for filepath in provider_files:
        module_name = filepath.stem
        
        try:
            # Dynamically import the module
            module = importlib.import_module(
                f".{module_name}", 
                package="pipeline.core.llm_providers"
            )
            
            # Find all classes that inherit from BaseLLMProvider
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseLLMProvider) and 
                    obj is not BaseLLMProvider):
                    # Instantiate and register
                    provider = obj()
                    registry.register(provider)
                    print(f"✓ Loaded LLM provider: {provider.get_provider_name()} ({name})")
        
        except Exception as e:
            print(f"⚠ Failed to load provider from {filepath.name}: {e}")
            # Continue loading other providers
            continue
    
    return registry


# Global registry - initialized on import
_registry = _auto_discover_providers()


# Public API
def chat_completion(*args, **kwargs):
    """
    Central chat completion API for all providers.
    
    Args:
        messages: List of {"role": "system"|"user"|"assistant", "content": str}
        model: Model name (e.g., "gpt-4", "gemini-pro", "claude-3-opus")
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum tokens in response (default: 256)
        json_mode: Request JSON-formatted response (default: False)
        response_schema: Optional JSON schema for structured output
        system_prompt_override: Override system message if needed
    
    Returns:
        Response text as string
    """
    return _registry.chat_completion(*args, **kwargs)


def set_debug_mode(debug: bool):
    """
    Set debug mode for all providers.
    
    Args:
        debug: If True, enables debug logging
    """
    _registry.set_debug_mode(debug)


def list_providers():
    """
    List all registered provider names.
    
    Returns:
        List of provider names (e.g., ['openai', 'gemini', 'claude'])
    """
    return _registry.list_providers()

def get_total_usage_stats() -> dict:
    """
    Get cumulative usage stats from ALL registered providers.
    """
    total_stats = {"input_tokens": 0, "output_tokens": 0}
    
    # _registry 是在第 58 行创建的 ProviderRegistry 实例
    # 根据 registry.py，provider 列表存储在 ._providers 属性中
    
    if not hasattr(_registry, '_providers'):
        print("Warning: Cannot get token stats. ProviderRegistry (in registry.py) "
              "does not have a '_providers' attribute to iterate over.")
        return total_stats
        
    # 遍历注册表中的所有 provider 实例
    for provider in _registry._providers:
        if hasattr(provider, 'get_usage_stats'):
            stats = provider.get_usage_stats() #
            total_stats["input_tokens"] += stats.get("input_tokens", 0)
            total_stats["output_tokens"] += stats.get("output_tokens", 0)
            
    return total_stats

__all__ = ['chat_completion', 'set_debug_mode', 'list_providers', 'get_total_usage_stats']

