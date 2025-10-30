"""Provider registry for managing LLM providers."""

from typing import Dict, List
from .base import BaseLLMProvider


class ProviderRegistry:
    """Central registry for all LLM providers."""
    
    def __init__(self):
        self._providers: List[BaseLLMProvider] = []
        self._initialized: Dict[str, bool] = {}
        self._debug_mode: bool = False
    
    def register(self, provider: BaseLLMProvider):
        """Register a new provider."""
        self._providers.append(provider)
    
    def get_provider(self, model: str) -> BaseLLMProvider:
        """
        Find provider that handles this model.
        Uses lazy initialization.
        """
        for provider in self._providers:
            if provider.matches_model(model):
                # Lazy initialization on first use
                provider_name = provider.get_provider_name()
                if not self._initialized.get(provider_name):
                    provider.initialize()
                    self._initialized[provider_name] = True
                    if self._debug_mode:
                        print(f"[DEBUG] Initialized provider: {provider_name}")
                return provider
        
        # No provider found
        available = [p.get_provider_name() for p in self._providers]
        raise ValueError(
            f"No provider found for model '{model}'. "
            f"Available providers: {', '.join(available)}"
        )
    
    def chat_completion(self, messages, model, **kwargs) -> str:
        """Central dispatch to appropriate provider."""
        provider = self.get_provider(model)
        return provider.chat_completion(messages, model, **kwargs)
    
    def set_debug_mode(self, debug: bool):
        """Set debug mode for all providers."""
        self._debug_mode = debug
        for provider in self._providers:
            provider.set_debug_mode(debug)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return [p.get_provider_name() for p in self._providers]

