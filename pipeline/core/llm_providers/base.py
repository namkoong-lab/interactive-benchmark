"""Abstract base class for all LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    To implement a new provider:
    1. Inherit from this class
    2. Implement all abstract methods
    3. Save as '*_provider.py' in this directory
    4. Auto-discovery will handle the rest
    """
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return unique provider name (e.g., 'openai', 'gemini', 'mymodel').
        Used for logging and debugging.
        """
        pass
    
    @abstractmethod
    def matches_model(self, model: str) -> bool:
        """
        Check if this provider handles the given model name.
        
        Examples:
            - return model.startswith("gpt-")
            - return model.startswith("gemini-")
            - return model in ["my-model-v1", "my-model-v2"]
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize provider (load API keys, create clients, etc.).
        Called once before first use (lazy initialization).
        
        Should raise RuntimeError if initialization fails.
        """
        pass
    
    @abstractmethod
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
        """
        Execute chat completion and return response text.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            model: Model name (e.g., "gpt-4", "gemini-pro")
            temperature: Sampling temperature (0.0 to 1.0+)
            max_tokens: Maximum tokens in response
            json_mode: Request JSON-formatted response
            response_schema: Optional JSON schema for structured output
            system_prompt_override: Override system message if needed
            count_usage: If False, do not add this call to cumulative token usage stats (e.g. product scoring).
        
        Returns:
            Response text as string
        """
        pass
    
    def chat_completion_with_tools(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Single chat completion turn with function tools. Returns assistant message as a dict
        (role, content, optional tool_calls) suitable for appending to messages and re-calling.

        Override in providers that support tool calling; default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"Provider '{self.get_provider_name()}' does not implement chat_completion_with_tools"
        )
    
    def supports_json_mode(self) -> bool:
        """Whether provider natively supports JSON mode."""
        return False
    
    def supports_response_schema(self) -> bool:
        """Whether provider supports structured response schemas."""
        return False
    
    def set_debug_mode(self, debug: bool):
        """Set debug mode (optional, can override for custom logging)."""
        pass

