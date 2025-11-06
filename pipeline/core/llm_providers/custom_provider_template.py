"""
Template for integrating a custom LLM provider into AIR.

This template implements the BaseLLMProvider abstract class (defined in base.py).
All providers must implement the abstract methods to work with AIR's provider registry.

HOW TO USE THIS TEMPLATE:
1. Copy this file to `my_provider.py` (e.g., `cohere_provider.py`)
2. Replace 'Custom' with your provider name throughout
3. Implement the marked sections:
   - get_provider_name(): Return unique name (e.g., "cohere")
   - matches_model(): Define which model names you handle
   - initialize(): Load API keys and create client
   - chat_completion(): Make API calls with retry logic
4. Register in __init__.py (see instructions at bottom)
5. Add API key to .env file

REQUIRED METHODS (from BaseLLMProvider):
- get_provider_name() -> str
- matches_model(model: str) -> bool  
- initialize() -> None
- chat_completion(...) -> str

OPTIONAL METHODS (have default implementations, but RECOMMENDED):
- supports_json_mode() -> bool (return True if your API has native JSON mode)
- supports_response_schema() -> bool (return True if your API supports JSON schemas)
- set_debug_mode(debug: bool) -> None (for debug logging)

NOTE: JSON mode is used for product scoring in AIR. If your API doesn't support
native JSON mode, the system will still work but relies on prompt-based JSON requests
(less reliable). See claude_provider.py for an example without native JSON mode.

EXAMPLE PROVIDERS TO REFERENCE:
- openai_provider.py: OpenAI API integration (supports JSON mode & schemas)
- claude_provider.py: Anthropic Claude API integration
- gemini_provider.py: Google Gemini API integration (supports JSON mode)
"""

import os
from typing import List, Dict, Any, Optional
from .base import BaseLLMProvider
from .utils import retry_with_backoff


class CustomProvider(BaseLLMProvider):
    """
    Template for custom LLM provider.
    
    Replace 'Custom' with your provider name (e.g., CohereProvider, MistralProvider).
    Must implement all abstract methods from BaseLLMProvider.
    """
    
    def __init__(self):
        """
        Initialize instance (lazy initialization pattern).
        
        Don't load API keys or create clients here - do that in initialize().
        """
        self.client: Optional[Any] = None
        self._debug_mode: bool = False
    
    def get_provider_name(self) -> str:
        """Return unique provider name for logging."""
        return "custom"  # Replace with your provider name (e.g., "cohere", "mistral")
    
    def matches_model(self, model: str) -> bool:
        """
        Check if this provider handles the given model name.
        
        Examples:
            - return model.startswith("cohere-")
            - return model in ["my-model-v1", "my-model-v2"]
            - return "mistral" in model.lower()
        """
        # Replace with your model matching logic
        return model.startswith("custom-")
    
    def initialize(self) -> None:
        """
        Initialize API client (called once before first use).
        
        Typical setup:
        1. Load API key from environment
        2. Create API client
        3. Validate connection (optional)
        
        Raises:
            RuntimeError: If initialization fails
        """
        # Load API key from environment
        api_key = os.getenv("MY_CUSTOM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "MY_CUSTOM_API_KEY not found in environment. "
                "Add it to your .env file."
            )
        
        # Initialize your API client
        # Example: self.client = MyCustomClient(api_key=api_key)
        self.client = None  # Replace with actual initialization
        
        if self._debug_mode:
            print(f"[DEBUG] Initialized {self.get_provider_name()} provider")
    
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
        """
        Generate chat completion using your custom API.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [{"role": "user", "content": "Hello!"}]
            model: Model identifier (e.g., "custom-model-v1")
            temperature: Sampling temperature (0.0 to 1.0+)
            max_tokens: Maximum tokens to generate
            json_mode: Request JSON-formatted response
            response_schema: Optional JSON schema for structured output
            system_prompt_override: Override system message if needed
            
        Returns:
            Generated text response as string
            
        Raises:
            Exception: If API call fails after retries
        """
        
        # Step 1: Handle system prompt override if provided
        if system_prompt_override:
            # Replace or prepend system message
            messages = [{"role": "system", "content": system_prompt_override}] + [
                m for m in messages if m["role"] != "system"
            ]
        
        # Step 2: Prepare API request parameters
        request_params = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Step 3: Handle JSON mode if requested
        if json_mode:
            # IMPORTANT: This is used for product scoring in AIR!
            # If your API supports native JSON mode, configure it here.
            # If not, the system will rely on the prompt asking for JSON.
            
            # Examples of native JSON mode:
            # - OpenAI: request_params["response_format"] = {"type": "json_object"}
            # - Gemini: request_params["generation_config"] = {"response_mime_type": "application/json"}
            
            # If your API doesn't support native JSON mode (like Claude):
            # - Leave this empty, JSON is already requested in the system prompt
            # - Set supports_json_mode() to return False
            pass
        
        # Step 4: Handle response schema if provided
        if response_schema:
            # Some APIs support structured output with schemas
            # Example: request_params["response_format"] = {"type": "json_schema", "json_schema": response_schema}
            pass
        
        # Step 3: Make API call with automatic retry on failure
        def _make_request():
            """Wrapped function for retry logic."""
            try:
                # Replace with your actual API call
                response = self.client.generate(**request_params)
                
                # Extract text from response
                # This depends on your API's response format
                if hasattr(response, 'text'):
                    return response.text
                elif hasattr(response, 'content'):
                    return response.content
                else:
                    # Adjust based on your API's response structure
                    return str(response)
                    
            except Exception as e:
                # Log the error for debugging
                print(f"[ERROR] Custom API call failed: {e}")
                raise
        
        # Use built-in retry logic (handles rate limits, timeouts, etc.)
        return retry_with_backoff(
            _make_request,
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0
        )
    
    def supports_json_mode(self) -> bool:
        """
        Whether provider natively supports JSON mode.
        
        Return True if your API has native JSON output support.
        """
        return False  # Change to True if your API supports JSON mode
    
    def supports_response_schema(self) -> bool:
        """
        Whether provider supports structured response schemas.
        
        Return True if your API can enforce JSON schemas.
        """
        return False  # Change to True if your API supports response schemas
    
    def set_debug_mode(self, debug: bool):
        """Enable/disable debug logging."""
        self._debug_mode = debug