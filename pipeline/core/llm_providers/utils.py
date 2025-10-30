"""Shared utilities for LLM providers."""

import time
import random
from typing import Callable, TypeVar

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0
) -> T:
    """
    Retry a function with exponential backoff and jitter.
    
    Args:
        func: Function to retry (should take no args)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    
    Returns:
        Result from successful function call
    
    Raises:
        Last exception if all retries exhausted
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                print(f"Max retries ({max_retries}) exceeded. Last error: {e}")
                raise
            
            # Exponential backoff with jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            total_delay = delay + jitter
            
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            print(f"Retrying in {total_delay:.2f}s...")
            time.sleep(total_delay)

