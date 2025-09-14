"""
Personas Pipeline: A framework for testing LLM-based product recommendation systems.

This package provides:
- Core simulation components (user models, LLM clients, metrics)
- Gym environments for reinforcement learning
- Experiment runners
"""

# Core components
from .core import (
    chat_completion,
    get_persona_description,
    get_products_by_category,
    list_categories,
    simulated_user_respond,
    score_products_for_persona,
    UserModel,
    EpisodeRecord,
    MetricsRecorder
)

# Environments
from .envs.reco_env import RecoEnv

# Wrappers
from .wrappers.metrics_wrapper import MetricsWrapper

__version__ = "1.0.0"

__all__ = [
    # Core
    'chat_completion',
    'get_persona_description',
    'get_products_by_category', 
    'list_categories',
    'simulated_user_respond',
    'score_products_for_persona',
    'UserModel',
    'EpisodeRecord',
    'MetricsRecorder',
    
    # Environments
    'RecoEnv',
    
    # Wrappers
    'MetricsWrapper'
]
