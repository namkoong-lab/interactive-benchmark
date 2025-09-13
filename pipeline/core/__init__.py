"""
Core modules for the personas pipeline.
"""

from .llm_client import LLMClient
from .personas import get_persona_text
from .simulate_interaction import (
    get_products_by_category, 
    list_categories,
    simulated_user_respond,
    score_products_for_persona,
    ai_recommender_interact
)
from .user_model import UserModel
from .metrics import EpisodeRecord, MetricsRecorder

__all__ = [
    'LLMClient',
    'get_persona_text', 
    'get_products_by_category',
    'list_categories',
    'simulated_user_respond',
    'score_products_for_persona',
    'ai_recommender_interact',
    'UserModel',
    'EpisodeRecord',
    'MetricsRecorder'
]
