"""
Core modules for the personas pipeline.
"""

from .llm_client import chat_completion
from .personas import get_persona_description
from .simulate_interaction import (
    get_products_by_category, 
    list_categories,
    simulated_user_respond,
    score_products_for_persona
)
from .user_model import UserModel
from .metrics import EpisodeRecord, MetricsRecorder

__all__ = [
    'chat_completion',
    'get_persona_description', 
    'get_products_by_category',
    'list_categories',
    'simulated_user_respond',
    'score_products_for_persona',
    'UserModel',
    'EpisodeRecord',
    'MetricsRecorder'
]
