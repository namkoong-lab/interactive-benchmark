import json
from typing import Dict, List, Tuple

from .personas import get_persona_description
from .simulate_interaction import (
    score_products_for_persona,
    simulated_user_respond,
    load_cached_scores,
    save_scores,
)


class UserModel:
    """
    Encapsulates a single simulated user with a fixed persona.

    Provides:
    - persona text access
    - answering questions as the persona
    - scoring products to form a hidden ground-truth ranking
    """

    def __init__(self, persona_index: int) -> None:
        self._persona_index = int(persona_index)
        self._persona_text: str = get_persona_description(self._persona_index)

    def get_persona_text(self) -> str:
        return self._persona_text

    def respond(self, question: str) -> str:
        return simulated_user_respond(self._persona_text, question)

    def score_products(self, category: str, products: List[Dict]) -> List[Tuple[int, float]]:
        product_ids = [int(p.get('id')) for p in products if p.get('id') is not None]
        # Try cache first
        cached_pairs = load_cached_scores(self._persona_index, category, product_ids)
        if cached_pairs and len({pid for pid, _ in cached_pairs}) == len(product_ids):
            return [(int(pid), float(score)) for pid, score in cached_pairs]

        # Compute fresh scores and persist
        scored = score_products_for_persona(self._persona_text, category, products)
        try:
            save_scores(self._persona_index, category, scored, model="ensemble")
        except Exception:
            # Don't break scoring if DB write fails
            pass
        return [(int(pid), float(score)) for pid, score, _ in scored]


