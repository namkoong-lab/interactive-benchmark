import json
from typing import Dict, List, Tuple

from .personas import get_persona_description
from .simulate_interaction import score_products_for_persona, simulated_user_respond


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
        scores = score_products_for_persona(self._persona_text, category, products)
        return [(int(pid), float(score)) for pid, score, _ in scores]


