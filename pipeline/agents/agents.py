import json
from typing import Dict, List, Tuple

from ..core.simulate_interaction import ai_recommender_interact
from ..core.user_model import UserModel


class RecommenderAgent:
    """
    Wrapper around the existing interactive recommender with configurable model and question limit.
    """

    def __init__(self, model: str = "gpt-4o", max_questions: int = 8) -> None:
        self.model = model
        self.max_questions = int(max_questions)

    def recommend(self, category: str, products: List[Dict], user: UserModel) -> Tuple[int, str, int]:
        rec_id, rationale = ai_recommender_interact(
            category=category,
            products=products,
            persona_description=user.get_persona_text(),
            llm_b_model=self.model,
            max_questions=self.max_questions,
        )
        # num_questions isn't returned; approximate by counting QA pairs in rationale if present is unreliable.
        # For now, return -1 to indicate unknown; we can instrument later if needed.
        return int(rec_id), str(rationale), -1


class RandomAgent(RecommenderAgent):
    """Baseline: choose a random product without asking questions."""

    def __init__(self) -> None:
        super().__init__(model="", max_questions=0)

    def recommend(self, category: str, products: List[Dict], user: UserModel) -> Tuple[int, str, int]:
        import random

        choice = random.choice(products)
        return int(choice["id"]), "Random baseline", 0


