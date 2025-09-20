import json
from typing import Dict, List, Tuple, Any

from .personas import get_persona_description
from .simulate_interaction import (
    score_products_for_persona,
    simulated_user_respond,
    load_cached_scores,
    save_scores,
)
from .llm_client import chat_completion


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

    def generate_feedback(self, 
                         chosen_product: Dict[str, Any],
                         chosen_score: float,
                         regret: float,
                         top_products: List[Tuple[int, float]],
                         category: str,
                         dialog_history: List[Dict[str, str]] = None) -> str:
        """
        Generate tone-based feedback that reflects how close/far the recommendation was,
        without revealing regret values or the correct product.
        
        Args:
            chosen_product: The product that was recommended
            chosen_score: Score of the chosen product
            regret: How much better the best option would have been
            top_products: List of (product_id, score) tuples for top products
            category: Product category
            dialog_history: Optional conversation history
            
        Returns:
            Feedback string with tone reflecting recommendation quality
        """
        # Special case: if this is the top1 product (regret = 0), give perfect feedback
        if regret == 0:
            return "Perfect! This is exactly what I was looking for. Great recommendation!"
        
        # Determine feedback tone based on regret level
        # Small regret (0-10): Pretty close, minor adjustments needed
        # Medium regret (10-30): Somewhat off, needs improvement
        # Large regret (30+): Far off, major mismatch
        
        # Build context about the chosen product
        chosen_info = f"Chosen product: {chosen_product.get('title', 'Unknown')} (Price: {chosen_product.get('price', 'Unknown')})"
        
        # Build conversation context if available
        conversation_context = ""
        if dialog_history:
            recent_dialog = dialog_history[-2:]  # Last 2 exchanges
            conversation_context = "Our conversation included: " + "; ".join([f"Q: {q['question'][:50]}... A: {q['answer'][:50]}..." for q in recent_dialog])
        
        # Create tone instructions based on regret level
        if regret <= 10:
            tone_instruction = "Respond positively and encouragingly - the recommendation was quite close to what you wanted, just needs minor adjustments. Sound pleased but mention one small thing that could be better."
        elif regret <= 30:
            tone_instruction = "Respond with moderate satisfaction - the recommendation was okay but not quite right. Sound somewhat pleased but mention what you were actually looking for instead."
        else:
            tone_instruction = "Respond with clear dissatisfaction - the recommendation was far from what you wanted. Sound disappointed and explain what you were actually looking for."
        
        prompt = f"""You are a user with this persona:
{self._persona_text}

A recommendation agent just suggested a product to you.

Context:
- {chosen_info}
{f"- {conversation_context}" if conversation_context else ""}

{tone_instruction}

Respond naturally as this persona would - like you're talking to a helpful salesperson or friend. Be conversational and specific about your preferences. Keep it to 1-2 sentences and sound like a real person, not a formal review. Make it a statement about your preferences, not a question. Never mention specific scores, regret values, or reveal which product would be better.

Your response:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
                temperature=0.7,
                max_tokens=150
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating persona feedback: {e}")
            raise e


