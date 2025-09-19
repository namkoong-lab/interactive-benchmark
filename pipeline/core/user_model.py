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
        Generate feedback as the persona would, given the regret value and top products.
        
        Args:
            chosen_product: The product that was recommended
            chosen_score: Score of the chosen product
            regret: How much better the best option would have been
            top_products: List of (product_id, score) tuples for top products (without revealing which is best)
            category: Product category
            dialog_history: Optional conversation history
            
        Returns:
            Feedback string from the persona's perspective
        """
        # Build context about the chosen product
        chosen_info = f"Chosen product: {chosen_product.get('title', 'Unknown')} (Price: {chosen_product.get('price', 'Unknown')})"
        
        # Create a summary of top products without revealing which is best
        top_products_summary = ""
        if top_products:
            # Get product details for top products
            from .simulate_interaction import get_products_by_category
            all_products = get_products_by_category(category)
            id_to_product = {p['id']: p for p in all_products}
            
            top_product_titles = []
            for product_id, score in top_products[:3]:  # Show top 3
                product = id_to_product.get(product_id)
                if product:
                    title = product.get('title', 'Unknown')[:50]
                    top_product_titles.append(f"{title} (Score: {score:.1f})")
            
            top_products_summary = f"Some highly-rated {category} options include: " + ", ".join(top_product_titles)
        
        # Build conversation context if available
        conversation_context = ""
        if dialog_history:
            conversation_context = "Our conversation included: " + "; ".join([f"Q: {q['question'][:50]}... A: {q['answer'][:50]}..." for q in dialog_history[-3:]])
        
        prompt = f"""You are a user with this persona:
{self._persona_text}

A recommendation agent just suggested a product to you, but it wasn't quite right for your taste.

Context:
- {chosen_info}
- How much better the best option would have been: {regret:.1f} points higher
- {top_products_summary}
{f"- {conversation_context}" if conversation_context else ""}

Respond naturally as this persona would - like you're talking to a helpful salesperson or friend who made a suggestion that missed the mark. Be conversational and specific about what you're looking for instead. Keep it to 1-2 sentences and sound like a real person, not a formal review. Make it a statement about your preferences, not a question. Don't mention specific scores or reveal which product would be better.

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
            # Fallback to simple regret-based feedback
            return f"I wasn't satisfied with this recommendation. A better option would have been {regret:.1f} points higher in my preferences."


