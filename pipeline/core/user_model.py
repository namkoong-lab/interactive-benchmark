import json
from typing import Dict, List, Tuple, Any

from .personas import get_persona_description
from .simulate_interaction import (
    score_products_for_persona,
    simulated_user_respond,
    load_cached_scores,
    save_scores,
)
from .llm_providers import chat_completion


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
        self._last_feedback_prompt = None  # Store last feedback prompt for debugging

    def get_persona_text(self) -> str:
        return self._persona_text

    def respond(self, question: str, category: str, dialog_history: List[Tuple[str, str]] = None) -> str:
        return simulated_user_respond(self._persona_text, question, category, dialog_history)

    def score_products(self, category: str, products: List[Dict]) -> List[Tuple[int, float]]:
        product_ids = [int(p.get('id')) for p in products if p.get('id') is not None]
        
        # Load cached scores from database
        cached_pairs = load_cached_scores(self._persona_index, category, product_ids)
        cached_dict = {pid: score for pid, score in cached_pairs} if cached_pairs else {}
        
        # Check which products need scoring
        uncached_product_ids = [pid for pid in product_ids if pid not in cached_dict]
        
        # If all products are cached, return immediately
        if not uncached_product_ids:
            print(f"✅ Using {len(product_ids)} cached scores from database (100% cache hit)")
            return [(int(pid), float(cached_dict[pid])) for pid in product_ids]
        
        # Partial cache hit - only score uncached products
        if cached_dict:
            print(f"🔄 Partial cache hit: {len(cached_dict)}/{len(product_ids)} cached, scoring {len(uncached_product_ids)} new products")
        else:
            print(f"🔄 Computing scores for {len(uncached_product_ids)} products (no cache)")
        
        # Get only uncached products for scoring
        uncached_products = [p for p in products if int(p.get('id')) in uncached_product_ids]
        
        # Score only uncached products
        scored = score_products_for_persona(self._persona_text, category, uncached_products)
        
        # Save new scores to database
        try:
            save_scores(self._persona_index, category, scored, model="ensemble")
            print(f"💾 Saved {len(scored)} new scores to database cache")
        except Exception as e:
            print(f"⚠️ Failed to save scores to database: {e}")
            pass
        
        # Combine cached and newly scored
        result_dict = cached_dict.copy()
        for pid, score, _ in scored:
            result_dict[int(pid)] = float(score)
        
        # Return in original product order
        return [(int(pid), float(result_dict[pid])) for pid in product_ids if pid in result_dict]

    def generate_feedback(self, 
                         chosen_product: Dict[str, Any],
                         chosen_score: float,
                         regret: float,
                         category: str,
                         dialog_history: List[Tuple[str, str]] = None) -> str:
        """
        Generate tone-based feedback that reflects how close/far the recommendation was,
        without revealing regret values or specific product scores.
        
        Args:
            chosen_product: The product that was recommended
            chosen_score: Score of the chosen product
            regret: How much better the best option would have been
            category: Product category
            dialog_history: List of (question, answer) tuples from conversation
            
        Returns:
            Feedback string with tone reflecting recommendation quality
        """
        if regret == 0:
            return "Perfect! This is exactly what I was looking for. Great recommendation!"
        
        # Build conversation context with full dialog historyl
        chosen_info = f"Chosen product: {chosen_product.get('title', 'Unknown')} (Price: {chosen_product.get('price', 'Unknown')})"
        conversation_context = ""
        if dialog_history:
            conversation_lines = []
            for i, (q, a) in enumerate(dialog_history, 1):
                conversation_lines.append(f"Q{i}: {q}")
                conversation_lines.append(f"A{i}: {a}")
            conversation_context = "\n".join(conversation_lines)
        
        # Set tone based on regret level (without revealing the actual regret value)
        if regret <= 10:
            tone_instruction = "Respond positively and encouragingly - the recommendation is quite close to what you wanted, just needs minor adjustments. Sound pleased but mention one small thing that could be better based on your persona's preferences."
        elif regret <= 30:
            tone_instruction = "Respond with moderate satisfaction - the recommendation is okay but not quite right. Sound somewhat pleased but mention what you were actually looking for based on your persona's preferences."
        else:
            tone_instruction = "Respond with clear dissatisfaction - the recommendation is far from what you wanted. Sound disappointed and explain what you were actually looking for based on your persona's preferences."
        
        prompt = f"""You are role-playing as a customer who just received a product recommendation for {category} products.

**Who You Are:**
{self._persona_text}
---
**Current Situation:**
A recommendation agent just suggested this product to you: "{chosen_product.get('title', 'Unknown')}"

**Recommended Product:**
- {chosen_info}

**Full Conversation History:**
{conversation_context if conversation_context else "No previous conversation."}
---
**Your Task:**
Provide conversational feedback on this recommendation. {tone_instruction}

Base your feedback on:
1. Your general preferences and needs from your persona description
2. What you mentioned during the full conversation above
3. Whether this product seems to match what you're looking for

Keep it to 1-2 sentences. Sound natural and conversational, like talking to a salesperson. Don't mention scores or technical details.

Your response:"""

        try:
            import time
            start_time = time.time()
            
            # Store prompt for debugging
            self._last_feedback_prompt = prompt
            
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o",
                temperature=0.7,
                max_tokens=150
            )
            
            elapsed = time.time() - start_time
            print(f"[TIMING] Persona feedback generation: {elapsed:.2f}s")
            return response.strip()
        except Exception as e:
            print(f"Error generating persona feedback: {e}")
            raise e


