#!/usr/bin/env python3
"""
Feedback System for Recommendation Experiments.

This module provides different types of feedback mechanisms for training
LLM agents in recommendation tasks.

Feedback Types:
1. No Feedback: Agent receives no information about recommendation quality
2. Regret Feedback: Agent receives precise numerical regret value
3. Quality Feedback: Agent receives qualitative feedback based on score (80+ = great, 60-80 = ok, <60 = bad)
4. Persona Feedback: Agent receives contextual feedback from the persona agent about why the selection wasn't optimal
"""

from typing import Dict, Any, Optional, List, Tuple
from .llm_client import chat_completion


class FeedbackSystem:
    """
    Generates different types of feedback for recommendation agents.
    """
    
    def __init__(self, feedback_type: str = "none", persona_description: str = None):
        """
        Initialize feedback system.
        
        Args:
            feedback_type: Type of feedback to generate ("none", "regret", "quality", "persona")
            persona_description: Persona description for generating contextual feedback (required for "persona" type)
        """
        self.feedback_type = feedback_type
        self.persona_description = persona_description
        
        # Validate feedback type
        valid_types = ["none", "regret", "quality", "persona"]
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback_type: {feedback_type}. Must be one of {valid_types}")
        
        # Validate persona description for persona feedback type
        if feedback_type == "persona" and not persona_description:
            raise ValueError("persona_description is required when feedback_type is 'persona'")
    
    def generate_feedback(self, 
                         chosen_score: float, 
                         best_score: float, 
                         regret: float,
                         chosen_product: Dict[str, Any] = None,
                         available_products: List[Dict[str, Any]] = None,
                         category: str = None) -> str:
        """
        Generate feedback based on the configured feedback type.
        
        Args:
            chosen_score: Score of the chosen product
            best_score: Score of the best possible product
            regret: Regret value (best_score - chosen_score)
            chosen_product: Information about the chosen product (for persona feedback)
            available_products: List of all available products (for persona feedback)
            category: Product category (for persona feedback)
            
        Returns:
            Feedback string for the agent
        """
        if self.feedback_type == "none":
            return self._generate_no_feedback()
        elif self.feedback_type == "regret":
            return self._generate_regret_feedback(regret, chosen_score, best_score)
        elif self.feedback_type == "quality":
            return self._generate_quality_feedback(chosen_score)
        elif self.feedback_type == "persona":
            return self._generate_persona_feedback(regret, chosen_score, best_score, 
                                                 chosen_product, available_products, category)
        else:
            raise ValueError(f"Unknown feedback type: {self.feedback_type}")
    
    def _generate_no_feedback(self) -> str:
        """Generate no feedback (empty string)."""
        return ""
    
    def _generate_regret_feedback(self, regret: float, chosen_score: float, best_score: float) -> str:
        """Generate precise numerical feedback."""
        return f"Recommendation feedback: Chosen score: {chosen_score:.1f}, Best possible: {best_score:.1f}, Regret: {regret:.1f}"
    
    def _generate_quality_feedback(self, chosen_score: float) -> str:
        """Generate qualitative feedback based on score ranges."""
        if chosen_score >= 80:
            return "Great recommendation!"
        elif chosen_score >= 60:
            return "OK recommendation."
        else:
            return "Bad recommendation."
    
    def _generate_persona_feedback(self, regret: float, chosen_score: float, best_score: float,
                                 chosen_product: Dict[str, Any], available_products: List[Dict[str, Any]], 
                                 category: str) -> str:
        """Generate contextual feedback from the persona agent about why the selection wasn't optimal."""
        if not self.persona_description:
            return self._generate_regret_feedback(regret, chosen_score, best_score)
        
        # Build context about the chosen product and available options
        chosen_info = ""
        if chosen_product:
            chosen_info = f"Chosen product: {chosen_product.get('title', 'Unknown')} (Price: {chosen_product.get('price', 'Unknown')})"
        
        # Create a summary of available products without revealing which is best
        products_summary = ""
        if available_products:
            products_summary = f"Available {category} products: "
            product_titles = [p.get('title', 'Unknown')[:50] for p in available_products[:5]]  # Show first 5
            products_summary += ", ".join(product_titles)
            if len(available_products) > 5:
                products_summary += f" (and {len(available_products) - 5} more)"
        
        prompt = f"""You are a user with the following persona:
{self.persona_description}

A recommendation agent just recommended a product to you, but it wasn't the best choice for your preferences. 

Context:
- {chosen_info}
- {products_summary}
- Your satisfaction with the chosen product: {chosen_score:.1f}/100
- How much better the best option would have been: {regret:.1f} points higher

Provide a short, helpful feedback statement (1-2 sentences) explaining why this recommendation wasn't ideal for you. Focus on your specific preferences and what the agent should consider differently next time. Do NOT reveal which specific product would have been better.

Feedback:"""

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
            # Fallback to regret feedback if persona feedback fails
            return self._generate_regret_feedback(regret, chosen_score, best_score)
    
    
    def get_feedback_type(self) -> str:
        """Get the current feedback type."""
        return self.feedback_type
    
    def set_feedback_type(self, feedback_type: str):
        """Change the feedback type."""
        valid_types = ["none", "regret", "quality"]
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback_type: {feedback_type}. Must be one of {valid_types}")
        self.feedback_type = feedback_type


class FeedbackAnalyzer:
    """
    Analyzes feedback patterns and effectiveness.
    """
    
    def __init__(self):
        self.feedback_history = []
    
    def log_feedback(self, feedback: str, regret: float, feedback_type: str):
        """Log feedback for analysis."""
        self.feedback_history.append({
            'feedback': feedback,
            'regret': regret,
            'feedback_type': feedback_type,
            'length': len(feedback)
        })
    
    def analyze_feedback_effectiveness(self) -> Dict[str, Any]:
        """Analyze feedback patterns and effectiveness."""
        if not self.feedback_history:
            return {}
        
        # Group by feedback type
        by_type = {}
        for entry in self.feedback_history:
            ftype = entry['feedback_type']
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append(entry)
        
        analysis = {}
        for ftype, entries in by_type.items():
            regrets = [e['regret'] for e in entries]
            lengths = [e['length'] for e in entries]
            
            analysis[ftype] = {
                'count': len(entries),
                'avg_regret': sum(regrets) / len(regrets),
                'avg_feedback_length': sum(lengths) / len(lengths),
                'regret_range': (min(regrets), max(regrets))
            }
        
        return analysis


# Convenience functions for common feedback types
def create_no_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides no feedback."""
    return FeedbackSystem(feedback_type="none")

def create_regret_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides precise regret values."""
    return FeedbackSystem(feedback_type="regret")

def create_quality_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides qualitative feedback."""
    return FeedbackSystem(feedback_type="quality")

def create_persona_feedback_system(persona_description: str) -> FeedbackSystem:
    """Create a feedback system that provides contextual feedback from a persona agent."""
    return FeedbackSystem(feedback_type="persona", persona_description=persona_description)
