#!/usr/bin/env python3
"""
Feedback System for Recommendation Experiments.

This module provides different types of feedback mechanisms for training
LLM agents in recommendation tasks.

Feedback Types:
1. No Feedback: Agent receives no information about recommendation quality
2. Regret Feedback: Agent receives precise numerical regret value
3. Persona Feedback: Agent receives contextual feedback from the persona agent about why the selection wasn't optimal
4. Star Rating Feedback: Agent receives lower dimension representation like "3 out of 5 stars"
"""

from typing import Dict, Any, Optional, List, Tuple
from .llm_providers import chat_completion


class FeedbackSystem:
    """
    Generates different types of feedback for recommendation agents.
    """
    
    def __init__(self, feedback_type: str = "none", persona_agent=None):
        """
        Initialize feedback system.
        
        Args:
            feedback_type: Type of feedback to generate ("none", "regret", "persona", "star_rating")
            persona_agent: UserModel instance for generating persona feedback (required for "persona" type)
        """
        self.feedback_type = feedback_type
        self.persona_agent = persona_agent
        self.last_feedback_prompt = None  # Store last prompt for debugging
        
        valid_types = ["none", "regret", "persona", "star_rating"]
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback_type: {feedback_type}. Must be one of {valid_types}")
        
        if feedback_type == "persona" and not persona_agent:
            raise ValueError("persona_agent is required when feedback_type is 'persona'")
    
    def generate_feedback(self, 
                         chosen_score: float, 
                         best_score: float, 
                         regret: float,
                         chosen_product: Dict[str, Any] = None,
                         available_products: List[Dict[str, Any]] = None,
                         category: str = None,
                         dialog_history: List[Tuple[str, str]] = None) -> str:
        """
        Generate feedback based on the configured feedback type.
        
        Args:
            chosen_score: Score of the chosen product
            best_score: Score of the best possible product
            regret: Regret value (best_score - chosen_score)
            chosen_product: Information about the chosen product (for persona feedback)
            available_products: List of all available products (for persona feedback)
            category: Product category (for persona feedback)
            dialog_history: Conversation history between agent and persona (for persona feedback)
            
        Returns:
            Feedback string for the agent
        """
        if self.feedback_type == "none":
            return self._generate_no_feedback()
        elif self.feedback_type == "regret":
            return self._generate_regret_feedback(regret, chosen_score, best_score)
        elif self.feedback_type == "star_rating":
            return self._generate_star_rating_feedback(chosen_score, best_score)
        elif self.feedback_type == "persona":
            return self._generate_persona_feedback(regret, chosen_score, best_score, 
                                                 chosen_product, available_products, category, dialog_history)
        else:
            raise ValueError(f"Unknown feedback type: {self.feedback_type}")
    
    def _generate_no_feedback(self) -> str:
        """Generate no feedback (empty string)."""
        return ""
    
    def _generate_regret_feedback(self, regret: float, chosen_score: float, best_score: float) -> str:
        """Generate precise numerical feedback."""
        return f"Recommendation feedback: Chosen score: {chosen_score:.1f}, Best possible: {best_score:.1f}, Regret: {regret:.1f}"
    
    def _generate_star_rating_feedback(self, chosen_score: float, best_score: float) -> str:
        """Generate star rating feedback (lower dimension representation)."""
        if best_score <= 0:
            raise ValueError(f"Invalid best_score: {best_score}. Must be positive for star rating calculation.")
        
        chosen_stars = max(1, min(5, round((chosen_score / best_score) * 5)))
        best_stars = 5  
        
        return f"Recommendation rating: {chosen_stars} out of 5 stars (best possible: {best_stars} stars)"
    
    def _generate_persona_feedback(self, regret: float, chosen_score: float, best_score: float,
                                 chosen_product: Dict[str, Any], available_products: List[Dict[str, Any]], 
                                 category: str, dialog_history: List[Tuple[str, str]] = None) -> str:
        """Generate contextual feedback from the actual persona agent about why the selection wasn't optimal."""
        if not self.persona_agent:
            return self._generate_regret_feedback(regret, chosen_score, best_score)
        
        feedback = self.persona_agent.generate_feedback(
            chosen_product=chosen_product,
            chosen_score=chosen_score,
            regret=regret,
            category=category,
            dialog_history=dialog_history
        )
        
        # Capture the prompt used
        if hasattr(self.persona_agent, '_last_feedback_prompt'):
            self.last_feedback_prompt = self.persona_agent._last_feedback_prompt
        
        return feedback
    
    
    def get_feedback_type(self) -> str:
        """Get the current feedback type."""
        return self.feedback_type
    
    def set_feedback_type(self, feedback_type: str):
        """Change the feedback type."""
        valid_types = ["none", "regret", "persona", "star_rating"]
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

def create_no_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides no feedback."""
    return FeedbackSystem(feedback_type="none")

def create_regret_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides precise regret values."""
    return FeedbackSystem(feedback_type="regret")


def create_persona_feedback_system(persona_description: str) -> FeedbackSystem:
    """Create a feedback system that provides contextual feedback from a persona agent."""
    return FeedbackSystem(feedback_type="persona", persona_description=persona_description)

def create_star_rating_feedback_system() -> FeedbackSystem:
    """Create a feedback system that provides star rating feedback (lower dimension representation)."""
    return FeedbackSystem(feedback_type="star_rating")
