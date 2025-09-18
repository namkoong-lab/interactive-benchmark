#!/usr/bin/env python3
"""
Feedback System for Recommendation Experiments.

This module provides different types of feedback mechanisms for training
LLM agents in recommendation tasks.

Feedback Types:
1. No Feedback: Agent receives no information about recommendation quality
2. Regret Feedback: Agent receives precise numerical regret value
3. Quality Feedback: Agent receives qualitative feedback based on score (80+ = great, 60-80 = ok, <60 = bad)
"""

from typing import Dict, Any, Optional


class FeedbackSystem:
    """
    Generates different types of feedback for recommendation agents.
    """
    
    def __init__(self, feedback_type: str = "none"):
        """
        Initialize feedback system.
        
        Args:
            feedback_type: Type of feedback to generate ("none", "regret", "quality")
        """
        self.feedback_type = feedback_type
        
        # Validate feedback type
        valid_types = ["none", "regret", "quality"]
        if feedback_type not in valid_types:
            raise ValueError(f"Invalid feedback_type: {feedback_type}. Must be one of {valid_types}")
    
    def generate_feedback(self, 
                         chosen_score: float, 
                         best_score: float, 
                         regret: float) -> str:
        """
        Generate feedback based on the configured feedback type.
        
        Args:
            chosen_score: Score of the chosen product
            best_score: Score of the best possible product
            regret: Regret value (best_score - chosen_score)
            
        Returns:
            Feedback string for the agent
        """
        if self.feedback_type == "none":
            return self._generate_no_feedback()
        elif self.feedback_type == "regret":
            return self._generate_regret_feedback(regret, chosen_score, best_score)
        elif self.feedback_type == "quality":
            return self._generate_quality_feedback(chosen_score)
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
