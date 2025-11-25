#!/usr/bin/env python3
"""
Persona Alignment Evaluator

Temporary experiment module to evaluate alignment between persona agent responses
and persona descriptions using Claude Sonnet 4.

This module will be deleted after the experiment is complete.
"""

from typing import List, Tuple, Optional
from datetime import datetime
from pipeline.core.llm_providers import chat_completion


class PersonaAlignmentEvaluator:
    """
    Evaluates whether persona agent responses align with their persona descriptions.
    
    Uses Claude Sonnet 4 to classify responses as:
    - "Aligned": Response is consistent with persona description
    - "Conflict": Response contradicts persona description
    - "Neutral": Response is neither clearly aligned nor in conflict
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.3):
        """
        Initialize the alignment evaluator.
        
        Args:
            model: Claude model to use (default: claude-sonnet-4-20250514)
            temperature: Temperature for LLM calls (default: 0.3 for more consistent classification)
        """
        self.model = model
        self.temperature = temperature
        self.evaluation_count = 0
    
    def evaluate_alignment(
        self,
        persona_description: str,
        answer: str
    ) -> dict:
        """
        Evaluate alignment of a persona response with the persona description.
        
        Args:
            persona_description: The persona description text
            answer: The persona agent's response
        
        Returns:
            Dictionary with:
            - 'alignment': "Aligned", "Conflict", or "Neutral"
            - 'evaluator_response': Full response from Sonnet 4
            - 'timestamp': ISO timestamp
        """
        self.evaluation_count += 1
        
        # Create evaluation prompt
        prompt = f"""You are evaluating whether a persona agent's response aligns with their persona description.

**Persona Description:**
{persona_description}

**Persona Agent's Answer:**
{answer}

**Task:**
Classify the persona agent's answer as one of:
- "Aligned": The answer is clearly consistent with the persona description. The response directly reflects the persona's explicitly stated characteristics, preferences, lifestyle, or needs.
- "Conflict": The answer clearly contradicts or conflicts with the persona description. There must be explicit evidence of contradiction (e.g., persona says they hate X but answer expresses love for X, or persona explicitly states they don't do Y but answer assumes they do Y). Do NOT mark as Conflict just because something isn't mentioned - absence of information is not conflict.
- "Neutral": The answer is neither clearly aligned nor in conflict. 

**Output Format:**
Respond with ONLY one word: "Aligned", "Conflict", or "Neutral"
Do not include any explanation or additional text."""

        try:
            # Call Sonnet 4
            response = chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of persona consistency. Respond with only one word: Aligned, Conflict, or Neutral."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=10,  # Very short response expected
                json_mode=False
            )
            
            # Clean and normalize response
            alignment = response.strip().upper()
            
            # Map to standard values
            if "ALIGNED" in alignment:
                alignment = "Aligned"
            elif "CONFLICT" in alignment:
                alignment = "Conflict"
            elif "NEUTRAL" in alignment:
                alignment = "Neutral"
            else:
                # Fallback: try to extract from response
                if any(word in alignment for word in ["ALIGN", "CONSISTENT", "MATCH"]):
                    alignment = "Aligned"
                elif any(word in alignment for word in ["CONFLICT", "CONTRADICT", "AGAINST"]):
                    alignment = "Conflict"
                else:
                    alignment = "Neutral"  # Default to neutral if unclear
            
            return {
                'alignment': alignment,
                'evaluator_response': response.strip(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in alignment evaluation: {e}")
            # Return neutral as safe default on error
            return {
                'alignment': 'Neutral',
                'evaluator_response': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count
        }

