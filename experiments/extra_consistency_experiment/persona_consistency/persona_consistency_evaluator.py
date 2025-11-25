#!/usr/bin/env python3
"""
Persona Consistency Evaluator

Temporary experiment module to evaluate consistency of persona agent responses
across multiple runs of the same question using Claude Sonnet 4.

This module will be deleted after the experiment is complete.
"""

from typing import List
from datetime import datetime
from pipeline.core.llm_providers import chat_completion


class PersonaConsistencyEvaluator:
    """
    Evaluates consistency of persona agent responses across multiple runs.
    
    Takes a set of answers to the same question and evaluates how many are
    aligned vs misaligned with the persona description.
    """
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.3):
        """
        Initialize the consistency evaluator.
        
        Args:
            model: Claude model to use (default: claude-sonnet-4-20250514)
            temperature: Temperature for LLM calls (default: 0.3 for more consistent classification)
        """
        self.model = model
        self.temperature = temperature
        self.evaluation_count = 0
    
    def evaluate_consistency(
        self,
        answers: List[str]
    ) -> dict:
        """
        Evaluate consistency of multiple answers by identifying the majority position
        and counting how many deviate from it.
        
        Args:
            answers: List of answers from multiple runs (should be 10)
        
        Returns:
            Dictionary with:
            - 'majority_position': Description of what the majority of answers say
            - 'aligned_with_majority_count': Number of answers that match the majority
            - 'misaligned_with_majority_count': Number of answers that deviate from majority
            - 'total_answers': Total number of answers evaluated
            - 'consistency_rate': aligned_with_majority_count / total_answers
            - 'breakdown': Per-answer classification
            - 'evaluator_response': Full response from Sonnet 4
            - 'timestamp': ISO timestamp
        """
        self.evaluation_count += 1
        
        # Format answers for prompt
        answers_text = ""
        for i, answer in enumerate(answers, 1):
            answers_text += f"Answer {i}: {answer}\n"
        
        # Create evaluation prompt
        prompt = f"""You are evaluating the consistency of multiple statements.

**Statements (from 10 separate runs):**
{answers_text}

**Task:**
1. Analyze all 10 statements together and identify what the MAJORITY position is (what most statements are saying/expressing).
2. For each statement, determine if it aligns with the majority position or deviates from it.
3. Count how many statements are aligned with the majority vs misaligned.

**Guidelines:**
- The majority position is what most statements express (could be a preference, value, need, opinion, etc.)
- A statement is "Aligned" if it matches or is consistent with the majority position
- A statement is "Misaligned" if it contradicts, conflicts with, or significantly deviates from the majority position
- Consider both explicit statements and implied meanings
- If statements are split evenly or there's no clear majority, identify the most common position

**Output Format:**
Respond with a JSON object containing:
{{
  "majority_position": "description of what the majority of statements express",
  "aligned_with_majority_count": <number>,
  "misaligned_with_majority_count": <number>,
  "breakdown": [
    {{"answer_num": 1, "aligned_with_majority": true}},
    {{"answer_num": 2, "aligned_with_majority": false}},
    ...
  ]
}}

**Critical JSON Rules:**
- Use only valid JSON syntax
- Escape all quotes in string values with backslash: \\"
- Use true/false (lowercase, not True/False or strings)
- Keep majority_position description concise
- Do NOT include the "reason" field in breakdown (it causes parsing errors)
- Ensure all 10 answers are included in breakdown"""

        try:
            # Call Sonnet 4 - request JSON format
            response = chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of statement consistency. You must respond with valid JSON only, no other text. Ensure all strings are properly escaped."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000,  # Increased for longer breakdowns with reasons
                json_mode=False  # Claude provider may not support json_mode, parse manually
            )
            
            # Parse JSON response (may be wrapped in markdown code blocks)
            import json
            import re
            
            # Try to extract JSON from response (handle markdown code blocks)
            response_text = response.strip()
            
            # Debug: print raw response if empty
            if not response_text:
                print(f"[DEBUG] Empty response from Claude")
                raise ValueError("Empty response from Claude")
            
            # Try to extract JSON from markdown code blocks
            if response_text.startswith("```"):
                # Extract JSON from code block - use non-greedy match
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if match:
                    response_text = match.group(1)
            elif "{" in response_text:
                # Find the JSON object - match balanced braces
                # Start from first { and find matching }
                start_idx = response_text.find("{")
                if start_idx >= 0:
                    brace_count = 0
                    end_idx = start_idx
                    for i in range(start_idx, len(response_text)):
                        if response_text[i] == "{":
                            brace_count += 1
                        elif response_text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break
                    if brace_count == 0:
                        response_text = response_text[start_idx:end_idx]
            
            # Try to parse JSON
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                # If parsing fails, log for debugging
                print(f"[DEBUG] JSON parse error: {e}")
                print(f"[DEBUG] Error at position: {e.pos if hasattr(e, 'pos') else 'unknown'}")
                print(f"[DEBUG] Response length: {len(response_text)}")
                print(f"[DEBUG] Response preview (first 500 chars):\n{response_text[:500]}")
                if len(response_text) > 500:
                    print(f"[DEBUG] Response preview (last 200 chars):\n{response_text[-200:]}")
                raise ValueError(f"JSON parsing failed: {e}")
            
            majority_position = result.get('majority_position', 'Unknown')
            aligned_with_majority_count = result.get('aligned_with_majority_count', 0)
            misaligned_with_majority_count = result.get('misaligned_with_majority_count', 0)
            total_answers = len(answers)
            
            # Validate counts match
            if aligned_with_majority_count + misaligned_with_majority_count != total_answers:
                print(f"Warning: Count mismatch. Expected {total_answers}, got aligned={aligned_with_majority_count}, misaligned={misaligned_with_majority_count}")
                # Recalculate from breakdown if available
                breakdown = result.get('breakdown', [])
                if breakdown:
                    aligned_count_recalc = sum(1 for item in breakdown if item.get('aligned_with_majority', False))
                    misaligned_count_recalc = len(breakdown) - aligned_count_recalc
                    if aligned_count_recalc + misaligned_count_recalc == total_answers:
                        aligned_with_majority_count = aligned_count_recalc
                        misaligned_with_majority_count = misaligned_count_recalc
                        print(f"  Recalculated from breakdown: aligned={aligned_with_majority_count}, misaligned={misaligned_with_majority_count}")
            
            return {
                'majority_position': majority_position,
                'aligned_with_majority_count': aligned_with_majority_count,
                'misaligned_with_majority_count': misaligned_with_majority_count,
                'total_answers': total_answers,
                'consistency_rate': aligned_with_majority_count / total_answers if total_answers > 0 else 0.0,
                'breakdown': result.get('breakdown', []),
                'evaluator_response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error in consistency evaluation: {e}")
            # Print more details for debugging
            if hasattr(e, 'pos'):
                print(f"  JSON error at position: {e.pos}")
            if 'response_text' in locals():
                print(f"  Response length: {len(response_text)}")
                print(f"  Response preview: {response_text[:300]}")
            import traceback
            traceback.print_exc()
            # Return safe default on error
            return {
                'majority_position': 'Error: Could not determine',
                'aligned_with_majority_count': 0,
                'misaligned_with_majority_count': len(answers),
                'total_answers': len(answers),
                'consistency_rate': 0.0,
                'breakdown': [],
                'evaluator_response': f'Error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_stats(self) -> dict:
        """Get evaluation statistics."""
        return {
            'total_evaluations': self.evaluation_count
        }

