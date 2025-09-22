#!/usr/bin/env python3
"""
Greedy variant wrapper for the Fixed Questions experiment.

This keeps the original experiment code unchanged and exposes a
greedy-mode entrypoint that configures prompts to act greedily
(ask the best next question assuming only one question remains),
while keeping K and episode counts identical to the original.
"""

from typing import List, Dict, Any, Optional

# Reuse the core implementation and enable greedy mode
from .experiment_fixed_questions import run_fixed_questions_experiment as _run_base


def run_fixed_questions_experiment_greedy(
    categories: List[str] = None,
    num_categories: int = 10,
    episodes_per_category: int = 1,
    model: str = "gpt-4o",
    feedback_type: str = "persona",
    min_score_threshold: float = 60.0,
    output_dir: str = "fixed_questions_greedy_results",
    seed: Optional[int] = None,
    context_mode: str = "raw",
) -> Dict[str, Any]:
    """
    Run the greedy fixed-questions experiment with the same K and episode
    schedule as the standard fixed-questions setup.
    """
    return _run_base(
        categories=categories,
        num_categories=num_categories,
        episodes_per_category=episodes_per_category,
        model=model,
        feedback_type=feedback_type,
        min_score_threshold=min_score_threshold,
        output_dir=output_dir,
        seed=seed,
        context_mode=context_mode,
        greedy_mode=True,
    )


