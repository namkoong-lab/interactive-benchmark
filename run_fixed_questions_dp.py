#!/usr/bin/env python3
"""
Run Dynamic Programming Fixed Questions Experiment (same K and episodes as control).
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.experiment_fixed_questions_dp import run_fixed_questions_experiment_dp

if __name__ == "__main__":
    # Use same seed policy as run_fixed_questions.py
    seed = 732239
    print(f"Using seed: {seed}")
    
    results = run_fixed_questions_experiment_dp(
        categories=None,
        num_categories=10,           # same as control
        episodes_per_category=1,     # same as control (10 total episodes)
        model="claude-sonnet-4-20250514",
        feedback_type="persona",
        min_score_threshold=60.0,
        output_dir="fixed_questions_dp_results",
        seed=seed,
        context_mode="raw",
    )
    
    print("\nDynamic Programming experiment completed!")
    print("Results saved to: fixed_questions_dp_results/")

