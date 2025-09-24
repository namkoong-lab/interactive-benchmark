#!/usr/bin/env python3
"""
Run Dynamic Programming Planning Experiment
"""

import sys
import os
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.planning_dp import run_fixed_questions_experiment_dp

if __name__ == "__main__":
    seed = 732239
    print(f"Using seed: {seed}")
    
    results = run_fixed_questions_experiment_dp(
        categories=None,
        num_categories=10,           
        episodes_per_category=1,     
        model="claude-sonnet-4-20250514",
        feedback_type="persona",
        min_score_threshold=60.0,
        output_dir="fixed_questions_dp_results",
        seed=seed,
        context_mode="raw",
    )
    
    print("\nDynamic Programming experiment completed!")
    print("Results saved to: fixed_questions_dp_results/")

