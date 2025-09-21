#!/usr/bin/env python3
"""
Simple runner for the 'Random Recommendation' baseline (Baseline 2).
"""

import sys
import os
import argparse

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_random import run_baseline_random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 2 with checkpointing.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file to resume from.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible category selection (product selection remains random)")
    args = parser.parse_args()

    print("Running Baseline Experiment 2: Random Recommendation")
    
    # Run with settings that mirror the main experiment
    run_baseline_random(
        categories=None, # Will select num_categories from all available
        num_categories=10,
        episodes_per_category=1,
        feedback_type="none",         
        min_score_threshold=60.0,
        output_dir="baseline_random_results_with_checkpoints",
        checkpoint_file=args.resume_from,
        seed=args.seed if args.seed is not None else 653845                
    )
    
    print("\n'Random Recommendation' baseline experiment completed!")
    print("Check the 'baseline_random_results_with_checkpoints' directory for detailed results.")