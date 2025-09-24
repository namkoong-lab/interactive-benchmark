#!/usr/bin/env python3
"""
Simple runner for the 'Popularity Recommendation' baseline (Baseline 3).
"""

import sys
import os
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.baseline_popularity import run_baseline_popularity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 3 with checkpointing.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file to resume from.")
    args = parser.parse_args()

    print("Running Baseline Experiment 3: Popularity Recommendation")
    
    # Run with settings that mirror the main experiment
    run_baseline_popularity(
        categories=None, # Will select num_categories from all available
        num_categories=10,
        episodes_per_category=1,
        max_questions=0,                # This agent does not ask questions
        model="popularity",             # Model name for logging
        feedback_type="none",           # This agent does not use feedback
        min_score_threshold=60.0,
        output_dir="baseline_popularity_results_with_checkpoints",
        checkpoint_file=args.resume_from,
        seed=179322                      # Use the same seed for reproducibility
    )
    
    print("\n'Popularity Recommendation' baseline experiment completed!")
    print("Check the 'baseline_popularity_results_with_checkpoints' directory for detailed results.")