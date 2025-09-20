#!/usr/bin/env python3
"""
Simple runner for the 'Popularity Recommendation' baseline (Baseline 3).
"""

import sys
import os
import argparse

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_popularity import run_baseline_popularity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 3 with checkpointing.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file to resume from.")
    args = parser.parse_args()

    print("Running Baseline Experiment 3: Popularity Recommendation")
    
    # Define a standard list of categories to ensure fair comparison across all experiments
    standard_categories = [
        "Bath Towel Sets", "Children's History", "Refillable Cosmetic Container Kits",
        "Hand Percussion Sound Effects", "Men's Watches", "Garage & Shop Products",
        "Women's Novelty Tanks & Camis", "Manual Foot Massagers", 
        "Men's Active & Performance Jackets", "Powersports Inner Tubes", "Today's Country"
    ]
    
    # Run with settings that mirror the main experiment
    run_baseline_popularity(
        categories=None, # Use the standard list for consistency
        num_categories=3,
        episodes_per_category=1,
        max_questions=0,                # This agent does not ask questions
        model="popularity",             # Model name for logging
        feedback_type="none",           # This agent does not use feedback
        min_score_threshold=60.0,
        output_dir="baseline_popularity_results_with_checkpoints",
        checkpoint_file=args.resume_from,
        seed=60751                      # Use the same seed for reproducibility
    )
    
    print("\n'Popularity Recommendation' baseline experiment completed!")
    print("Check the 'baseline_popularity_results_with_checkpoints' directory for detailed results.")