#!/usr/bin/env python3
"""
Simple runner for the 'Prompting Tricks' baseline, now with checkpointing.
"""

import sys
import os
import random
import argparse

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_prompting_tricks import run_baseline_prompting_tricks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 4 with checkpointing.")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint file to resume from.")
    args = parser.parse_args()

    print("Running Baseline Experiment: Prompting Tricks + Everything in Context")
    
    # Define a standard list of categories to ensure fair comparison across all experiments
    standard_categories = [
        "Bath Towel Sets", "Children's History", "Refillable Cosmetic Container Kits",
        "Hand Percussion Sound Effects", "Men's Watches", "Garage & Shop Products",
        "Women's Novelty Tanks & Camis", "Manual Foot Massagers", 
        "Men's Active & Performance Jackets", "Powersports Inner Tubes", "Today's Country"
    ]
    
    # Run with settings that mirror the main experiment
    run_baseline_prompting_tricks(
        persona_index=254,
        categories=None,
        num_categories=3, # Use all specified categories
        episodes_per_category=1,               # Increased for more stable results
        max_questions=5,                        # Agent will ask 3 questions before recommending
        model="gpt-4o",                         # The LLM to use
        feedback_type="none",                   # This agent does not use feedback
        min_score_threshold=50.0,
        output_dir="baseline_prompting_tricks_results_with_checkpoints",
        checkpoint_file=args.resume_from,       # Pass the checkpoint file path
        seed=60751         # Use a random seed for reproducibility
    )
    
    print("\n'Prompting Tricks' baseline experiment completed!")
    print("Check the 'baseline_prompting_tricks_results' directory for detailed results.")

