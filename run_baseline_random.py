#!/usr/bin/env python3
"""
Simple runner for the Random Recommendation baseline experiment.
"""

import sys
import os

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_random import run_baseline_random

if __name__ == "__main__":
    # Run a small version of the experiment for testing
    print("Running Baseline Experiment: Random Recommendation")
    print("This will establish a performance lower-bound using the Experiment 1 setup.")
    
    # Run with the same settings as the main experiment for a fair comparison
    run_baseline_random(
        persona_index=42,
        categories=["Headphones", "Coffee Makers"],  # Use the same categories
        episodes_per_category=3,                  # Use the same number of episodes
        output_dir="baseline_random_results"      # Save results to a separate directory
    )
    
    print("\nRandom baseline experiment completed!")
    print("Check the 'baseline_random_results' directory for detailed results.")

