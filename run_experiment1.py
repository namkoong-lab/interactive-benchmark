#!/usr/bin/env python3
"""
Simple runner for Experiment 1.
"""

import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.experiment1 import run_experiment1

if __name__ == "__main__":
    # Run a small version of the experiment for testing
    print("Running Experiment 1: LLM Learning Across Categories")
    print("This will test whether an LLM can learn consistent user preferences across different product categories.")
    
    # Run with minimal settings for testing
    results, category_results = run_experiment1(
        persona_index=42,
        categories=["Headphones", "Coffee Makers"],  # Just 2 categories for testing
        episodes_per_category=3,  # 3 episodes per category
        model="gpt-4o",
        output_dir="experiment1_results"
    )
    
    print("\nExperiment completed!")
    print("Check the 'experiment1_results' directory for detailed results.")
