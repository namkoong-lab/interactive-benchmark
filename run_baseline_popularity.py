#!/usr/bin/env python3
"""
Simple runner for the Popularity Recommendation baseline experiment.
"""

import sys
import os

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_popularity import run_baseline_popularity

if __name__ == "__main__":
    print("Running Baseline Experiment: Popularity Recommendation")
    print("This will test a simple strategy of recommending the highest-rated or cheapest product.")
    
    run_baseline_popularity(
        persona_index=42,
        categories=["Headphones", "Coffee Makers", "Clocks"],
        episodes_per_category=30,
        output_dir="baseline_popularity_results30"
    )
    
    print("\nPopularity baseline experiment completed!")
    print("Check the 'baseline_popularity_results' directory for detailed results.")
