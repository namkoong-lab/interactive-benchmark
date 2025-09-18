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
    print("Running Experiment 1: LLM Learning Across Categories")
    print("This will test whether an LLM can learn consistent user preferences across different product categories.")
    
    results, category_results = run_experiment1(
        persona_index=254,
        categories=None,  
        num_categories=30, 
        episodes_per_category=1,
        max_questions=30,  
        model="gpt-4o",
        feedback_type="regret",
        min_score_threshold=50.0,
        output_dir="experiment1_results"
    )
    
    print("\nExperiment completed!")
