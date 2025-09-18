#!/usr/bin/env python3
"""
Simple runner for Experiment 2.
"""

import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.experiment2 import run_experiment2

if __name__ == "__main__":
    print("Running Experiment 2: LLM Learning Questioning Strategies Across Users")
    print("This will test whether an LLM can learn optimal questioning strategies for a category across different user personas.")
    
    results, persona_results, learned_strategies = run_experiment2(
        category="Electronics",
        persona_indices=None,  # Will randomly select
        num_personas=10,
        episodes_per_persona=2,
        max_questions=30,
        model="gpt-4o",
        feedback_type="none",
        min_score_threshold=50.0,
        output_dir="experiment2_results"
    )
    
    print("\nExperiment completed!")
    print(f"Learned questioning strategies: {learned_strategies}")
