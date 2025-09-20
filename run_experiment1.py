#!/usr/bin/env python3
"""
Run Experiment 1 with incremental checkpointing to prevent data loss.
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.experiment1 import run_experiment1

if __name__ == "__main__":
    # Generate a random seed for reproducible randomness (affects category selection and persona selection)
    random_seed = random.randint(1, 1000000)
    print(f"Using random seed: {random_seed}")
    
    # Example usage with checkpointing
    results, category_results = run_experiment1(
        categories=None,  
        num_categories=20,  
        episodes_per_category=1, 
        max_questions=20,
        model="gpt-4o",
        feedback_type="persona",
        min_score_threshold=50.0,
        output_dir="experiment1_results_with_checkpoints",
        checkpoint_file=None,
        seed=60751
    )
    
    print(f"\nExperiment completed!")
    print(f"Total episodes: {len(results)}")
    print(f"Categories tested: {list(category_results.keys())}")
    
    # Show how to resume from a checkpoint
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python run_experiment1.py --resume_from experiment1_results_with_checkpoints/checkpoint_categories_XX_episode_XXX_gpt-4o_regret.json")
