#!/usr/bin/env python3
"""
Run Experiment 2 with incremental checkpointing to prevent data loss.
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.experiment2 import run_experiment2

if __name__ == "__main__":
    # Generate a random seed for reproducible randomness
    random_seed = random.randint(1, 1000000)
    print(f"Using random seed: {random_seed}")
    
    # Example usage with checkpointing
    results, persona_results, learned_strategies = run_experiment2(
        category="Electronics",
        persona_indices=None,  # Will randomly select
        num_personas=10,
        episodes_per_persona=2,
        max_questions=30,
        model="gpt-4o",
        feedback_type="none",
        min_score_threshold=50.0,
        output_dir="experiment2_results_with_checkpoints",
        checkpoint_file=None,
        seed=60751,
        context_mode="raw",
        prompting_tricks="none"
    )
    
    print(f"\nExperiment completed!")
    print(f"Total episodes: {len(results)}")
    print(f"Personas tested: {list(persona_results.keys())}")
    print(f"Learned questioning strategies: {learned_strategies}")
    
    # Show how to resume from a checkpoint
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python run_experiment2_with_checkpoints.py --resume_from experiment2_results_with_checkpoints/checkpoint_personas_XX_episode_XXX_gpt-4o_none.json")
