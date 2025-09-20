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
    results, persona_results = run_experiment2(
        persona_indices=None, 
        num_personas=5,
        episodes_per_persona=1,
        max_questions=20,
        model="gpt-4o", #Options: gpt-4o, gpt-5-nano-2025-08-07, gemini-2.5-pro, gemini-2.5-flash, claude-opus-4-20250514, claude-sonnet-4-20250514
        feedback_type="persona", #Options: regret, persona, star_rating
        min_score_threshold=60.0, 
        output_dir="experiment2_results_with_checkpoints",
        checkpoint_file=None,
        seed=60751,
        context_mode="raw", #Options: raw, summary 
        prompting_tricks="none" #Options: none, all
    )
    
    print(f"\nExperiment completed!")
    print(f"Total episodes: {len(results)}")
    print(f"Personas tested: {list(persona_results.keys())}")
    
    # Show how to resume from a checkpoint
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python run_experiment2_with_checkpoints.py --resume_from experiment2_results_with_checkpoints/checkpoint_personas_XX_episode_XXX_gpt-4o_none.json")
