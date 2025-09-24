#!/usr/bin/env python3
"""
Run Variable Persona Experiment with incremental checkpointing to prevent data loss.
"""

import sys
import os
import random
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.variable_persona import run_experiment2

if __name__ == "__main__":
    random_seed = random.randint(1, 1000000)
    print(f"Using random seed: {random_seed}")
    
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
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python 'experiment runners/run_variable_persona.py' --resume_from experiment2_results_with_checkpoints/checkpoint_personas_XX_episode_XXX_gpt-4o_none.json")
