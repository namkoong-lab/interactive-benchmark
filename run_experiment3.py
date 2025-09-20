#!/usr/bin/env python3
"""
Run Experiment 3 with incremental checkpointing to prevent data loss.
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.experiment3 import run_experiment3

if __name__ == "__main__":
    random_seed = random.randint(1, 1000000)
    print(f"Using random seed: {random_seed}")
    
    results, persona_category_results = run_experiment3(
        total_episodes=5, 
        max_questions=20,
        model="gpt-4o", #Options: gpt-4o, gpt-5-nano-2025-08-07, gemini-2.5-pro, gemini-2.5-flash, claude-opus-4-20250514, claude-sonnet-4-20250514
        feedback_type="persona", #Options: regret, persona, star_rating
        min_score_threshold=60.0,
        output_dir="experiment3_results_with_checkpoints",
        checkpoint_file=None,
        seed=179322,
        context_mode="raw", #Options: raw, summary 
        prompting_tricks="none" #Options: none, all
    )
    
    print(f"\nExperiment completed!")
    print(f"Total episodes: {len(results)}")
    print(f"Unique personas: {len(set(result['persona_index'] for result in results))}")
    print(f"Unique categories: {len(set(result['category'] for result in results))}")
    print(f"Unique persona-category combinations: {len(persona_category_results)}")
    
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python run_experiment3.py --resume_from experiment3_results_with_checkpoints/checkpoint_episode_XXX_gemini-2.5-pro_persona.json")
