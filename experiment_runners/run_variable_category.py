#!/usr/bin/env python3
"""
Run Variable Category Experimentwith incremental checkpointing to prevent data loss.
"""

import sys
import os
import random
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.variable_category import run_experiment1

if __name__ == "__main__":
    random_seed = random.randint(1, 1000000)
    print(f"Using random seed: {random_seed}")
    
    results, category_results = run_experiment1(
        categories=None,  
        num_categories=10, 
        episodes_per_category=1,
        max_questions=20,
        model="claude-sonnet-4-20250514", #Options: gpt-4o, gpt-4o-mini, gpt-5-nano-2025-08-07, gemini-2.5-pro, gemini-2.5-flash-lite, claude-opus-4-20250514, claude-sonnet-4-20250514
        feedback_type="persona", #Options: regret, persona, star_rating
        min_score_threshold=60.0,
        output_dir="experiment1_results_with_checkpoints",
        checkpoint_file=None,
        seed = 384894,
        context_mode="raw", #Options: raw, summary 
        prompting_tricks="none", #Options: none, all
        persona_index_override=20718
    )
    
    print(f"\nExperiment completed!")
    print(f"Total episodes: {len(results)}")
    print(f"Categories tested: {list(category_results.keys())}")
    
    print(f"\nTo resume from a checkpoint, use:")
    print(f"python 'experiment runners/run_variable_category.py' --resume_from experiment1_results_with_checkpoints/checkpoint_categories_XX_episode_XXX_gpt-4o_regret.json")
