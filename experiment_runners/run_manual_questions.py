#!/usr/bin/env python3
"""
Run Manual Questions Experiment

This script allows you to run the manual questions experiment with specific
persona, category, and questions that YOU provide.
"""

import sys
import os
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.manual_questions import run_manual_questions_interactive

if __name__ == "__main__":
    persona_index = 25905
    category = "Office Racks & Displays"  # Change this to your desired category

    print("Running Manual Questions Experiment (Interactive Mode):")
    print(f"  Persona: {persona_index}")
    print(f"  Category: {category}")
    print("Type your questions. Enter /done when finished.\n")

    results = run_manual_questions_interactive(
        persona_index=persona_index,
        category=category,
        model="gpt-4o",
        feedback_type="persona",
        min_score_threshold=60.0,
        output_dir="manual_questions_results"
    )

    if 'error' in results:
        print(f"Experiment failed: {results['error']}")
    else:
        print("\nExperiment completed successfully!")
        episode_result = results['episode_result']
        print(f"Chosen product rank: {episode_result['chosen_rank']}")
        print(f"Final regret: {episode_result['final_info']['regret']:.1f}")
        print(f"Results saved to: manual_questions_results/")
