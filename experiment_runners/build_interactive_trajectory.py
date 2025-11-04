#!/usr/bin/env python3
"""
Interactive Trajectory Builder
Allows a user to manually build a "golden" trajectory, one turn at a time,
for debugging and evaluation.

Usage:
1. Start a new trajectory (Episode 1):
   python experiment_runners/build_interactive_trajectory.py \
       --config config/your_config.yaml \
       --persona_id 44348 \
       --category "Patio Coffee Tables" \
       --history_file test_runs/golden_trajectory_1.json \
       --start_new

2. Continue an existing trajectory (Episode 2, 3, ...):
   python experiment_runners/build_interactive_trajectory.py \
       --config config/your_config.yaml \
       --history_file test_runs/golden_trajectory_1.json \
       --next_category "Bass Drum Pedals"
"""

import argparse
import sys
import os
import json
import numpy as np
import random
from datetime import datetime

# Add the project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.experiment_config import ExperimentConfig
from experiments.unified_experiment import UnifiedExperiment
from pipeline.core.unified_agent import UnifiedAgent
from pipeline.core import llm_providers, simulate_interaction

def load_history(agent: UnifiedAgent, history_file_path: str) -> (list, int, int):
    """Loads history from a results.json file to "warm up" the Agent."""
    try:
        with open(history_file_path, 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        if not results:
            print(f"⚠️ Warning: History file {history_file_path} is empty. Starting a new trajectory.")
            return [], 0, 0

        # Key step: Populate the Agent's memory with historical data
        for episode_result in results:
            agent.update_episode_history(episode_result)

        # Get the Persona ID from history
        persona_id = results[0]['persona_index']
        episode_count = len(results)
        
        print(f"✅ Successfully loaded {episode_count} historical episodes.")
        print(f"   Agent memory is warmed up.")
        print(f"   Will use Persona ID: {persona_id}")
        
        return results, persona_id, episode_count

    except FileNotFoundError:
        print(f"History file '{history_file_path}' not found. A new trajectory will be created.")
        return [], 0, 0
    except Exception as e:
        print(f"❌ Error loading history file: {e}")
        sys.exit(1)

def save_history(history_file_path: str, all_results: list, config: ExperimentConfig):
    """Appends the new episode to the history file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(history_file_path), exist_ok=True)
    
    # We only save config, results, and agent_history; summary can be ignored
    output_data = {
        "config": config.to_dict_complete(),
        "results": all_results,
        "agent_history": [ep.get('dialog', []) for ep in all_results], # Simple example
        "timestamp": datetime.now().isoformat()
    }

    with open(history_file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✅ Success! Trajectory updated and saved to: {history_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Interactively build a 'golden' trajectory")
    parser.add_argument("--config", required=True, help="Path to the base config.yaml file")
    parser.add_argument("--history_file", required=True, help="Path to the results.json file to read from (and overwrite)")
    
    # Mutually exclusive arguments: you either start a new trajectory or continue an old one
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--start_new", action="store_true", help="Start a new trajectory (must also provide --persona_id and --category)")
    group.add_argument("--next_category", help="Run the next category for an existing trajectory")

    parser.add_argument("--persona_id", type=int, help="(Required with --start_new) The Persona ID for this new trajectory")
    parser.add_argument("--category", help="(Required with --start_new) The category for Episode 1")
    
    args = parser.parse_args()

    # --- 1. Argument Validation ---
    if args.start_new and (args.persona_id is None or args.category is None):
        parser.error("--start_new requires both --persona_id and --category.")
        
    if args.next_category and not os.path.exists(args.history_file):
        parser.error(f"--history_file '{args.history_file}' does not exist. Did you mean to create it with --start_new?")

    # --- 2. Load Config and Agent ---
    print("--- 1. Loading Config and Agent ---")
    config = ExperimentConfig.from_yaml(args.config)
    
    # Force debug mode so we can see the prompts
    config.debug_mode = True 
    llm_providers.set_debug_mode(True)
    simulate_interaction.set_debug_mode(True)

    # We need a temporary experiment object to borrow its _run_regular_episode method
    experiment = UnifiedExperiment(config)
    experiment.output_path = os.path.dirname(args.history_file) # Ensure wrapper can save logs
    
    # Create the Agent
    agent = experiment._create_agent()
    experiment.agent = agent # Link the agent to the experiment object

    # --- 3. Load History or Start New ---
    print("\n--- 2. Loading Trajectory History ---")
    
    all_results, persona_id, episode_count = load_history(agent, args.history_file)
    
    next_category = None
    
    if args.start_new:
        if episode_count > 0:
            if input(f"⚠️ Warning: '{args.history_file}' already contains {episode_count} episodes. Are you sure you want to overwrite it? (y/n): ").lower() != 'y':
                sys.exit("Operation cancelled.")
        # Start a new trajectory
        all_results = []
        persona_id = args.persona_id
        next_category = args.category
        episode_num = 1
        print(f"🚀 Starting new trajectory (Episode 1) @ Persona {persona_id}, Category {next_category}")

    else: # --next_category
        if episode_count == 0:
            print("❌ Error: History file is empty. You must start with --start_new.")
            sys.exit(1)
        next_category = args.next_category
        episode_num = episode_count + 1
        print(f"🚀 Continuing trajectory (Episode {episode_num}) @ Persona {persona_id}, Category {next_category}")

    # --- 4. Run the Single Episode ---
    print("\n--- 3. Running Single Episode ---")
    
    # We need to get the scores for this category
    is_relevant, max_score, scores = experiment._is_category_relevant_for_persona(
        next_category, persona_id, seed=42 # Use a fixed seed
    )
    if not is_relevant:
        print(f"❌ Error: Category '{next_category}' is not relevant for Persona {persona_id} (max_score: {max_score}).")
        sys.exit(1)

    # Run this single episode
    new_episode_result = experiment._run_regular_episode(
        episode_num=episode_num,
        persona_index=persona_id,
        category=next_category,
        cached_scores=scores,
        trajectory_seed=42, # Keep the seed stable
        trajectory_num=1 # Just a label
    )

    if not new_episode_result:
        print("❌ Error: Episode run failed. Check logs above.")
        sys.exit(1)

    # --- 5. Interactive Check ---
    print("\n--- 4. Episode Result Evaluation ---")
    print(f"   Category: {new_episode_result['category']}")
    print(f"   Regret: {new_episode_result['final_info']['regret']}")
    print(f"   Score: {new_episode_result['final_info']['chosen_score']} (Best: {new_episode_result['final_info']['best_score']})")
    print("\n   --- Dialog ---")
    for q, a in new_episode_result['full_dialog']:
        print(f"     Q: {q}")
        print(f"     A: {a}")
    print(f"\n   --- Feedback ---")
    print(f"   {new_episode_result['final_info']['feedback']}")
    print("   -----------------")

    if input("\nSave this 'good' episode to the trajectory? (y/n): ").lower() == 'y':
        all_results.append(new_episode_result)
        save_history(args.history_file, all_results, config)
        print(f"\nSuccess! To run the next episode, use --next_category [NewCategoryName] and point to the same history_file.")
    else:
        print("Discarded. History file was not modified.")
        print("You can try again with the same --next_category or try a different one.")


if __name__ == "__main__":
    main()

