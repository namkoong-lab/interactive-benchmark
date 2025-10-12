#!/usr/bin/env python3
"""
Batch runner for Experiment 1.

This script executes pipeline/experiment1.py multiple times with different seeds,
allowing for reproducible batch experiments.

It now includes functionality to aggregate the progression of regret and questions
across all runs into a single summary JSON file.
"""

import subprocess
import os
import json
import numpy as np

def run_batch_experiments():
    config = {
        "num_categories": 3,
        "model": "claude-sonnet-4-20250514",
        "episodes_per_category": 1,
        "min_score_threshold": 60
    }

    seeds = [798407, 415909]
    
    # experiment_script_path = os.path.join('pipeline', 'experiment1.py')
    experiment_script_path = os.path.join('experiments', 'baseline_oracle.py')

    if not os.path.exists(experiment_script_path):
        print(f"Error: Experiment script not found at '{experiment_script_path}'")
        print("Please ensure this script (`run_all.py`) is in your project's root directory.")
        return

    # --- Loop through each seed and run the experiment ---
    total_runs = len(seeds)
    output_dirs = []
    for i, seed in enumerate(seeds):
        print("=" * 60)
        print(f"STARTING RUN {i + 1}/{total_runs} WITH SEED: {seed}")
        print("=" * 60)

        output_dir = (
            f"baseline_oracle_exp1_results_{config['num_categories']}_{config['model']}_{seed}"
        )
        output_dirs.append(output_dir)

        command = [
            # 'python','-m', 'pipeline.experiment1',
            'python', '-m', 'experiments.baseline_oracle',
            '--num_categories', str(config['num_categories']),
            '--episodes_per_category', str(config['episodes_per_category']),
            '--model', config['model'],
            '--min_score_threshold', str(config['min_score_threshold']),
            '--output_dir', output_dir,
            '--seed', str(seed),
        ]
        
        print(f"Output Directory: {output_dir}")
        print(f"Executing command...\n")

        try:
            subprocess.run(command, check=True, text=True)
            print(f"\nSUCCESSFULLY COMPLETED RUN WITH SEED: {seed}")
        except FileNotFoundError:
            print("Error: 'python' command not found. Please ensure Python is installed and in your system's PATH.")
            break 
        except subprocess.CalledProcessError as e:
            print(f"\nERROR DURING RUN WITH SEED: {seed}")
            print(f"The experiment script exited with an error (return code {e.returncode}).")
            print("Check the output above for the specific error message from the script.")
    
    print("=" * 60)
    print("All experiment runs are complete!")

    print("\nAggregating results from all successful runs...")

    # Dynamically generate the results filename based on the config
    results_filename = f"experiment1_results_{config['model']}_{config['feedback_type']}.json"
    print(f"  -> Looking for result files named: '{results_filename}'")

    all_regrets_across_seeds = []
    all_questions_across_seeds = []
    
    successful_runs = 0
    for directory in output_dirs:
        results_file_path = os.path.join(directory, results_filename)
        if not os.path.exists(results_file_path):
            print(f"  -> WARNING: Result file not found in '{directory}'. Skipping.")
            continue
        
        try:
            with open(results_file_path, 'r') as f:
                data = json.load(f)
            
            # Extract regret and questions progressions
            regret_progression_data = data.get('summary', {}).get('regret_progression', {})
            episode_regrets = regret_progression_data.get('episode_regrets')
            episode_data = regret_progression_data.get('episode_data', [])
            
            if episode_regrets and isinstance(episode_regrets, list):
                all_regrets_across_seeds.append(episode_regrets)
                
                # Extract questions from episode_data
                questions_list = [ep.get('questions') for ep in episode_data]
                if all(q is not None for q in questions_list):
                    all_questions_across_seeds.append(questions_list)
                else:
                    print(f"  -> WARNING: Missing 'questions' data in '{results_file_path}'. Skipping questions for this run.")
                    # Add a placeholder or skip this run for questions to maintain alignment
                    all_questions_across_seeds.append([None] * len(episode_regrets))


            else:
                print(f"  -> WARNING: 'episode_regrets' key not found or not a list in '{results_file_path}'. Skipping.")
                continue
            
            successful_runs += 1

        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"  -> WARNING: Could not read or process '{results_file_path}'. Error: {e}. Skipping.")

    if successful_runs == 0:
        print("\nNo valid data found to aggregate. Exiting.")
        return

    print(f"\nFound and processed data from {successful_runs} successful runs.")

    # --- Calculate Mean and Standard Error ---
    summary_data = {}

    # --- Process Regrets ---
    if all_regrets_across_seeds:
        regrets_array = np.array(all_regrets_across_seeds)
        mean_regret = np.mean(regrets_array, axis=0)
        std_error_regret = np.std(regrets_array, axis=0, ddof=1) / np.sqrt(regrets_array.shape[0])
        summary_data['regret_progression'] = {
            "all_seed_data": regrets_array.tolist(),
            "mean": mean_regret.tolist(),
            "standard_error": std_error_regret.tolist()
        }

    # --- Process Questions ---
    # Filter out any runs that had None placeholders for questions
    valid_questions_data = [q_list for q_list in all_questions_across_seeds if all(q is not None for q in q_list)]
    if valid_questions_data:
        questions_array = np.array(valid_questions_data)
        mean_questions = np.mean(questions_array, axis=0)
        std_error_questions = np.std(questions_array, axis=0, ddof=1) / np.sqrt(questions_array.shape[0])
        summary_data['questions_progression'] = {
            "all_seed_data": questions_array.tolist(),
            "mean": mean_questions.tolist(),
            "standard_error": std_error_questions.tolist()
        }

    # --- Write summary file ---
    summary_filename = (
        f"baseline_oracle_exp1_summary_{config['num_categories']}_{config['model']}_"
        f"{config['feedback_type']}_{config['context_mode']}_"
        f"{config['prompting_tricks']}.json"
    )

    try:
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"\nSuccessfully aggregated results and saved to '{summary_filename}'")
    except IOError as e:
        print(f"\nError writing summary file '{summary_filename}'. Error: {e}")


if __name__ == "__main__":
    run_batch_experiments()