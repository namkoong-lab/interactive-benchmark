#!/usr/bin/env python3
"""
Batch runner for the Fixed Questions Experiment.

This script executes pipeline/experiment_fixed_questions.py multiple times with
different seeds, allowing for reproducible batch experiments.

It now includes functionality to aggregate detailed metrics (regret, rank,
confidence progressions, etc.) from specific tracking episodes (1, 5, 10)
across all runs into a single summary JSON file.
"""

import subprocess
import os
import json
import numpy as np
from collections import defaultdict

def run_batch_experiments():

    config = {
        "num_categories": 10,
        "model": "gpt-4o",  
        "feedback_type": "persona",
        "context_mode": "raw", 
        "episodes_per_category": 1,
        "min_score_threshold": 60.0,
    }

    # --- Specify the different seeds you want to run ---
    seeds = [
         798407, 415909
    ]

    experiment_script_module = 'experiments.planning_greedy'
    experiment_script_path = os.path.join('experiments', 'planning_greedy.py')

    if not os.path.exists(experiment_script_path):
        print(f"Error: Experiment script not found at '{experiment_script_path}'")
        return

    total_runs = len(seeds)
    output_dirs = []

    for i, seed in enumerate(seeds):
        print("=" * 60)
        print(f"STARTING RUN {i + 1}/{total_runs} WITH SEED: {seed}")
        print("=" * 60)

        output_dir = (
            f"fixed_questions_greedy_results_{config['num_categories']}_{config['model']}_"
            f"{config['feedback_type']}_{config['context_mode']}_{seed}"
        )
        output_dirs.append(output_dir)

        command = [
            'python','-m', experiment_script_module,
            '--num_categories', str(config['num_categories']),
            '--episodes_per_category', str(config['episodes_per_category']),
            '--model', config['model'],
            '--feedback_type', config['feedback_type'],
            '--min_score_threshold', str(config['min_score_threshold']),
            '--output_dir', output_dir,
            '--seed', str(seed),
            '--context_mode', config['context_mode'],
        ]

        print(f"Output Directory: {output_dir}")
        print(f"Executing command...\n")

        try:
            subprocess.run(command, check=True, text=True)
            print(f"\nSUCCESSFULLY COMPLETED RUN WITH SEED: {seed}")
        except FileNotFoundError:
            print("Error: 'python' command not found.")
            break
        except subprocess.CalledProcessError as e:
            print(f"\nERROR DURING RUN WITH SEED: {seed}")
            print(f"The experiment script exited with an error (return code {e.returncode}).")

    print("=" * 60)
    print("All experiment runs are complete!")
    print("=" * 60)

    print("\nAggregating detailed results from all successful runs...")

    # --- Configuration for result aggregation ---
    results_filename = f"fixed_questions_greedy_experiment_{config['model']}_{config['feedback_type']}.json"    
    tracked_episodes_to_analyze = {1, 5, 10}
    # Structure to hold all the data collected from all seeds
    # e.g. all_data[1]['final_regret'] will be a list of final regrets for episode 1 from all seeds
    all_data = {ep: defaultdict(list) for ep in tracked_episodes_to_analyze}
    
    successful_runs = 0
    for directory in output_dirs:
        results_file_path = os.path.join(directory, results_filename)
        if not os.path.exists(results_file_path):
            print(f"  -> WARNING: Result file not found in '{directory}'. Skipping.")
            continue
        
        try:
            with open(results_file_path, 'r') as f:
                data = json.load(f)
            
            if 'tracking_episodes_analysis' not in data:
                print(f"  -> WARNING: 'tracking_episodes_analysis' key not found in '{results_file_path}'. Skipping.")
                continue

            found_episodes_in_file = set()
            for episode_data in data['tracking_episodes_analysis']:
                ep_num = episode_data.get('episode')
                if ep_num in tracked_episodes_to_analyze:
                    found_episodes_in_file.add(ep_num)
                    
                    # --- Extract single-value metrics ---
                    all_data[ep_num]['final_regret'].append(episode_data.get('final_regret'))
                    all_data[ep_num]['final_rank'].append(episode_data.get('final_rank'))

                    # --- Extract progression metrics ---
                    confidence_prog = episode_data.get('confidence_progression', [])
                    if confidence_prog and isinstance(confidence_prog, list):
                        all_data[ep_num]['prob_favorite'].append([step.get('confidence_favorite_prob') for step in confidence_prog])
                        all_data[ep_num]['prob_top5'].append([step.get('confidence_top5_prob') for step in confidence_prog])
                        all_data[ep_num]['predicted_regret'].append([step.get('predicted_regret') for step in confidence_prog])
                        all_data[ep_num]['conf_within_5'].append([step.get('confidence_regret_within_5') for step in confidence_prog])
                        all_data[ep_num]['conf_within_10'].append([step.get('confidence_regret_within_10') for step in confidence_prog])
                        all_data[ep_num]['conf_within_20'].append([step.get('confidence_regret_within_20') for step in confidence_prog])
                        all_data[ep_num]['conf_within_30'].append([step.get('confidence_regret_within_30') for step in confidence_prog])
            
            if not found_episodes_in_file:
                 print(f"  -> WARNING: None of the tracked episodes {tracked_episodes_to_analyze} found in '{results_file_path}'.")
                 continue
            
            successful_runs += 1

        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"  -> WARNING: Could not read or process '{results_file_path}'. Error: {e}. Skipping.")
    
    if successful_runs == 0:
        print("\nNo valid data found to aggregate. Exiting.")
        return
        
    print(f"\nFound data from {successful_runs} successful runs.")

    # --- Calculate Mean and Standard Error ---
    summary_results = {}
    for ep_num, metrics_data in all_data.items():
        ep_key = f"EP{ep_num}"
        summary_results[ep_key] = {}
        
        # Helper function for calculations
        def calculate_stats(data_list):
            # Filter out None values that might have been appended
            valid_data = [x for x in data_list if x is not None]
            if not valid_data:
                return None, None
            
            arr = np.array(valid_data)
            mean = np.mean(arr, axis=0)
            # Standard Error of the Mean (SEM)
            sem = np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
            return mean.tolist(), sem.tolist()

        # Calculate for single value metrics
        regret_mean, regret_se = calculate_stats(metrics_data['final_regret'])
        rank_mean, rank_se = calculate_stats(metrics_data['final_rank'])
        summary_results[ep_key]['Regret_mean'] = regret_mean
        summary_results[ep_key]['Regret_se'] = regret_se
        summary_results[ep_key]['Actual_Rank_mean'] = rank_mean
        summary_results[ep_key]['Actual_Rank_se'] = rank_se

        # Calculate for progression metrics
        prob_fav_mean, prob_fav_se = calculate_stats(metrics_data['prob_favorite'])
        prob_top5_mean, prob_top5_se = calculate_stats(metrics_data['prob_top5'])
        pred_regret_mean, pred_regret_se = calculate_stats(metrics_data['predicted_regret'])
        conf5_mean, conf5_se = calculate_stats(metrics_data['conf_within_5'])
        conf10_mean, conf10_se = calculate_stats(metrics_data['conf_within_10'])
        conf20_mean, conf20_se = calculate_stats(metrics_data['conf_within_20'])
        conf30_mean, conf30_se = calculate_stats(metrics_data['conf_within_30'])

        summary_results[ep_key]['Probability_Favourite_progression_mean'] = prob_fav_mean
        summary_results[ep_key]['Probability_Favourite_progression_se'] = prob_fav_se
        summary_results[ep_key]['Probability_Top5_progression_mean'] = prob_top5_mean
        summary_results[ep_key]['Probability_Top5_progression_se'] = prob_top5_se
        summary_results[ep_key]['Expected_Regret_progression_mean'] = pred_regret_mean
        summary_results[ep_key]['Expected_Regret_progression_se'] = pred_regret_se
        summary_results[ep_key]['Confidence_within_5_progression_mean'] = conf5_mean
        summary_results[ep_key]['Confidence_within_5_progression_se'] = conf5_se
        summary_results[ep_key]['Confidence_within_10_progression_mean'] = conf10_mean
        summary_results[ep_key]['Confidence_within_10_progression_se'] = conf10_se
        summary_results[ep_key]['Confidence_within_20_progression_mean'] = conf20_mean
        summary_results[ep_key]['Confidence_within_20_progression_se'] = conf20_se
        summary_results[ep_key]['Confidence_within_30_progression_mean'] = conf30_mean
        summary_results[ep_key]['Confidence_within_30_progression_se'] = conf30_se

    # --- Write summary file ---
    summary_filename = (
        f"fixed_questions_greedy_results_{config['num_categories']}_{config['model']}_"
        f"{config['feedback_type']}_{config['context_mode']}_summary.json"
    )

    try:
        with open(summary_filename, 'w') as f:
            json.dump(summary_results, f, indent=4)
        print(f"\nSuccessfully aggregated detailed results and saved to '{summary_filename}'")
    except IOError as e:
        print(f"\nError writing summary file '{summary_filename}'. Error: {e}")


if __name__ == "__main__":
    run_batch_experiments()