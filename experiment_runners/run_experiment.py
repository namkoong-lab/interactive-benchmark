#!/usr/bin/env python3
"""
Unified experiment runner - single entry point for all experiments.

Replaces:
- run_variable_category.py + run_variable_category_batch.py
- run_variable_persona.py + run_variable_persona_batch.py  
- run_variable_settings.py + run_variable_settings_batch.py

Usage examples:
    # Single trajectory
    python run_experiment.py --experiment_type variable_category --model gpt-4o --total_trajectories 1 --seeds 42
    
    # Multiple trajectories with specific seeds
    python run_experiment.py --experiment_type variable_category --model gpt-4o --total_trajectories 5 --seeds 42 43 44
    
    # Multiple trajectories with random seeds
    python run_experiment.py --experiment_type variable_category --model gpt-4o --total_trajectories 10
    
    # From config file
    python run_experiment.py --config my_config.yaml
"""

import argparse
import sys
import os
import json
import numpy as np
from datetime import datetime

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.experiment_config import ExperimentConfig
from experiments.unified_experiment import UnifiedExperiment


def aggregate_results(results_list: list, output_path: str):
    """
    Aggregate results from multiple seeds.
    
    Args:
        results_list: List of results dictionaries from different seeds
        output_path: Path to save aggregated results
    """
    print(f"\n{'='*70}")
    print(f"  AGGREGATING RESULTS FROM {len(results_list)} RUNS")
    print(f"{'='*70}")
    
    # Collect regrets from all runs
    all_regrets = []
    all_scores = []
    all_questions = []
    
    for results in results_list:
        for episode in results['results']:
            if 'regret' in episode['final_info']:
                all_regrets.append(episode['final_info']['regret'])
            if 'chosen_score' in episode['final_info']:
                all_scores.append(episode['final_info']['chosen_score'])
            all_questions.append(episode.get('steps', 0))
    
    # Calculate aggregate statistics
    aggregate = {
        'num_runs': len(results_list),
        'total_episodes': len(all_regrets),
        'regret': {
            'mean': float(np.mean(all_regrets)) if all_regrets else 0.0,
            'std': float(np.std(all_regrets)) if all_regrets else 0.0,
            'min': float(np.min(all_regrets)) if all_regrets else 0.0,
            'max': float(np.max(all_regrets)) if all_regrets else 0.0,
        },
        'score': {
            'mean': float(np.mean(all_scores)) if all_scores else 0.0,
            'std': float(np.std(all_scores)) if all_scores else 0.0,
        },
        'questions': {
            'mean': float(np.mean(all_questions)) if all_questions else 0.0,
            'total': sum(all_questions)
        },
        'individual_runs': [r['summary'] for r in results_list],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save aggregated results
    aggregate_file = os.path.join(output_path, "aggregated_results.json")
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\nAggregated Statistics:")
    print(f"  Runs: {aggregate['num_runs']}")
    print(f"  Total episodes: {aggregate['total_episodes']}")
    print(f"  Average regret: {aggregate['regret']['mean']:.2f} ± {aggregate['regret']['std']:.2f}")
    print(f"  Average score: {aggregate['score']['mean']:.2f} ± {aggregate['score']['std']:.2f}")
    print(f"  Average questions: {aggregate['questions']['mean']:.2f}")
    print(f"\nAggregated results saved to: {aggregate_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for all experiment types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single trajectory with specific seed
  %(prog)s --experiment_type variable_category --model gpt-4o --total_trajectories 1 --seeds 42
  
  # Multiple trajectories with specific seeds
  %(prog)s --experiment_type variable_category --model gpt-4o --total_trajectories 5 --seeds 42 43 44
  
  # Multiple trajectories with random seeds
  %(prog)s --experiment_type variable_category --model gpt-4o --total_trajectories 10
  
  # From config file
  %(prog)s --config my_config.yaml
        """
    )
    
    # Config file option
    parser.add_argument("--config", help="Path to YAML or JSON config file")
    
    # Experiment type
    parser.add_argument("--experiment_type", 
                       choices=["variable_category", "variable_persona", "variable_settings"],
                       help="Type of experiment to run")
    
    # Model settings
    parser.add_argument("--model", default="gpt-4o", 
                       help="LLM model (gpt-4o, claude-3-5-sonnet, gemini-2.0-flash, etc.)")
    
    # Seed settings
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional list of seeds (first total_trajectories used, pad with random if needed)")
    
    # Experiment parameters
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--min_score_threshold", type=float, default=60.0, 
                       help="Min score threshold for category relevance")
    
    # Learning settings
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw",
                       help="How to carry context between episodes")
    parser.add_argument("--prompting_tricks", choices=["none", "all"], default="none",
                       help="Prompting enhancements (none or all)")
    parser.add_argument("--feedback_type", 
                       choices=["none", "regret", "persona", "star_rating"], default="none",
                       help="Type of feedback to provide")
    
    # Data settings
    parser.add_argument("--max_products", type=int, default=100, help="Max products per category (default: 100)")
    # NOTE: categories and persona_indices must be specified via config file (list-of-lists structure)
    # Command-line only supports random generation
    
    # Output
    parser.add_argument("--output_dir", default="experiment_results", help="Output directory")
    parser.add_argument("--name", help="Custom experiment name")
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode with detailed output")
    
    # Experiment-specific parameters
    parser.add_argument("--episodes_per_trajectory", type=int,
                       help="Number of episodes per trajectory (continuous context/feedback)")
    parser.add_argument("--total_trajectories", type=int,
                       help="Number of trajectories (agent runs)")
    
    args = parser.parse_args()
    
    # Create config from file or command-line args
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config = ExperimentConfig.from_yaml(args.config) if args.config.endswith('.yaml') or args.config.endswith('.yml') else ExperimentConfig.from_json(args.config)
    else:
        if not args.experiment_type:
            parser.error("--experiment_type is required when not using --config")
        
        # Create base config with common parameters
        # NOTE: categories and persona_indices not supported via CLI (use config file for list-of-lists)
        config_kwargs = {
            'experiment_type': args.experiment_type,
            'model': args.model,
            'seeds': args.seeds,
            'max_questions': args.max_questions,
            'context_mode': args.context_mode,
            'prompting_tricks': args.prompting_tricks,
            'feedback_type': args.feedback_type,
            'min_score_threshold': args.min_score_threshold,
            'max_products_per_category': args.max_products,
            'categories': None,  # Not supported via CLI
            'persona_indices': None,  # Not supported via CLI
            'output_dir': args.output_dir,
            'experiment_name': args.name,
            'debug_mode': args.debug_mode,
        }
        
        # Add experiment-specific parameters only if provided
        if args.episodes_per_trajectory is not None:
            config_kwargs['episodes_per_trajectory'] = args.episodes_per_trajectory
        if args.total_trajectories is not None:
            config_kwargs['total_trajectories'] = args.total_trajectories
        else:
            # total_trajectories is required but has a default in ExperimentConfig
            pass
            
        config = ExperimentConfig(**config_kwargs)
        
        # Validate experiment constraints
        config.validate_experiment_constraints()
    
    # Get seeds for trajectories
    seeds = config.get_seeds()
    
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Type: {config.experiment_type}")
    print(f"Model: {config.model}")
    print(f"Total trajectories: {config.total_trajectories}")
    print(f"Seeds: {seeds if len(seeds) <= 5 else f'{seeds[:5]}... ({len(seeds)} total)'}")
    print(f"{'='*70}\n")
    
    # Run single experiment with all trajectories
    experiment = UnifiedExperiment(config)
    results = experiment.run()
    
    print(f"\n✅ Experiment complete!")


if __name__ == "__main__":
    main()

