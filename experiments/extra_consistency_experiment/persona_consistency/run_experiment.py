#!/usr/bin/env python3
"""
Runner script for Persona Consistency Experiment

Temporary experiment - this entire folder can be deleted after completion.

Usage:
    python -m experiments.persona_consistency.run_experiment --config config.yaml
    python -m experiments.persona_consistency.run_experiment --experiment_type variable_category --total_trajectories 5
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.experiment_config import ExperimentConfig
from experiments.persona_consistency.persona_consistency_experiment import PersonaConsistencyExperiment


def main():
    parser = argparse.ArgumentParser(
        description="Persona Consistency Experiment - Measures consistency of persona responses across multiple runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From config file
  %(prog)s --config my_config.yaml
  
  # Command line
  %(prog)s --experiment_type variable_category --total_trajectories 5 --episodes_per_trajectory 3
  
  # With custom evaluator model and number of runs
  %(prog)s --config config.yaml --evaluator_model claude-sonnet-4-20250514 --num_runs 10
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
                       help="LLM model for recommendation agent (gpt-4o, claude-3-5-sonnet, etc.)")
    parser.add_argument("--evaluator_model", default="claude-sonnet-4-20250514",
                       help="Claude model for consistency evaluation (default: claude-sonnet-4-20250514)")
    parser.add_argument("--num_runs", type=int, default=10,
                       help="Number of times to ask the same question for consistency check (default: 10)")
    
    # Seed settings
    parser.add_argument("--seeds", nargs="+", type=int, 
                       help="Optional list of seeds (first total_trajectories used)")
    
    # Experiment parameters
    parser.add_argument("--max_questions", type=int, default=8, 
                       help="Max questions per episode")
    parser.add_argument("--min_score_threshold", type=float, default=60.0, 
                       help="Min score threshold for category relevance")
    
    # Learning settings
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw",
                       help="How to carry context between episodes")
    parser.add_argument("--prompting_tricks", choices=["none", "all"],
                       help="Prompting enhancements (none or all)")
    parser.add_argument("--feedback_type", 
                       choices=["none", "regret", "persona", "star_rating"], default="none",
                       help="Type of feedback to provide")
    
    # Data settings
    parser.add_argument("--max_products", type=int, default=100, 
                       help="Max products per category (default: 100)")
    
    # Output
    parser.add_argument("--output_dir", default="experiment_results", 
                       help="Output directory for base experiment")
    parser.add_argument("--name", help="Custom experiment name")
    parser.add_argument("--debug_mode", action="store_true", 
                       help="Enable debug mode with detailed output")
    
    # Experiment-specific parameters
    parser.add_argument("--episodes_per_trajectory", type=int,
                       help="Number of episodes per trajectory")
    parser.add_argument("--total_trajectories", type=int,
                       help="Number of trajectories (agent runs)")
    
    args = parser.parse_args()
    
    # Create config from file or command-line args
    if args.config:
        config_path = args.config
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = ExperimentConfig.from_yaml(config_path)
        elif config_path.endswith('.json'):
            config = ExperimentConfig.from_json(config_path)
        else:
            raise ValueError(f"Config file must be .yaml, .yml, or .json, got: {config_path}")
    else:
        # Create config from command-line args
        if not args.experiment_type:
            parser.error("--experiment_type is required when not using --config")
        if not args.total_trajectories:
            parser.error("--total_trajectories is required when not using --config")
        if not args.episodes_per_trajectory:
            parser.error("--episodes_per_trajectory is required when not using --config")
        
        config = ExperimentConfig(
            experiment_type=args.experiment_type,
            model=args.model,
            max_questions=args.max_questions,
            context_mode=args.context_mode,
            prompting_tricks=args.prompting_tricks or "none",
            feedback_type=args.feedback_type,
            max_products_per_category=args.max_products,
            min_score_threshold=args.min_score_threshold,
            episodes_per_trajectory=args.episodes_per_trajectory,
            total_trajectories=args.total_trajectories,
            seeds=args.seeds,
            output_dir=args.output_dir,
            experiment_name=args.name,
            debug_mode=args.debug_mode
        )
    
    # Validate config
    config.validate()
    config.validate_experiment_constraints()
    
    # Run consistency experiment
    experiment = PersonaConsistencyExperiment(
        config, 
        evaluator_model=args.evaluator_model,
        num_runs=args.num_runs
    )
    experiment.run()
    
    print("\n✅ Persona Consistency Experiment completed successfully!")


if __name__ == "__main__":
    main()

