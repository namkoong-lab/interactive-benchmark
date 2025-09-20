#!/usr/bin/env python3
"""
Simple runner for the 'Oracle Recommendation' baseline (Baseline 4).
"""

import sys
import os
import argparse

# Add the 'pipeline' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

# Import the specific runner function for this baseline
from pipeline.baseline_oracle import run_baseline_oracle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 4 with checkpointing.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--persona_index", type=int, default=None, help="Persona index to use (if None, will be randomly selected based on seed)")
    parser.add_argument("--num_categories", type=int, default=20, help="Number of categories to test")
    parser.add_argument("--episodes_per_category", type=int, default=1, help="Episodes per category")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories to test")
    parser.add_argument("--output_dir", type=str, default="baseline_oracle_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=798407, help="Random seed (same as other baselines for fair comparison)")
    args = parser.parse_args()

    print("Running Baseline Experiment 4: Oracle Recommendation")
    
    # Run with settings that mirror the main experiment
    run_baseline_oracle(
        categories=args.categories, 
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        model=args.model,
        persona_index=args.persona_index,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\n'Oracle Recommendation' baseline experiment completed!")
    print(f"Check the '{args.output_dir}' directory for detailed results.")
