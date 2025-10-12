#!/usr/bin/env python3
"""
Runner for Oracle Recommendation Baseline
"""

import sys
import os
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.baseline_oracle import run_baseline_oracle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 4 with checkpointing.")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--persona_index", type=int, default=None, help="Persona index to use (if None, will be randomly selected based on seed)")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories to test")
    parser.add_argument("--episodes_per_category", type=int, default=1, help="Episodes per category")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories to test")
    parser.add_argument("--output_dir", type=str, default="baseline_oracle_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=732239, help="Random seed (same as other baselines for fair comparison)")
    args = parser.parse_args()

    print("Running Baseline Experiment 4: Oracle Recommendation")
    
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
