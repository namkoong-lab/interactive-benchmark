#!/usr/bin/env python3
"""
Simple runner for the 'Prompting Tricks' baseline experiment.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.baseline_prompting_tricks import run_baseline_prompting_tricks

if __name__ == "__main__":
    print("Running Baseline Experiment: Prompting Tricks + Everything in Context")
    
    run_baseline_prompting_tricks(
        persona_index=42,
        categories=["Coffee Makers", "Clocks"],
        episodes_per_category=10,
        max_questions=3,
        model="gpt-4o",
        output_dir="baseline_prompting_tricks_results"
    )
    
    print("\n'Prompting Tricks' baseline experiment completed!")
    print("Check the 'baseline_prompting_tricks_results' directory for detailed results.")
