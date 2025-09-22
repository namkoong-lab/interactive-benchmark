#!/usr/bin/env python3
"""
Test persona scoring consistency by running multiple independent scoring sessions.

This script:
1. Picks a random category and persona
2. Runs scoring multiple times with fresh sessions
3. Analyzes variance and consistency
"""

import sys
import os
import random
import numpy as np
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.core.simulate_interaction import get_products_by_category, list_categories, score_products_for_persona
from pipeline.core.personas import get_persona_description


def test_single_category_consistency(persona_text: str, category: str, num_trials: int, max_products: Optional[int]):
    """Test consistency for a single category."""
    print(f"\n=== Testing Category: {category} ===")
    
    # Get products for this category
    products = get_products_by_category(category)
    if not products:
        print(f"No products found for category '{category}'")
        return None
    
    # Optionally limit products for faster testing (seed control happens outside via global RNG)
    if max_products is not None and len(products) > max_products:
        products = random.sample(products, max_products)
    
    print(f"Products to score: {len(products)}")
    
    # Run multiple scoring trials
    all_trials = []
    
    for trial in range(num_trials):
        print(f"Trial {trial + 1}/{num_trials}...", end=" ", flush=True)
        
        try:
            # Score products with fresh session
            scores = score_products_for_persona(persona_text, category, products)
            
            # Convert to dict for easier analysis
            score_dict = {pid: score for pid, score, _ in scores}
            all_trials.append(score_dict)
            
            print(f"✓ ({len(score_dict)} products, range: {min(score_dict.values()):.1f}-{max(score_dict.values()):.1f})")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    if not all_trials:
        print(f"No successful trials for {category}")
        return None
    
    # Get all product IDs that were scored in all trials
    common_products = set(all_trials[0].keys())
    for trial in all_trials[1:]:
        common_products &= set(trial.keys())
    
    if not common_products:
        print(f"No products scored consistently across trials for {category}")
        return None
    
    # Calculate statistics for each product
    product_stats = {}
    for pid in common_products:
        scores_for_product = [trial[pid] for trial in all_trials]
        product_stats[pid] = {
            'scores': scores_for_product,
            'mean': np.mean(scores_for_product),
            'std': np.std(scores_for_product),
            'min': min(scores_for_product),
            'max': max(scores_for_product),
            'range': max(scores_for_product) - min(scores_for_product)
        }
    
    # Calculate ranking consistency
    trial_rankings = []
    for trial in all_trials:
        ranking = sorted(common_products, key=lambda pid: trial[pid], reverse=True)
        trial_rankings.append(ranking)
    
    rank_stats = {}
    for pid in common_products:
        ranks = [ranking.index(pid) for ranking in trial_rankings]
        rank_stats[pid] = {
            'mean_rank': np.mean(ranks),
            'rank_std': np.std(ranks),
            'rank_range': max(ranks) - min(ranks)
        }
    
    # Overall metrics for this category
    all_stds = [stats['std'] for stats in product_stats.values()]
    all_ranges = [stats['range'] for stats in product_stats.values()]
    all_rank_stds = [stats['rank_std'] for stats in rank_stats.values()]
    
    category_results = {
        'category': category,
        'num_trials': len(all_trials),
        'num_products': len(common_products),
        'product_stats': product_stats,
        'rank_stats': rank_stats,
        'metrics': {
            'avg_std': float(np.mean(all_stds)),
            'max_std': float(np.max(all_stds)),
            'avg_range': float(np.mean(all_ranges)),
            'max_range': float(np.max(all_ranges)),
            'avg_rank_std': float(np.mean(all_rank_stds)),
            'max_rank_std': float(np.max(all_rank_stds))
        }
    }
    
    return category_results


def test_scoring_consistency(persona_index: int = 105, 
                           num_categories: int = 10,
                           num_trials: int = 5,
                           max_products: Optional[int] = None,
                           seed: Optional[int] = 42):
    """
    Large-scale test of scoring consistency across multiple categories.
    
    Args:
        persona_index: Which persona to use
        num_categories: Number of categories to test
        num_trials: Number of independent scoring runs per category
        max_products: Max products to score per trial
    """
    
    print(f"=== Large-Scale Persona Scoring Consistency Test ===")
    print(f"Persona: {persona_index}")
    print(f"Categories: {num_categories}")
    print(f"Trials per category: {num_trials}")
    print(f"Max products per trial: {max_products}")
    if seed is not None:
        print(f"Seed: {seed}")

    # Seed like experiment1 (controls category ordering and sampling)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Get persona description
    try:
        persona_text = get_persona_description(persona_index)
        print(f"Persona preview: {persona_text[:150]}...")
    except Exception as e:
        print(f"Error loading persona {persona_index}: {e}")
        return
    
    # Get available categories
    available_categories = list_categories()
    if len(available_categories) < num_categories:
        print(f"Warning: Only {len(available_categories)} categories available, testing all of them")
        categories_to_test = available_categories
    else:
        # Randomized order based on seed, then take first N categories
        categories_shuffled = available_categories.copy()
        random.shuffle(categories_shuffled)
        categories_to_test = categories_shuffled[:num_categories]
    
    print(f"Categories to test: {categories_to_test}")
    
    # Test each category
    all_category_results = []
    
    for i, category in enumerate(categories_to_test):
        print(f"\n[{i+1}/{len(categories_to_test)}] Testing {category}...")
        result = test_single_category_consistency(persona_text, category, num_trials, max_products)
        if result:
            all_category_results.append(result)
    
    if not all_category_results:
        print("No successful category tests!")
        return
    
    # Aggregate statistics across all categories
    print(f"\n=== Overall Summary Across {len(all_category_results)} Categories ===")
    
    # Collect all metrics
    all_avg_stds = [r['metrics']['avg_std'] for r in all_category_results]
    all_max_stds = [r['metrics']['max_std'] for r in all_category_results]
    all_avg_ranges = [r['metrics']['avg_range'] for r in all_category_results]
    all_max_ranges = [r['metrics']['max_range'] for r in all_category_results]
    all_avg_rank_stds = [r['metrics']['avg_rank_std'] for r in all_category_results]
    all_max_rank_stds = [r['metrics']['max_rank_std'] for r in all_category_results]
    
    # Overall statistics
    print(f"Score Consistency:")
    print(f"  Average std deviation: {np.mean(all_avg_stds):.2f} ± {np.std(all_avg_stds):.2f}")
    print(f"  Max std deviation: {np.mean(all_max_stds):.2f} ± {np.std(all_max_stds):.2f}")
    print(f"  Average score range: {np.mean(all_avg_ranges):.2f} ± {np.std(all_avg_ranges):.2f}")
    print(f"  Max score range: {np.mean(all_max_ranges):.2f} ± {np.std(all_max_ranges):.2f}")
    
    print(f"Ranking Consistency:")
    print(f"  Average rank std: {np.mean(all_avg_rank_stds):.2f} ± {np.std(all_avg_rank_stds):.2f}")
    print(f"  Max rank std: {np.mean(all_max_rank_stds):.2f} ± {np.std(all_max_rank_stds):.2f}")
    
    # Category breakdown
    print(f"\nCategory Breakdown:")
    print(f"{'Category':<20} {'Products':<8} {'Avg Std':<8} {'Max Std':<8} {'Avg Range':<10} {'Rank Std':<8}")
    print("-" * 70)
    
    for result in sorted(all_category_results, key=lambda x: x['metrics']['avg_std']):
        print(f"{result['category']:<20} {result['num_products']:<8} "
              f"{result['metrics']['avg_std']:<8.2f} {result['metrics']['max_std']:<8.2f} "
              f"{result['metrics']['avg_range']:<10.2f} {result['metrics']['avg_rank_std']:<8.2f}")
    
    # Consistency assessment
    print(f"\n=== Consistency Assessment ===")
    
    # Define thresholds for "good" consistency
    good_std_threshold = 5.0  # Standard deviation < 5 points
    good_range_threshold = 10.0  # Range < 10 points
    good_rank_std_threshold = 2.0  # Ranking std < 2 positions
    
    categories_good_std = sum(1 for r in all_category_results if r['metrics']['avg_std'] < good_std_threshold)
    categories_good_range = sum(1 for r in all_category_results if r['metrics']['avg_range'] < good_range_threshold)
    categories_good_rank = sum(1 for r in all_category_results if r['metrics']['avg_rank_std'] < good_rank_std_threshold)
    
    print(f"Categories with good score consistency (std < {good_std_threshold}): {categories_good_std}/{len(all_category_results)} ({categories_good_std/len(all_category_results)*100:.1f}%)")
    print(f"Categories with good range consistency (range < {good_range_threshold}): {categories_good_range}/{len(all_category_results)} ({categories_good_range/len(all_category_results)*100:.1f}%)")
    print(f"Categories with good ranking consistency (rank std < {good_rank_std_threshold}): {categories_good_rank}/{len(all_category_results)} ({categories_good_rank/len(all_category_results)*100:.1f}%)")
    
    # Overall verdict
    overall_consistency = (categories_good_std + categories_good_range + categories_good_rank) / (3 * len(all_category_results))
    print(f"\nOverall Consistency Score: {overall_consistency*100:.1f}%")
    
    if overall_consistency > 0.8:
        print("✓ EXCELLENT: Scoring is highly consistent across categories")
    elif overall_consistency > 0.6:
        print("✓ GOOD: Scoring is reasonably consistent across categories")
    elif overall_consistency > 0.4:
        print("⚠ MODERATE: Scoring shows some inconsistency")
    else:
        print("✗ POOR: Scoring is inconsistent across categories")
    
    # Save comprehensive results
    comprehensive_results = {
        'persona_index': persona_index,
        'num_categories_tested': len(all_category_results),
        'num_trials_per_category': num_trials,
        'max_products_per_trial': max_products,
        'overall_metrics': {
            'avg_std_mean': float(np.mean(all_avg_stds)),
            'avg_std_std': float(np.std(all_avg_stds)),
            'max_std_mean': float(np.mean(all_max_stds)),
            'max_std_std': float(np.std(all_max_stds)),
            'avg_range_mean': float(np.mean(all_avg_ranges)),
            'avg_range_std': float(np.std(all_avg_ranges)),
            'avg_rank_std_mean': float(np.mean(all_avg_rank_stds)),
            'avg_rank_std_std': float(np.std(all_avg_rank_stds)),
            'overall_consistency_score': float(overall_consistency)
        },
        'category_results': all_category_results,
        'timestamp': datetime.now().isoformat()
    }
    
    output_file = f"large_scale_consistency_results_{persona_index}_{len(all_category_results)}cats_seed{seed}.json"
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nComprehensive results saved to: {output_file}")
    
    return comprehensive_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Large-scale test of persona scoring consistency")
    parser.add_argument("--persona_index", type=int, default=42, help="Persona index to test")
    parser.add_argument("--num_categories", type=int, default=10, help="Number of categories to test (first N after seeding)")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of independent scoring trials per category")
    parser.add_argument("--max_products", type=int, default=None, help="Max products to score per trial (omit or None for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same behavior as experiment1)")
    
    args = parser.parse_args()
    
    test_scoring_consistency(
        persona_index=args.persona_index,
        num_categories=args.num_categories,
        num_trials=args.num_trials,
        max_products=args.max_products,
        seed=args.seed
    )
