#!/usr/bin/env python3
"""
Run Fixed Questions Experiment to measure persona elicitation effectiveness.
"""

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline.experiment_fixed_questions import run_fixed_questions_experiment

if __name__ == "__main__":
    # Use same seed as experiment1 for consistency
    seed = 522792
    print(f"Using seed: {seed}")
    
    results = run_fixed_questions_experiment(
        categories=None,  # Will randomly select relevant categories
        num_categories=10,  # Test 10 categories (10 episodes total)
        episodes_per_category=1,  # 1 episode per category (10 total episodes)
        model="claude-sonnet-4-20250514",  # Same model as experiment1
        feedback_type="persona",  # Same feedback type as experiment1
        min_score_threshold=60.0,  # Same threshold as experiment1
        output_dir="fixed_questions_results",
        seed=seed,
        context_mode="raw"  # Test with context from previous episodes
    )
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: fixed_questions_results/")
    
    # Print summary analysis
    print(f"\n=== Summary Analysis (10 questions per episode) ===")
    
    # Get tracking episodes analysis data
    tracking_episodes_analysis = results.get('tracking_episodes_analysis', [])
    
    print(f"Total episodes completed: {len(results['all_results'])}")
    print(f"Tracking episodes (1, 5, 10): {len([r for r in results['all_results'] if r.get('is_tracking_episode', False)])}")
    print(f"Normal episodes (2, 3, 4, 6, 7, 8, 9): {len([r for r in results['all_results'] if not r.get('is_tracking_episode', False)])}")
    
    # Analyze tracking episodes with regret and confidence progression
    if tracking_episodes_analysis:
        print(f"\nTracking Episodes Analysis:")
        for episode_data in tracking_episodes_analysis:
            episode_num = episode_data['episode']
            category = episode_data['category']
            regret_progression = episode_data['regret_progression']
            confidence_progression = episode_data['confidence_progression']
            final_regret = episode_data['final_regret']
            qa_pairs = episode_data['questions_and_answers']
            
            print(f"\n  Episode {episode_num} ({category}):")
            print(f"    Final regret: {final_regret:.1f}")
            
            # Show regret progression
            if regret_progression:
                print(f"    Regret progression: {[f'{r:.1f}' for r in regret_progression]}")
                if len(regret_progression) > 1:
                    improvement = regret_progression[0] - regret_progression[-1]  # Positive means improvement
                    print(f"    Regret improvement from 1st to last question: {improvement:+.1f}")
            
            # Show confidence progression
            if confidence_progression:
                print(f"    Confidence progression:")
                for i, conf_scores in enumerate(confidence_progression, 1):
                    print(f"      After question {i}:")
                    print(f"        Favorite prob: {conf_scores['confidence_favorite_prob']:.2f}")
                    print(f"        Top5 prob: {conf_scores['confidence_top5_prob']:.2f}")
                    print(f"        Expected score: {conf_scores['confidence_expected_score']:.1f}")
                    print(f"        Expected regret: {conf_scores['confidence_expected_regret']:.1f}")
                
                # Calculate confidence trends
                if len(confidence_progression) > 1:
                    first_conf = confidence_progression[0]
                    last_conf = confidence_progression[-1]
                    print(f"    Confidence trends:")
                    print(f"      Favorite prob: {first_conf['confidence_favorite_prob']:.2f} → {last_conf['confidence_favorite_prob']:.2f} ({last_conf['confidence_favorite_prob'] - first_conf['confidence_favorite_prob']:+.2f})")
                    print(f"      Top5 prob: {first_conf['confidence_top5_prob']:.2f} → {last_conf['confidence_top5_prob']:.2f} ({last_conf['confidence_top5_prob'] - first_conf['confidence_top5_prob']:+.2f})")
                    print(f"      Expected score: {first_conf['confidence_expected_score']:.1f} → {last_conf['confidence_expected_score']:.1f} ({last_conf['confidence_expected_score'] - first_conf['confidence_expected_score']:+.1f})")
                    print(f"      Expected regret: {first_conf['confidence_expected_regret']:.1f} → {last_conf['confidence_expected_regret']:.1f} ({last_conf['confidence_expected_regret'] - first_conf['confidence_expected_regret']:+.1f})")
            
            # Show questions and answers
            if qa_pairs:
                print(f"    Questions and answers ({len(qa_pairs)}):")
                for i, (question, answer) in enumerate(qa_pairs[:3], 1):  # Show first 3 Q&A pairs
                    print(f"      {i}. Q: {question}")
                    print(f"         A: {answer}")
                if len(qa_pairs) > 3:
                    print(f"      ... and {len(qa_pairs) - 3} more Q&A pairs")
    
    # Calculate overall statistics
    all_final_regrets = [r['final_info'].get('regret', 0) for r in results['all_results'] if 'regret' in r['final_info']]
    tracking_final_regrets = [r['final_info'].get('regret', 0) for r in results['all_results'] if r.get('is_tracking_episode', False) and 'regret' in r['final_info']]
    normal_final_regrets = [r['final_info'].get('regret', 0) for r in results['all_results'] if not r.get('is_tracking_episode', False) and 'regret' in r['final_info']]
    
    if all_final_regrets:
        print(f"\nOverall regret statistics:")
        print(f"  Average final regret (all episodes): {sum(all_final_regrets)/len(all_final_regrets):.2f}")
        
        if tracking_final_regrets:
            print(f"  Average final regret (tracking episodes): {sum(tracking_final_regrets)/len(tracking_final_regrets):.2f}")
        
        if normal_final_regrets:
            print(f"  Average final regret (normal episodes): {sum(normal_final_regrets)/len(normal_final_regrets):.2f}")
    
    print(f"\nCategories tested: {list(set(r['category'] for r in results['all_results']))}")
