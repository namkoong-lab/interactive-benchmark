#!/usr/bin/env python3
"""
Demo script to test the Gym-style recommendation environment.
Runs a simple random policy to verify the environment works correctly.
"""

import argparse
import random
import numpy as np
from typing import Dict, Any

try:
    from envs.reco_env import RecoEnv
    from wrappers.metrics_wrapper import MetricsWrapper, EpisodeLogger
except ImportError:
    from pipeline.envs.reco_env import RecoEnv
    from pipeline.wrappers.metrics_wrapper import MetricsWrapper, EpisodeLogger


def random_policy(env: RecoEnv, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
    """Simple random policy for testing."""
    action_map = info.get("action_map", {})
    recommend_actions = action_map.get("recommend", [])
    ask_actions = action_map.get("ask", [])
    
    # 70% chance to ask question if available, 30% to recommend
    if ask_actions and random.random() < 0.7:
        return random.choice(ask_actions)
    elif recommend_actions:
        return random.choice(recommend_actions)
    else:
        return 0


def greedy_policy(env: RecoEnv, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
    """Greedy policy: ask a few questions then recommend."""
    action_map = info.get("action_map", {})
    recommend_actions = action_map.get("recommend", [])
    ask_actions = action_map.get("ask", [])
    
    # Ask up to 3 questions, then recommend
    asked_questions = obs["asked_questions"]
    num_asked = int(np.sum(asked_questions))
    
    if num_asked < 3 and ask_actions:
        # Find unasked questions
        unasked = [i for i, asked in enumerate(asked_questions) if asked == 0]
        if unasked:
            question_idx = random.choice(unasked)
            return action_map["ask"][question_idx]
    
    # Recommend
    if recommend_actions:
        return random.choice(recommend_actions)
    else:
        return 0


def run_episode(env: RecoEnv, policy_func, render: bool = False) -> Dict[str, Any]:
    """Run a single episode with the given policy."""
    obs, info = env.reset()
    
    if render:
        env.render()
        print(f"Initial info: {info}")
    
    total_reward = 0.0
    step_count = 0
    
    while True:
        action = policy_func(env, obs, info)
        
        if render:
            action_map = info.get("action_map", {})
            if action in action_map.get("recommend", []):
                product_idx = action
                print(f"Step {step_count}: RECOMMEND product {product_idx}")
            elif action in action_map.get("ask", []):
                question_idx = action - len(action_map.get("recommend", []))
                print(f"Step {step_count}: ASK question {question_idx}")
            else:
                print(f"Step {step_count}: INVALID action {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if render:
            print(f"  Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}")
            if "action_type" in info:
                print(f"  Action type: {info['action_type']}")
        
        if terminated or truncated:
            break
    
    if render:
        print(f"Episode completed. Total reward: {total_reward:.3f}, Steps: {step_count}")
        if "chosen_product_id" in info:
            print(f"Chosen: {info['chosen_product_id']}, Best: {info.get('best_product_id', 'N/A')}")
            print(f"Regret: {info.get('regret', 0):.1f}, Top1: {info.get('top1', False)}")
    
    return {
        "total_reward": total_reward,
        "step_count": step_count,
        "final_info": info
    }


def main():
    parser = argparse.ArgumentParser(description="Demo the Gym recommendation environment")
    parser.add_argument("--persona_index", type=int, default=0, help="Persona index to use")
    parser.add_argument("--max_questions", type=int, default=8, help="Maximum questions per episode")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--policy", choices=["random", "greedy"], default="random", help="Policy to use")
    parser.add_argument("--render", action="store_true", help="Render episodes to console")
    parser.add_argument("--categories", nargs="*", help="Specific categories to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="demo_results", help="Output file prefix")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = RecoEnv(
        persona_index=args.persona_index,
        max_questions=args.max_questions,
        categories=args.categories,
        seed=args.seed
    )
    
    # Wrap with metrics logging
    env = MetricsWrapper(env)
    
    if args.render:
        env = EpisodeLogger(env)
    
    # Select policy
    if args.policy == "random":
        policy_func = random_policy
    elif args.policy == "greedy":
        policy_func = greedy_policy
    else:
        raise ValueError(f"Unknown policy: {args.policy}")
    
    print(f"Running {args.episodes} episodes with {args.policy} policy...")
    print(f"Persona index: {args.persona_index}")
    print(f"Max questions: {args.max_questions}")
    print(f"Categories: {args.categories or 'all'}")
    print()
    
    # Run episodes
    episode_results = []
    for i in range(args.episodes):
        print(f"Episode {i+1}/{args.episodes}")
        result = run_episode(env, policy_func, render=args.render)
        episode_results.append(result)
        print()
    
    # Save metrics
    env.save_metrics(f"{args.output}.jsonl", f"{args.output}.csv")
    
    # Print summary
    summary = env.get_metrics_summary()
    print("=== Summary ===")
    print(f"Total episodes: {summary.get('total_episodes', 0)}")
    print(f"Average regret: {summary.get('avg_regret', 0):.2f}")
    print(f"Top-1 accuracy: {summary.get('top1_accuracy', 0):.2%}")
    print(f"Top-3 accuracy: {summary.get('top3_accuracy', 0):.2%}")
    print(f"Average questions: {summary.get('avg_questions', 0):.1f}")
    
    # Episode-level results
    print("\n=== Episode Results ===")
    for i, result in enumerate(episode_results):
        final_info = result["final_info"]
        print(f"Episode {i+1}: reward={result['total_reward']:.3f}, "
              f"steps={result['step_count']}, "
              f"regret={final_info.get('regret', 0):.1f}, "
              f"top1={final_info.get('top1', False)}")
    
    env.close()


if __name__ == "__main__":
    main()
