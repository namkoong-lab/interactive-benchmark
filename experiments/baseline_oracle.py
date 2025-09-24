#!/usr/bin/env python3
"""
Oracle Baseline: Direct product recommendation with full persona context.
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
from pipeline.envs.reco_env import RecoEnv
from pipeline.core.llm_client import chat_completion
from pipeline.wrappers.metrics_wrapper import MetricsWrapper
from pipeline.core.personas import get_persona_description


class OracleAgent:
    """
    Oracle-based agent that receives the full persona description and directly predicts the best product.
    """
    
    def __init__(self, model: str = "gpt-4o", persona_index: int = 0):
        self.model = model
        self.persona_index = persona_index
        self.persona_description = get_persona_description(persona_index)
        self.episode_count = 0
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Directly predict the best product using full persona context."""
        if 'num_products' in info:
            num_products = info['num_products']
            category = info['category']
        else:
            num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
            category = "unknown"
        
        return self._oracle_recommendation(obs, info, category, num_products)
    
    def _oracle_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                              category: str, num_products: int) -> int:
        """Make direct recommendation using full persona context."""
        products = self._get_product_info(obs, info, num_products)
        
        base_prompt = f"""You are an oracle recommendation agent with perfect knowledge of a customer's preferences.

Customer Persona:
{self.persona_description}

Product Category: {category}

Available Products:
{self._format_products(products)}

Task:
Given the customer's complete persona description, choose the single best product that would most satisfy their preferences and needs. You have perfect knowledge of what this customer would want.

Output format (MUST be exactly one line, no extra text):
RECOMMEND: <array_index_0_to_{num_products-1}>

IMPORTANT: Return the PRODUCT INDEX (0, 1, 2, etc.), NOT the product ID number.

Rules:
- Choose the product that best matches the customer's persona
- Consider all aspects of their preferences, lifestyle, and needs
- Return the array index (0-based), not the product ID
- No explanations, just the recommendation index
- You must recommend exactly one product"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.1,  
                max_tokens=100
            )
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.strip().startswith('RECOMMEND:'):
                    try:
                        rec_num = int(line.split(':')[1].strip())
                        if 0 <= rec_num < num_products:
                            return rec_num
                    except (ValueError, IndexError):
                        pass
            
            print(f"Warning: Failed to parse oracle recommendation: {response}")
            return random.randint(0, num_products - 1)
            
        except Exception as e:
            print(f"Error in oracle recommendation: {e}")
            return random.randint(0, num_products - 1)
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict[str, Any]]:
        """Extract product information from observation."""
        products = []
        product_descriptions = info.get('product_descriptions', [])
        full_products = info.get('full_products', [])
        
        for i in range(num_products):
            product_features = obs['product_features'][i]
            if i < len(full_products):
                full_product = full_products[i]
                raw_data = full_product.get("raw", {})
                
                description = ""
                if "description" in raw_data and raw_data["description"]:
                    if isinstance(raw_data["description"], list) and raw_data["description"]:
                        description = raw_data["description"][0]
                    elif isinstance(raw_data["description"], str):
                        description = raw_data["description"]
                
                if not description:
                    description = full_product.get("title", "No description available")
                
                if len(description) > 500:
                    description = description[:500] + "..."
                
                product = {
                    "id": full_product.get("id", i),
                    "title": full_product.get("title", f"Product {i}"),
                    "price": full_product.get("price", float(product_features[0]) if len(product_features) > 0 and product_features[0] is not None else 0.0),
                    "store": full_product.get("store", f"Store {int(product_features[1])}" if len(product_features) > 1 and product_features[1] is not None else "Unknown"),
                    "description": description,
                    "raw_attributes": {
                        k: v for k, v in raw_data.items()
                        if isinstance(v, (str, int, float)) and k not in {"description", "title"}
                    }
                }
            else:
                product = {
                    "id": info.get('product_ids', [])[i] if i < len(info.get('product_ids', [])) else i,
                    "title": f"Product {i}",
                    "price": float(product_features[0]) if len(product_features) > 0 and product_features[0] is not None else 0.0,
                    "store": f"Store {int(product_features[1])}" if len(product_features) > 1 and product_features[1] is not None else "Unknown",
                    "description": product_descriptions[i] if i < len(product_descriptions) else "No description available",
                    "raw_attributes": {}
                }
            
            products.append(product)
        
        return products
    
    def _format_products(self, products: List[Dict[str, Any]]) -> str:
        """Format product information for the LLM prompt."""
        formatted = []
        for i, product in enumerate(products):
            description = product.get('description', 'No description available')
            if len(description) > 300:
                description = description[:300] + "..."
            
            formatted.append(f"Product {i}:")
            formatted.append(f"  ID: {product['id']}")
            formatted.append(f"  Title: {product.get('title', 'No title')}")
            price = product.get('price', 0.0)
            price_str = f"${price:.2f}" if isinstance(price, (int, float)) else str(price)
            formatted.append(f"  Price: {price_str}")
            formatted.append(f"  Store: {product['store']}")
            formatted.append(f"  Description: {description}")
            
            raw_attributes = product.get('raw_attributes', {})
            if raw_attributes:
                formatted.append(f"  Key Attributes:")
                important_attrs = ['brand', 'material', 'color', 'style', 'size', 'rating', 'average_rating']
                for attr in important_attrs:
                    if attr in raw_attributes:
                        value = raw_attributes[attr]
                        if isinstance(value, (str, int, float)):
                            formatted.append(f"    {attr.title()}: {value}")
            
            formatted.append("")  
        
        return "\n".join(formatted)


def train_oracle_agent_multi_category(categories: List[str],
                                    cached_scores_map: Dict[str, List[Tuple[int, float]]],
                                    agent: OracleAgent,
                                    persona_index: int,
                                    episodes_per_category: int = 10,
                                    output_dir: str = "oracle_results",
                                    seed: Optional[int] = None,
                                    min_score_threshold: float = 60.0,
                                    num_categories: int = 10) -> Dict[str, Any]:
    """
    Train the oracle agent across multiple categories (same pattern as other baselines).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    category_performance = {}
    all_episodes = []
    episode_num = 0
    successful_episodes_count = 0
    
    target_total_episodes = num_categories * episodes_per_category
    total_episodes = episodes_per_category * len(categories)
    print(f"Running up to {total_episodes} oracle episodes across {len(categories)} categories")
    print(f"Target: {target_total_episodes} total episodes")
    
    def is_category_relevant_for_persona(category: str, persona_index: int, min_score_threshold: float):
        """Check if category is relevant for persona (same as experiment1)."""
        try:
            from pipeline.core.simulate_interaction import get_products_by_category
            from pipeline.core.user_model import UserModel
            
            products = get_products_by_category(category)
            if not products:
                return False, 0.0, []
            
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            
            max_score = max(score for _, score in scores) if scores else 0.0
            is_relevant = max_score > min_score_threshold
            
            return is_relevant, max_score, scores
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []
    
    for category in categories:
        if episode_num >= target_total_episodes:
            break
            
        print(f"\n--- Testing Category: {category} ---")
        
        is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
        
        if not is_relevant:
            print(f"  ✗ Category {category}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping")
            continue
        
        print(f"  ✓ Category {category}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        
        env = RecoEnv(
            persona_index=persona_index,
            max_questions=0,  
            categories=[category],
            seed=seed,
            agent=agent,
            feedback_system=None,  
            cached_scores=cached_scores
        )
        
        for episode in range(episodes_per_category):
            if episode_num >= target_total_episodes:
                break
                
            episode_num += 1
            print(f"Episode {episode_num} (Category: {category}) - {episode_num}/{target_total_episodes} total episodes")
            
            try:
                from pipeline.wrappers.metrics_wrapper import MetricsWrapper
                metrics_wrapper = MetricsWrapper(env, output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
                
                obs, initial_info = metrics_wrapper.reset()
                
                agent.current_env = env
                
                if episode == 0:
                    print(f"  Products in {category}: {len(env.products)}")
                    print(f"  Top 3 products by score:")
                    for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
                        product = next((p for p in env.products if p['id'] == product_id), None)
                        if product:
                            title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                            print(f"    {i+1}. {title} (Score: {score:.1f})")
                
                action = agent.get_action(obs, initial_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                
                print(f"  Recommended product {info['chosen_product_id']}")
                regret_value = info.get('regret', 0.0)
                regret_str = f"{regret_value:.1f}" if isinstance(regret_value, (int, float)) else str(regret_value)
                print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}, Regret: {regret_str}")
                
                episode_result = {
                    'episode': episode_num, 'category': category, 'episode_in_category': episode + 1,
                    'steps': 1, 'terminated': True, 'truncated': False, 
                    'final_info': info, 'full_dialog': [], 'product_info': {}, 'thinking_block': None
                }
                
                episode_info = info  
                
                if category not in category_performance:
                    category_performance[category] = {
                        'episodes': 0,
                        'total_regret': 0.0,
                        'top1_count': 0,
                        'top3_count': 0,
                        'regrets': []
                    }
                
                perf = category_performance[category]
                perf['episodes'] += 1
                perf['total_regret'] += episode_info.get('regret', 0.0)
                perf['regrets'].append(episode_info.get('regret', 0.0))
                
                if episode_info.get('top1', False):
                    perf['top1_count'] += 1
                if episode_info.get('top3', False):
                    perf['top3_count'] += 1
                
                episode_data = {
                    'episode': episode_num,
                    'category': category,
                    'regret': episode_info.get('regret', 0.0),
                    'top1': episode_info.get('top1', False),
                    'top3': episode_info.get('top3', False),
                    'chosen_score': episode_info.get('chosen_score', 0.0),
                    'best_score': episode_info.get('best_score', 0.0)
                }
                all_episodes.append(episode_data)
                successful_episodes_count += 1
                
            except Exception as e:
                print(f"  Error in episode {episode_num}: {e}")
                continue
    
    episode_regrets = [ep['regret'] for ep in all_episodes]
    avg_regret = np.mean(episode_regrets) if episode_regrets else 0.0
    
    if len(episode_regrets) >= 10:
        first_half = np.mean(episode_regrets[:len(episode_regrets)//2])
        second_half = np.mean(episode_regrets[len(episode_regrets)//2:])
        if second_half < first_half - 0.1:
            regret_trend = "improving"
        elif second_half > first_half + 0.1:
            regret_trend = "worsening"
        else:
            regret_trend = "stable"
    else:
        regret_trend = "insufficient_data"
    
    final_metrics = {
        'experiment': 'Oracle Baseline: Direct Recommendation with Full Persona',
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'regret_progression': {
                'episode_regrets': episode_regrets,  
                'avg_regret': avg_regret,
                'regret_trend': regret_trend
            },
            'categories_tested': list(set(ep['category'] for ep in all_episodes)),
            'total_episodes': len(all_episodes),
            'successful_episodes': len(all_episodes),
            'target_successful_episodes': target_total_episodes,
            'episodes_by_category': {cat: len([ep for ep in all_episodes if ep['category'] == cat]) for cat in set(ep['category'] for ep in all_episodes)},
            'overall_metrics': {
                'avg_regret': avg_regret,
                'top1_rate': np.mean([ep['top1'] for ep in all_episodes]) if all_episodes else 0.0,
                'top3_rate': np.mean([ep['top3'] for ep in all_episodes]) if all_episodes else 0.0,
                'regret_std': np.std([ep['regret'] for ep in all_episodes]) if all_episodes else 0.0
            }
        },
        'config': {
            'model': agent.model,
            'persona_index': agent.persona_index,
            'categories': categories,
            'episodes_per_category': episodes_per_category,
            'seed': seed
        },
        'category_results': {}
    }
    
    for category, perf in category_performance.items():
        if perf['episodes'] > 0:
            category_episodes = [ep for ep in all_episodes if ep['category'] == category]
            final_metrics['category_results'][category] = {
                'avg_score': np.mean([ep['chosen_score'] for ep in category_episodes]),
                'top1_rate': perf['top1_count'] / perf['episodes'],
                'top3_rate': perf['top3_count'] / perf['episodes'],
                'episode_count': perf['episodes'],
                'avg_regret': perf['total_regret'] / perf['episodes'],
                'regret_std': np.std(perf['regrets'])
            }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    episodes_file = os.path.join(output_dir, f"oracle_episodes_{timestamp}.json")
    with open(episodes_file, 'w') as f:
        json.dump(all_episodes, f, indent=2)
    
    summary_file = os.path.join(output_dir, f"oracle_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nRegret Analysis:")
    print(f"  Average Regret: {avg_regret:.1f}")
    print(f"  Trend: {regret_trend}")
    print(f"  Episodes: {len(episode_regrets)}")
    
    print(f"\nOracle baseline completed!")
    print(f"Overall metrics:")
    print(f"  Average regret: {final_metrics['summary']['overall_metrics']['avg_regret']:.3f}")
    print(f"  Top-1 accuracy: {final_metrics['summary']['overall_metrics']['top1_rate']:.3f}")
    print(f"  Top-3 accuracy: {final_metrics['summary']['overall_metrics']['top3_rate']:.3f}")
    print(f"  Regret std dev: {final_metrics['summary']['overall_metrics']['regret_std']:.3f}")
    
    return final_metrics


def run_baseline_oracle(
    categories: List[str] = None,
    num_categories: int = 3,
    episodes_per_category: int = 1,
    model: str = "gpt-4o",
    persona_index: int = None,
    output_dir: str = "baseline_oracle_results",
    seed: int = 60751,
    min_score_threshold: float = 60.0
) -> Dict[str, Any]:
    """Run oracle baseline with specified parameters."""
    
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    if persona_index is None:
        persona_index = random.randint(0, 47000)
        print(f"Randomly selected persona: {persona_index}")
    else:
        print(f"Using specified persona: {persona_index}")
    
    if categories is None:
        from pipeline.core.simulate_interaction import list_categories
        available_categories = list_categories()
        print(f"Available categories: {len(available_categories)}")

        def get_categories_for_seed(available_categories, seed):
            """Get categories in randomized order based on seed."""
            random.seed(seed)
            categories = available_categories.copy()
            random.shuffle(categories)
            random.seed()  
            return categories
        
        selected_categories = get_categories_for_seed(available_categories, seed)
        print(f"Categories selected with randomization from seed {seed}")
        categories = selected_categories
    
    print(f"Oracle Baseline Configuration:")
    print(f"  Model: {model}")
    print(f"  Persona: {persona_index}")
    print(f"  Num categories: {len(categories)}")
    print(f"  Episodes per category: {episodes_per_category}")
    print(f"  Categories: {len(categories)} total")
    print(f"  Seed: {seed}")
    
    agent = OracleAgent(model=model, persona_index=persona_index)
    
    metrics = train_oracle_agent_multi_category(
        categories=categories,
        cached_scores_map={},  
        agent=agent,
        persona_index=persona_index,
        episodes_per_category=episodes_per_category,
        output_dir=output_dir,
        seed=seed,
        min_score_threshold=min_score_threshold,
        num_categories=num_categories
    )
    
    return metrics


def main():
    """Main training script for oracle baseline."""
    parser = argparse.ArgumentParser(description="Oracle Baseline Training")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--persona_index", type=int, default=None, help="Persona index to use (if None, will be randomly selected based on seed)")
    parser.add_argument("--num_categories", type=int, default=3, help="Number of categories to test")
    parser.add_argument("--episodes_per_category", type=int, default=10, help="Episodes per category")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Categories to test")
    parser.add_argument("--output_dir", type=str, default="oracle_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (same as experiment1 for fair comparison)")
    
    args = parser.parse_args()
    
    return run_baseline_oracle(
        categories=args.categories,
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        model=args.model,
        persona_index=args.persona_index,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
