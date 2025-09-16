#!/usr/bin/env python3
"""
Experiment 1: Cross-Category Learning with Same User Persona.

This experiment tests whether an LLM can learn latent user preferences that 
transfer across different product categories. The agent interacts with the same
user persona across multiple categories sequentially.

Key questions:
1. Can the LLM learn latent user preferences that transfer across categories?
2. Does recommendation performance improve as the agent experiences more categories?
3. Is the feedback signal sufficient for cross-category learning?

Setup: Same user persona, different categories tested sequentially.
Hypothesis: Agent should learn consistent preferences (e.g., price sensitivity, 
brand preferences) that apply across all categories.
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime

# Import our modules
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .core.metrics import EpisodeRecord, MetricsRecorder
from .wrappers.metrics_wrapper import MetricsWrapper


class LLMAgent:
    """
    LLM-based agent that can ask questions and make recommendations.
    This agent should learn user preferences across episodes.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8):
        self.model = model
        self.max_questions = max_questions
        self.episode_count = 0
        self.learned_preferences = {}  # Store learned user preferences
        self.current_episode_info = None  # Store current episode info
        self.last_response = None  # Store last LLM response for question extraction
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """
        Decide whether to ask a question or make a recommendation using LLM.
        
        Args:
            obs: Current observation (product features, dialog history, etc.)
            info: Environment info (category, product_ids, etc.)
            
        Returns:
            Action: 0 to num_products-1 for recommend, num_products for ask question
        """
        # Handle different info formats from reset() vs step()
        if 'num_products' in info:
            # From reset() - has full environment info, store it
            self.current_episode_info = info
            num_products = info['num_products']
            category = info['category']
        else:
            # From step() - use stored episode info
            if self.current_episode_info is None:
                # Fallback: count non-zero product features
                num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
                category = "unknown"
            else:
                num_products = self.current_episode_info['num_products']
                category = self.current_episode_info['category']
        
        # Get dialog history from environment if available
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        # Use LLM to decide whether to ask or recommend
        return self._llm_decide_action(obs, info, dialog_history, category, num_products)
    
    def _extract_dialog_history(self, obs: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
        """Extract dialog history from observation."""
        # For now, return empty list - we'll get dialog from environment directly
        # The character embedding decoding is complex and error-prone
        return []
    
    def _decode_char_embedding(self, char_array: np.ndarray) -> str:
        """Decode character embedding back to text."""
        # Convert normalized values back to ASCII
        chars = []
        for val in char_array:
            if val > 0:  # Non-zero values
                char_code = int(val * 127)  # Reverse the normalization
                if 32 <= char_code <= 126:  # Printable ASCII range
                    chars.append(chr(char_code))
        
        return ''.join(chars).strip()
    
    def _llm_decide_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        
        # Get product information
        products = self._get_product_info(obs, info, num_products)
        
        # Build context for LLM
        context = self._build_llm_context(products, dialog_history, category)
        
        # Single unified prompt
        unified_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for a user by asking strategic questions.

{context}

Based on the conversation so far, you have two options:

1. If you need more information, ask a strategic question to learn about their preferences
2. If you're confident enough, recommend a specific product

RESPOND IN ONE OF THESE FORMATS:

To ask a question:
QUESTION: [Your strategic question here]

To recommend a product:
RECOMMEND: [Product number 0-{num_products-1}]

Examples:
QUESTION: What's your budget range for this category?
RECOMMEND: 3

Your response:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=150
            )
            
            # Store the full response for question extraction
            self.last_response = response.strip()
            response = self.last_response.upper()
            
            if response.startswith("RECOMMEND:"):
                # Extract product number
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        return product_idx
                
                # Fallback to first product if parsing fails
                print(f"[WARN] Could not parse recommendation '{response}', using product 0")
                return 0
                
            elif response.startswith("QUESTION:"):
                # Ask a question - return num_products to trigger question generation
                return num_products
            else:
                # Fallback - try to extract any number for recommendation
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        print(f"[INFO] Interpreted response as recommendation: {product_idx}")
                        return product_idx
                
                # Default to asking a question
                print(f"[WARN] Unclear response '{response}', asking question")
                return num_products
                
        except Exception as e:
            print(f"[WARN] LLM decision failed: {e}, falling back to heuristic")
            # Fallback to simple heuristic
            if len(dialog_history) < 3:
                return num_products
            else:
                return self._choose_recommendation(obs, info, dialog_history)
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        product_features = obs['product_features']
        
        for i in range(num_products):
            if i < len(info.get('product_ids', [])):
                product_id = info['product_ids'][i]
                features = product_features[i]
                
                # Extract basic features (these are simplified - in practice you'd have more detailed product data)
                price = features[0] * 1000  # Denormalize price
                store_hash = features[1]
                title_length = features[2] * 100  # Denormalize title length
                
                products.append({
                    'id': product_id,
                    'price': f"${price:.0f}",
                    'store_hash': f"{store_hash:.2f}",
                    'title_length': f"{title_length:.0f} chars"
                })
        
        return products
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str) -> str:
        """Build context string for LLM decision making."""
        
        # Product list
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            product_list += f"{i}: Product ID {product['id']} - Price: {product['price']}, Store: {product['store_hash']}, Title: {product['title_length']}\n"
        
        # Dialog history
        dialog_text = "Conversation so far:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No questions asked yet.\n"
        
        return f"{product_list}\n{dialog_text}"
    
    
    
    def _choose_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                             dialog_history: List[Tuple[str, str]]) -> int:
        """Choose which product to recommend based on learned preferences."""
        # Use stored episode info for num_products
        if self.current_episode_info is not None:
            num_products = self.current_episode_info['num_products']
        else:
            # Fallback: count non-zero product features
            num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
        
        # Simple strategy: choose based on price preferences learned from dialog
        # In a real implementation, this would use the LLM to analyze preferences
        
        # For now, use a simple heuristic based on product features
        product_features = obs['product_features']
        
        # Extract price information (feature 0 is normalized price)
        prices = product_features[:num_products, 0]
        
        # Simple preference: prefer mid-range products (avoid extremes)
        price_scores = 1.0 - np.abs(prices - 0.5)  # Higher score for prices closer to 0.5
        best_product_idx = np.argmax(price_scores)
        
        return int(best_product_idx)
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """Update learned preferences based on episode outcome."""
        self.episode_count += 1
        
        # Store episode results for learning
        if 'chosen_score' in episode_result:
            score = episode_result['chosen_score']
            category = episode_result.get('category', 'unknown')
            
            # Simple learning: track performance by category
            if category not in self.learned_preferences:
                self.learned_preferences[category] = []
            self.learned_preferences[category].append(score)


def run_experiment1(persona_index: int = 42, 
                   categories: List[str] = None,
                   num_categories: int = 5,
                   episodes_per_category: int = 5,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   output_dir: str = "experiment1_results"):
    """
    Run Experiment 1: LLM learning across categories.
    
    Args:
        persona_index: Which persona to use (consistent across episodes)
        categories: List of categories to test (None = randomly choose)
        num_categories: Number of categories to randomly select (if categories is None)
        episodes_per_category: Number of episodes per category
        max_questions: Maximum questions per episode
        model: LLM model to use
        output_dir: Directory to save results
    """
    
    print(f"=== Experiment 1: LLM Learning Across Categories ===")
    print(f"Persona: {persona_index}")
    print(f"Episodes per category: {episodes_per_category}")
    print(f"Max questions: {max_questions}")
    print(f"Model: {model}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Register environment
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Create LLM agent
    agent = LLMAgent(model=model, max_questions=max_questions)
    
    # Get available categories
    from .core.simulate_interaction import list_categories
    import random
    available_categories = list_categories()
    if categories is None:
        # Randomly choose specified number of categories
        if len(available_categories) >= num_categories:
            categories = random.sample(available_categories, num_categories)
        else:
            categories = available_categories  # Use all if less than requested available
    else:
        categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Categories: {categories}")
    
    # Track results
    all_results = []
    category_results = {cat: [] for cat in categories}
    
    # Run episodes
    total_episodes = len(categories) * episodes_per_category
    episode_num = 0
    
    for category in categories:
        print(f"\n--- Testing Category: {category} ---")
        
        # Show product info for this category (from first episode)
        first_env = None
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            # Create environment for this episode
            env = RecoEnv(
                          persona_index=persona_index,
                          max_questions=max_questions,
                categories=[category],  
                agent=agent 
            )
            
            # Wrap with metrics
            metrics_wrapper = MetricsWrapper(env, 
                                           output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            # Reset environment
            obs, initial_info = metrics_wrapper.reset()
            
            # Show product info for first episode of each category
            if episode == 0:
                print(f"  Products in {category}: {len(env.products)}")
                print(f"  Top 3 products by score:")
                for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
                    product = next((p for p in env.products if p['id'] == product_id), None)
                    if product:
                        title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                        print(f"    {i+1}. {title} (Score: {score:.1f})")
            
            # Pass environment reference to agent for dialog access
            agent.current_env = env
            
            # Run episode
            terminated = False
            truncated = False
            step_count = 0
            current_info = initial_info  # Use initial info for first action
            
            while not terminated and not truncated and step_count < 20:  # Safety limit
                # Get action from agent
                action = agent.get_action(obs, current_info)
                
                # Take step
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info  # Update info for next iteration
                step_count += 1
                
                # Print progress
                if info['action_type'] == 'ask':
                    print(f"  Step {step_count}: Asked question")
                elif info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}")
                    print(f"    Top1: {info['top1']}, Top3: {info['top3']}")
                    break
            
            # Get full dialog history from environment
            full_dialog = []
            if hasattr(env, 'dialog_history'):
                full_dialog = env.dialog_history
            
            # Get simplified product information with individual scores
            product_info = {
                'num_products': len(env.products) if hasattr(env, 'products') else 0,
                'products_with_scores': []
            }
            
            if hasattr(env, 'products') and hasattr(env, 'oracle_scores'):
                # Get individual OpenAI and Gemini scores from the scoring process
                from .core.simulate_interaction import score_products_for_persona
                from .core.user_model import UserModel
                
                user_model = UserModel(persona_index)
                persona_text = user_model.get_persona_text()
                
                # Get detailed scores with individual model results
                detailed_scores = score_products_for_persona(persona_text, category, env.products)
                
                # Create simplified product list with names and individual scores
                for product_id, score, reason in detailed_scores:
                    product = next((p for p in env.products if p['id'] == product_id), None)
                    if product:
                        # Parse reason to extract individual scores
                        openai_score = None
                        gemini_score = None
                        if "OpenAI score:" in reason and "Gemini score:" in reason:
                            try:
                                parts = reason.split(" | ")
                                openai_part = parts[0].replace("OpenAI score: ", "")
                                gemini_part = parts[1].replace("Gemini score: ", "")
                                openai_score = float(openai_part) if openai_part != "N/A" else None
                                gemini_score = float(gemini_part) if gemini_part != "N/A" else None
                            except:
                                pass
                        
                        product_info['products_with_scores'].append({
                            'id': product_id,
                            'name': product.get('title', 'Unknown'),
                            'price': product.get('price', 'Unknown'),
                            'openai_score': openai_score,
                            'gemini_score': gemini_score,
                            'average_score': score
                        })
            
            # Store episode results
            episode_result = {
                'episode': episode_num,
                'category': category,
                'episode_in_category': episode + 1,
                'steps': step_count,
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info,
                'full_dialog': full_dialog,  # Add complete conversation history
                'product_info': product_info  # Add product list and scores
            }
            
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            
            # Update agent preferences
            agent.update_preferences(episode_result)
            
            # Close environment
            metrics_wrapper.close()
    
    # Analyze results
    print(f"\n=== Results Analysis ===")
    
    # Performance by category
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  {category}:")
            print(f"    Avg Score: {avg_score:.1f}")
            print(f"    Top1 Rate: {top1_rate:.1%}")
            print(f"    Episodes: {len(scores)}")
    
    # Learning progression (performance over episodes within each category)
    print("\nLearning Progression (Performance over episodes):")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
    # Calculate summary statistics
    episode_regrets = []
    episode_questions = []
    episode_scores = []
    
    for result in all_results:
        if 'chosen_score' in result['final_info']:
            episode_scores.append(result['final_info']['chosen_score'])
        if 'regret' in result['final_info']:
            episode_regrets.append(result['final_info']['regret'])
        if 'questions_asked' in result['final_info']:
            episode_questions.append(result['final_info']['questions_asked'])
    
    # Calculate progression metrics
    regret_progression = {
        'episode_regrets': episode_regrets,
        'avg_regret': np.mean(episode_regrets) if episode_regrets else 0,
        'regret_trend': 'improving' if len(episode_regrets) > 1 and episode_regrets[-1] < episode_regrets[0] else 'stable'
    }
    
    questions_progression = {
        'episode_questions': episode_questions,
        'avg_questions': np.mean(episode_questions) if episode_questions else 0,
        'total_questions': sum(episode_questions) if episode_questions else 0
    }
    
    # Save detailed results
    results_file = os.path.join(output_dir, "experiment1_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 1: LLM Learning Across Categories',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression,
                'overall_performance': {
                    'avg_score': np.mean(episode_scores) if episode_scores else 0,
                    'total_episodes': len(all_results),
                    'successful_episodes': len([r for r in all_results if 'chosen_score' in r['final_info']])
                }
            },
            'config': {
                'persona_index': persona_index,
                'categories': categories,
                'episodes_per_category': episodes_per_category,
                'max_questions': max_questions,
                'model': model
            },
            'results': all_results,
            'category_summary': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'products_with_scores': results[0]['product_info']['products_with_scores'] if results else []
                }
                for cat, results in category_results.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    
    return all_results, category_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1: LLM Learning Across Categories")
    parser.add_argument("--persona_index", type=int, default=42, help="Persona index to use")
    parser.add_argument("--categories", nargs="+", default=None, help="Categories to test")
    parser.add_argument("--num_categories", type=int, default=5, help="Number of categories to randomly select")
    parser.add_argument("--episodes_per_category", type=int, default=5, help="Episodes per category")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--output_dir", type=str, default="experiment1_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_experiment1(
        persona_index=args.persona_index,
        categories=args.categories,
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        max_questions=args.max_questions,
        model=args.model,
        output_dir=args.output_dir
    )
