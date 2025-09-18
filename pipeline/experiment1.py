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
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
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
        """Decide whether to ask a question or make a recommendation using LLM."""
        if 'num_products' in info:
            self.current_episode_info = info
            num_products = info['num_products']
            category = info['category']
        else:
            if self.current_episode_info is None:
                num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
                category = "unknown"
            else:
                num_products = self.current_episode_info['num_products']
                category = self.current_episode_info['category']
        
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        return self._llm_decide_action(obs, info, dialog_history, category, num_products)
    
    def _llm_decide_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        unified_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

Task:
Based on the conversation so far, either:
- Ask one short, consumer-friendly question to clarify user preferences, or
- If sufficiently confident, recommend one product by index. 

Output format (MUST be exactly one line, no extra text):
- To ask: QUESTION: <your question>
- To recommend: RECOMMEND: <number 0-{num_products-1}>

Rules:
- Do not include explanations, reasoning, bullets, or multiple questions
- Avoid jargon; use everyday language a shopper understands
- Keep questions specific and helpful (budget, size, brand/style preference, key feature)
- No meta commentary like "this is strategic because…", only the question or recommendation
"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=400
            )
            
            self.last_response = response.strip()
            response = self.last_response.upper()
            
            if response.startswith("RECOMMEND:"):
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        return product_idx
                print(f"[WARN] Could not parse recommendation '{response}', using product 0")
                return 0
                
            elif response.startswith("QUESTION:"):
                return num_products
            else:
                import re
                numbers = re.findall(r'\d+', response)
                if numbers:
                    product_idx = int(numbers[0])
                    if 0 <= product_idx < num_products:
                        print(f"[INFO] Interpreted response as recommendation: {product_idx}")
                        return product_idx
                print(f"[WARN] Unclear response '{response}', asking question")
                return num_products
                
        except Exception as e:
            raise RuntimeError(f"LLM decision failed: {e}")
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        product_features = obs['product_features']
        
        for i in range(num_products):
            if i < len(info.get('product_ids', [])):
                product_id = info['product_ids'][i]
                features = product_features[i]
                
                price = features[0] * 1000
                store_hash = features[1]
                title_length = features[2] * 100
                
                products.append({
                    'id': product_id,
                    'price': f"${price:.0f}",
                    'store_hash': f"{store_hash:.2f}",
                    'title_length': f"{title_length:.0f} chars"
                })
        
        return products
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str) -> str:
        """Build context string for LLM decision making."""
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            product_list += f"{i}: Product ID {product['id']} - Price: {product['price']}, Store: {product['store_hash']}, Title: {product['title_length']}\n"
        
        dialog_text = "Conversation so far:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No questions asked yet.\n"
        
        return f"{product_list}\n{dialog_text}"
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """Update learned preferences based on episode outcome."""
        self.episode_count += 1
        
        if 'final_info' in episode_result and 'chosen_score' in episode_result['final_info']:
            score = episode_result['final_info']['chosen_score']
            category = episode_result.get('category', 'unknown')
            feedback = episode_result['final_info'].get('feedback', '')
            feedback_type = episode_result['final_info'].get('feedback_type', 'regret')
            
            if category not in self.learned_preferences:
                self.learned_preferences[category] = []
            
            # Store both score and feedback for analysis
            self.learned_preferences[category].append({
                'score': score,
                'feedback': feedback,
                'feedback_type': feedback_type,
                'episode': self.episode_count
            })
            
            # Process feedback for learning
            self._process_feedback(feedback, feedback_type, score, category)
    
    def _process_feedback(self, feedback: str, feedback_type: str, score: float, category: str):
        """Process different types of feedback for learning."""
        if not feedback:  # No feedback case
            return
        
        # Store feedback history for analysis
        if not hasattr(self, 'feedback_history'):
            self.feedback_history = []
        
        self.feedback_history.append({
            'feedback': feedback,
            'feedback_type': feedback_type,
            'score': score,
            'category': category,
            'episode': self.episode_count
        })
        
        # Process feedback based on type
        if feedback_type == "regret":
            self._process_regret_feedback(feedback, score, category)
        elif feedback_type == "quality":
            self._process_quality_feedback(feedback, score, category)
    
    def _process_regret_feedback(self, feedback: str, score: float, category: str):
        """Process precise regret feedback."""
        # Extract regret value from feedback
        import re
        regret_match = re.search(r'Regret: ([\d.]+)', feedback)
        if regret_match:
            regret = float(regret_match.group(1))
            # Learn from regret - lower regret is better
            if not hasattr(self, 'regret_history'):
                self.regret_history = {}
            if category not in self.regret_history:
                self.regret_history[category] = []
            self.regret_history[category].append(regret)
    
    def _process_quality_feedback(self, feedback: str, score: float, category: str):
        """Process qualitative feedback."""
        # Extract quality level from feedback
        feedback_lower = feedback.lower()
        if 'great' in feedback_lower:
            quality = 'great'
        elif 'ok' in feedback_lower:
            quality = 'ok'
        elif 'bad' in feedback_lower:
            quality = 'bad'
        else:
            quality = 'unknown'
        
        # Store quality for learning
        if not hasattr(self, 'quality_history'):
            self.quality_history = {}
        if category not in self.quality_history:
            self.quality_history[category] = []
        self.quality_history[category].append(quality)


def run_experiment1(persona_index: int = 42, 
                   categories: List[str] = None,
                   num_categories: int = 5,
                   episodes_per_category: int = 5,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   feedback_type: str = "none",
                   min_score_threshold: float = 50.0,
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
        feedback_type: Type of feedback to provide ("none", "regret", "quality")
        min_score_threshold: Minimum score threshold for category relevance (default: 50.0)
        output_dir: Directory to save results
    """
    
    print(f"=== Experiment 1: LLM Learning Across Categories ===")
    print(f"Persona: {persona_index}, Episodes per category: {episodes_per_category}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    agent = LLMAgent(model=model, max_questions=max_questions)
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from .core.simulate_interaction import list_categories, get_products_by_category
    import random
    available_categories = list_categories()
    
    # Dynamic category filtering - check relevance as we encounter categories
    def is_category_relevant_for_persona(category, persona_index, min_score_threshold):
        """Check if a category is relevant for a specific persona."""
        from .core.user_model import UserModel
        try:
            products = get_products_by_category(category)
            if not products:
                return False, 0.0, []
                
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                return max_score > min_score_threshold, max_score, scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []
    
    # Initialize category selection
    if categories is None:
        # Start with a random selection from available categories
        if len(available_categories) >= num_categories:
            selected_categories = random.sample(available_categories, num_categories)
        else:
            selected_categories = available_categories.copy()
    else:
        # Use provided categories, filtered by availability
        selected_categories = [cat for cat in categories if cat in available_categories]
    
    print(f"Initial categories: {selected_categories}")
    
    all_results = []
    category_results = {}
    used_categories = set()
    
    total_episodes = len(selected_categories) * episodes_per_category
    episode_num = 0
    
    for category in selected_categories:
        print(f"\n--- Testing Category: {category} ---")
        
        # Check if this category is relevant for the persona
        is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
        if not is_relevant:
            print(f"  Category {category}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping category")
            # Skip all episodes for this category
            for episode in range(episodes_per_category):
                episode_num += 1
            continue
        
        print(f"  Category {category}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        used_categories.add(category)
        if category not in category_results:
            category_results[category] = []
        
        for episode in range(episodes_per_category):
            episode_num += 1
            print(f"Episode {episode_num}/{total_episodes} (Category: {category})")
            
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=max_questions,
                categories=[category],  
                agent=agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores
            )
            
            metrics_wrapper = MetricsWrapper(env, 
                                           output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            obs, initial_info = metrics_wrapper.reset()
            
            if episode == 0:
                print(f"  Products in {category}: {len(env.products)}")
                print(f"  Top 3 products by score:")
                for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
                    product = next((p for p in env.products if p['id'] == product_id), None)
                    if product:
                        title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                        print(f"    {i+1}. {title} (Score: {score:.1f})")
            
            agent.current_env = env
            
            terminated = False
            truncated = False
            step_count = 0
            current_info = initial_info
            
            while not terminated and not truncated and step_count < 20:
                action = agent.get_action(obs, current_info)
                obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                current_info = info
                step_count += 1
                
                if info['action_type'] == 'ask':
                    print(f"  Step {step_count}: Asked question")
                elif info['action_type'] == 'recommend':
                    print(f"  Step {step_count}: Recommended product {info['chosen_product_id']}")
                    print(f"    Score: {info['chosen_score']:.1f}, Best: {info['best_score']:.1f}")
                    print(f"    Top1: {info['top1']}, Top3: {info['top3']}")
                    if 'feedback' in info and info['feedback']:
                        print(f"    Feedback: {info['feedback']}")
                    break
            
            full_dialog = []
            if hasattr(env, 'dialog_history'):
                full_dialog = env.dialog_history
            
            product_info = {
                'num_products': len(env.products) if hasattr(env, 'products') else 0,
                'products_with_scores': []
            }
            
            # To keep regret and displayed scores consistent, use the environment's
            # oracle_scores (already averaged across providers) instead of
            # re-scoring here.
            if hasattr(env, 'products') and hasattr(env, 'oracle_scores'):
                id_to_product = {p['id']: p for p in env.products}
                for product_id, avg_score in env.oracle_scores:
                    product = id_to_product.get(product_id)
                    if product:
                        product_info['products_with_scores'].append({
                            'id': product_id,
                            'name': product.get('title', 'Unknown'),
                            'price': product.get('price', 'Unknown'),
                            'average_score': float(avg_score)
                        })
            
            episode_result = {
                'episode': episode_num,
                'category': category,
                'episode_in_category': episode + 1,
                'steps': step_count,
                'terminated': terminated,
                'truncated': truncated,
                'final_info': info,
                'full_dialog': full_dialog,
                'product_info': product_info
            }
            
            all_results.append(episode_result)
            category_results[category].append(episode_result)
            agent.update_preferences(episode_result)
            metrics_wrapper.close()
    
    print(f"\n=== Results Analysis ===")
    
    print("\nPerformance by Category:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  {category}: Avg Score: {avg_score:.1f}, Top1 Rate: {top1_rate:.1%}, Episodes: {len(scores)}")
    
    print("\nLearning Progression:")
    for category, results in category_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  {category}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
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
    
    # Category information
    category_info = {}
    for cat, results in category_results.items():
        if results:
            category_info[cat] = {
                'num_products': results[0]['product_info']['num_products'],
                'episodes': len(results)
            }
    
    # Create model-specific filename to avoid collisions
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace("/", "_").replace(":", "_")
    results_file = os.path.join(output_dir, f"experiment1_results_{model_safe_name}_{feedback_safe_name}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 1: LLM Learning Across Categories',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression,
                'category_info': category_info,
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
                'model': model,
                'feedback_type': feedback_type
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
    parser.add_argument("--feedback_type", type=str, default="none", choices=["none", "regret", "quality"], help="Type of feedback to provide")
    parser.add_argument("--min_score_threshold", type=float, default=50.0, help="Minimum score threshold for category relevance")
    parser.add_argument("--output_dir", type=str, default="experiment1_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_experiment1(
        persona_index=args.persona_index,
        categories=args.categories,
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        max_questions=args.max_questions,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir
    )