#!/usr/bin/env python3
"""
Manual Testing 
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
import re
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
from pipeline.envs.reco_env import RecoEnv
from pipeline.core.llm_providers import chat_completion
from pipeline.wrappers.metrics_wrapper import MetricsWrapper


class ManualQuestionsAgent:
    """
    Agent that takes pre-defined questions and answers, then makes a recommendation.
    """
    
    def __init__(self, model: str = "gpt-4o", questions_and_answers: List[Tuple[str, str]] = None):
        self.model = model
        self.questions_and_answers = questions_and_answers or []
        
    def get_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any]) -> int:
        """Make a recommendation based on the provided Q&A."""
        if 'num_products' in info:
            num_products = info['num_products']
            category = info['category']
        else:
            num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
            category = "unknown"
        
        return self._make_recommendation(obs, info, self.questions_and_answers, category, num_products)
    
    def _make_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                           dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Make recommendation based on the provided Q&A."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        base_prompt = f"""You are a product recommendation agent for {category} products.

Context:
{context}

Task:
Based on the conversation with the user, make your final recommendation.

Output format (MUST be exactly one line, no extra text):
RECOMMEND: <number 0-{num_products-1}>

Rules:
- Choose the product that best matches the user's expressed preferences
- Consider all the information gathered from the conversation
- No explanations, just the recommendation
- You must recommend exactly one product"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": base_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=100
            )
            
            if "RECOMMEND:" in response:
                import re
                rec_text = response.split("RECOMMEND:")[-1].strip()
                idx_match = re.search(r"index\s*[:=]\s*(\d+)", rec_text, re.IGNORECASE)
                if idx_match:
                    try:
                        candidate_idx = int(idx_match.group(1))
                        if 0 <= candidate_idx < num_products:
                            return candidate_idx
                    except Exception:
                        pass
                num_match = re.search(r"(\d+)", rec_text)
                if num_match:
                    try:
                        candidate_num = int(num_match.group(1))
                        if 0 <= candidate_num < num_products:
                            return candidate_num
                        product_ids = info.get('product_ids', [])
                        if product_ids and candidate_num in product_ids:
                            mapped_idx = product_ids.index(candidate_num)
                            if 0 <= mapped_idx < num_products:
                                return mapped_idx
                    except Exception:
                        pass
            print(f"Warning: Failed to parse recommendation from response: {response}")
            return 0
            
        except Exception as e:
            print(f"Error making recommendation: {e}")
            return 0
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        product_descriptions = info.get('product_descriptions', [])
        
        for i in range(num_products):
            if i < obs['product_features'].shape[0]:
                features = obs['product_features'][i]
                product = {
                    'id': info.get('product_ids', [])[i] if i < len(info.get('product_ids', [])) else i,
                    'price': float(features[0]) if not np.isnan(features[0]) else 0.0,
                    'store_hash': int(features[1]) if not np.isnan(features[1]) else 0,
                    'title_length': int(features[2]) if not np.isnan(features[2]) else 0,
                    'features': [float(f) for f in features[3:] if not np.isnan(f)],
                    'description': product_descriptions[i] if i < len(product_descriptions) else "No description available"
                }
                products.append(product)
        
        return products
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str) -> str:
        """Build context string for LLM decision making."""
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            description = product.get('description', 'No description available')
            product_list += f"{i}: Product ID {product['id']} - Price: ${product['price']:.2f}\n"
            product_list += f"   Description: {description}\n\n"
        
        dialog_text = "Conversation with user:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No conversation provided.\n"
        
        return f"{product_list}\n{dialog_text}"


def run_manual_questions_interactive(
    persona_index: int,
    category: str,
    model: str = "gpt-4o",
    feedback_type: str = "persona",
    min_score_threshold: float = 60.0,
    output_dir: str = "manual_questions_results",
    exit_key: str = "/done",
    max_products_per_category: Optional[int] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Interactive variant: user types questions in terminal, persona agent answers.
    Type exit_key (default "/done") on a new line to finish and trigger recommendation.
    """
    print(f"=== Manual Questions Experiment (Interactive) ===")
    print(f"Persona: {persona_index}")
    print(f"Category: {category}")
    print(f"Model: {model}, Feedback: {feedback_type}")
    print(f"Type questions to ask the persona. Type {exit_key} to finish and recommend.")

    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)

    from pipeline.core.feedback_system import FeedbackSystem
    from pipeline.core.user_model import UserModel

    persona_agent = UserModel(persona_index)
    if feedback_type == "persona":
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)

    from pipeline.core.simulate_interaction import get_products_by_category

    try:
        products = get_products_by_category(category, limit=max_products_per_category, seed=seed)
        if not products:
            raise ValueError(f"No products found for category: {category}")

        scores = persona_agent.score_products(category, products)
        if scores:
            max_score = max(score for _, score in scores)
            cached_scores = [(pid, score) for pid, score in scores]
            if max_score <= min_score_threshold:
                print(f"Warning: Category {category} max score {max_score:.1f} ≤ {min_score_threshold}")
        else:
            raise ValueError(f"No scores generated for category: {category}")
    except Exception as e:
        print(f"Error checking category {category}: {e}")
        return {'error': str(e)}

    print(f"Category {category}: Max score {max_score:.1f}, proceeding")

    questions_and_answers: List[Tuple[str, str]] = []
    try:
        while True:
            try:
                user_q = input("Q> ").strip()
            except EOFError:
                break
            if not user_q:
                continue
            if user_q == exit_key:
                break
            answer = persona_agent.respond(user_q)
            print(f"A> {answer}")
            questions_and_answers.append((user_q, answer))
    except KeyboardInterrupt:
        print("\nInterrupted. Proceeding to recommendation with collected dialog...")
    agent = ManualQuestionsAgent(
        model=model,
        questions_and_answers=questions_and_answers
    )

    try:
        env = RecoEnv(
            persona_index=persona_index,
            max_questions=0,  
            categories=[category],
            agent=agent,
            feedback_system=feedback_system,
            cached_scores=cached_scores,
            max_products_per_category=max_products_per_category,
            seed=seed
        )

        metrics_wrapper = MetricsWrapper(env,
                                       output_path=os.path.join(output_dir, f"manual_questions_episode.jsonl"))

        obs, initial_info = metrics_wrapper.reset()

        print(f"Products in {category}: {len(env.products)}")
        print(f"Top 3 products by score:")
        for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
            product = next((p for p in env.products if p['id'] == product_id), None)
            if product:
                title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                print(f"  {i+1}. {title} (Score: {score:.1f})")

        action = agent.get_action(obs, initial_info)
        obs, reward, terminated, truncated, final_info = metrics_wrapper.step(action)

        print(f"Final recommendation - Product {final_info['chosen_product_id']}")
        print(f"  Score: {final_info['chosen_score']:.1f}, Best: {final_info['best_score']:.1f}")
        print(f"  Top1: {final_info['top1']}, Top3: {final_info['top3']}")
        if 'feedback' in final_info and final_info['feedback']:
            print(f"  Feedback: {final_info['feedback']}")

        try:
            ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
            if final_info.get('chosen_product_id') in ordered_product_ids:
                chosen_rank = ordered_product_ids.index(final_info.get('chosen_product_id')) + 1
            else:
                chosen_rank = None
        except Exception:
            chosen_rank = None

        final_info['chosen_rank'] = chosen_rank
        final_info['total_products'] = len(env.products) if hasattr(env, 'products') else None

        product_info = {
            'num_products': len(env.products) if hasattr(env, 'products') else 0,
            'products_with_scores': []
        }

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
            'experiment_mode': 'interactive',
            'episode': 1,
            'category': category,
            'persona_index': persona_index,
            'final_info': final_info,
            'questions_and_answers': questions_and_answers,
            'product_info': product_info,
            'chosen_rank': chosen_rank,
            'total_products': len(env.products) if hasattr(env, 'products') else None
        }
        metrics_wrapper.close()
        model_safe_name = model.replace("/", "_").replace(":", "_")
        feedback_safe_name = feedback_type.replace(" ", "_")
        results_file = os.path.join(output_dir, f"manual_questions_experiment_{model_safe_name}_{feedback_safe_name}.json")

        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Manual Questions Experiment (Interactive)',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'persona_index': persona_index,
                    'category': category,
                    'num_questions': len(questions_and_answers),
                    'chosen_score': final_info.get('chosen_score', 0),
                    'best_score': final_info.get('best_score', 0),
                    'regret': final_info.get('regret', 0),
                    'chosen_rank': chosen_rank,
                    'total_products': len(env.products) if hasattr(env, 'products') else 0,
                    'top1': final_info.get('top1', False),
                    'top3': final_info.get('top3', False)
                },
                'config': {
                    'persona_index': persona_index,
                    'category': category,
                    'model': model,
                    'feedback_type': feedback_type
                },
                'episode_result': episode_result
            }, f, indent=2)

        print(f"\nResults saved to: {results_file}")
        print(f"Individual episode metrics saved to: {output_dir}/manual_questions_episode.jsonl")

        return {
            'experiment': 'Manual Questions Experiment (Interactive)',
            'episode_result': episode_result
        }

    except Exception as e:
        print(f"Error in experiment: {e}")
        return {'error': str(e)}

def run_manual_questions_experiment(
    persona_index: int,
    category: str,
    questions_and_answers: List[Tuple[str, str]],
    model: str = "gpt-4o",
    feedback_type: str = "persona",
    min_score_threshold: float = 60.0,
    output_dir: str = "manual_questions_results",
    seed: Optional[int] = None,
    max_products_per_category: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run manual questions experiment.
    
    Args:
        persona_index: Specific persona to use
        category: Specific category to test
        questions_and_answers: List of (question, answer) tuples
        model: LLM model to use
        feedback_type: Type of feedback to provide
        min_score_threshold: Minimum score threshold for category relevance
        output_dir: Directory to save results
        seed: Random seed for reproducibility
    """
    
    print(f"=== Manual Questions Experiment ===")
    print(f"Persona: {persona_index}")
    print(f"Category: {category}")
    print(f"Questions provided: {len(questions_and_answers)}")
    print(f"Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    from pipeline.core.feedback_system import FeedbackSystem
    from pipeline.core.user_model import UserModel
    
    if feedback_type == "persona":
        persona_agent = UserModel(persona_index)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from pipeline.core.simulate_interaction import get_products_by_category
    
    try:
        products = get_products_by_category(category, limit=max_products_per_category, seed=seed)
        if not products:
            raise ValueError(f"No products found for category: {category}")
            
        user_model = UserModel(persona_index)
        scores = user_model.score_products(category, products)
        if scores:
            max_score = max(score for _, score in scores)
            cached_scores = [(pid, score) for pid, score in scores]
            if max_score <= min_score_threshold:
                print(f"Warning: Category {category} max score {max_score:.1f} ≤ {min_score_threshold}")
        else:
            raise ValueError(f"No scores generated for category: {category}")
    except Exception as e:
        print(f"Error checking category {category}: {e}")
        return {'error': str(e)}
    
    print(f"Category {category}: Max score {max_score:.1f}, proceeding")
    
    agent = ManualQuestionsAgent(
        model=model,
        questions_and_answers=questions_and_answers
    )
    
    try:
        env = RecoEnv(
            persona_index=persona_index,
            max_questions=0, 
            categories=[category],  
            agent=agent,
            feedback_system=feedback_system,
            cached_scores=cached_scores,
            max_products_per_category=max_products_per_category,
            seed=seed
        )
        
        metrics_wrapper = MetricsWrapper(env, 
                                       output_path=os.path.join(output_dir, f"manual_questions_episode.jsonl"))
        
        obs, initial_info = metrics_wrapper.reset()
        
        print(f"Products in {category}: {len(env.products)}")
        print(f"Top 3 products by score:")
        for i, (product_id, score) in enumerate(env.oracle_scores[:3]):
            product = next((p for p in env.products if p['id'] == product_id), None)
            if product:
                title = product.get('title', 'Unknown')[:50] + "..." if len(product.get('title', '')) > 50 else product.get('title', 'Unknown')
                print(f"  {i+1}. {title} (Score: {score:.1f})")
        
        action = agent.get_action(obs, initial_info)
        obs, reward, terminated, truncated, final_info = metrics_wrapper.step(action)
        
        print(f"Final recommendation - Product {final_info['chosen_product_id']}")
        print(f"  Score: {final_info['chosen_score']:.1f}, Best: {final_info['best_score']:.1f}")
        print(f"  Top1: {final_info['top1']}, Top3: {final_info['top3']}")
        if 'feedback' in final_info and final_info['feedback']:
            print(f"  Feedback: {final_info['feedback']}")
        
        try:
            ordered_product_ids = [pid for pid, _ in env.oracle_scores] if hasattr(env, 'oracle_scores') else []
            if final_info.get('chosen_product_id') in ordered_product_ids:
                chosen_rank = ordered_product_ids.index(final_info.get('chosen_product_id')) + 1
            else:
                chosen_rank = None
        except Exception:
            chosen_rank = None

        final_info['chosen_rank'] = chosen_rank
        final_info['total_products'] = len(env.products) if hasattr(env, 'products') else None
        
        product_info = {
            'num_products': len(env.products) if hasattr(env, 'products') else 0,
            'products_with_scores': []
        }
        
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
            'episode': 1,
            'category': category,
            'persona_index': persona_index,
            'final_info': final_info,
            'questions_and_answers': questions_and_answers,
            'product_info': product_info,
            'chosen_rank': chosen_rank,
            'total_products': len(env.products) if hasattr(env, 'products') else None
        }
        
        metrics_wrapper.close()
        
        model_safe_name = model.replace("/", "_").replace(":", "_")
        feedback_safe_name = feedback_type.replace(" ", "_")
        seed_suffix = f"_seed{seed}" if seed is not None else ""
        results_file = os.path.join(output_dir, f"manual_questions_experiment_{model_safe_name}_{feedback_safe_name}{seed_suffix}.json")
        
        with open(results_file, 'w') as f:
            json.dump({
                'experiment': 'Manual Questions Experiment',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'persona_index': persona_index,
                    'category': category,
                    'num_questions': len(questions_and_answers),
                    'chosen_score': final_info.get('chosen_score', 0),
                    'best_score': final_info.get('best_score', 0),
                    'regret': final_info.get('regret', 0),
                    'chosen_rank': chosen_rank,
                    'total_products': len(env.products) if hasattr(env, 'products') else 0,
                    'top1': final_info.get('top1', False),
                    'top3': final_info.get('top3', False)
                },
                'config': {
                    'persona_index': persona_index,
                    'category': category,
                    'model': model,
                    'feedback_type': feedback_type,
                    'seed': seed
                },
                'episode_result': episode_result
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Individual episode metrics saved to: {output_dir}/manual_questions_episode.jsonl")
        
        return {
            'experiment': 'Manual Questions Experiment',
            'episode_result': episode_result
        }
        
    except Exception as e:
        print(f"Error in experiment: {e}")
        return {'error': str(e)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Questions Experiment")
    parser.add_argument("--persona_index", type=int, required=True, help="Persona index to use")
    parser.add_argument("--category", required=True, help="Category to test")
    parser.add_argument("--questions_file", help="JSON file containing questions and answers")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--feedback_type", default="persona", help="Feedback type")
    parser.add_argument("--min_score_threshold", type=float, default=60.0, help="Min score threshold")
    parser.add_argument("--output_dir", default="manual_questions_results", help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive terminal mode")
    
    args = parser.parse_args()
    
    if not args.interactive:
        print("Error: Non-interactive mode is disabled. Run with --interactive.")
        exit(1)

    run_manual_questions_interactive(
        persona_index=args.persona_index,
        category=args.category,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir
    )
