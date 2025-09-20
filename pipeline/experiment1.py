#!/usr/bin/env python3
"""
Experiment 1: LLM Learning Across Categories.
"""

import gymnasium as gym
import numpy as np
import json
import os
import random
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper


class LLMAgent:
    """
    LLM-based agent that can ask questions and make recommendations.
    Maintains simple context: dialog + feedback from previous episodes.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8, context_mode: str = "raw", prompting_tricks: str = "none"):
        self.model = model
        self.max_questions = max_questions
        self.context_mode = context_mode  
        self.prompting_tricks = prompting_tricks  
        self.episode_count = 0
        self.episode_history = [] 
        self.episode_summaries = []  
        self.current_episode_info = None  
        self.last_response = None 
        
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
        
        if len(dialog_history) >= self.max_questions:
            return self._force_recommendation(obs, info, dialog_history, category, num_products)
        
        return self._llm_decide_action(obs, info, dialog_history, category, num_products)
    
    def _force_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                             dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Force the agent to make a recommendation by prompting it to do so without revealing the limit."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        # Build previous episode context
        strategy_context = self._build_feedback_context(category)
        
        base_prompt = f"""You are a product recommendation agent for {category} products.

Context:
{context}

{strategy_context}

Task:
Based on the extensive conversation so far, you now have sufficient information to make a recommendation. Choose the best product for the customer.

Output format (MUST be exactly one line, no extra text):
RECOMMEND: <number 0-{num_products-1}>

Rules:
- You must make a recommendation now
- Choose the product that best matches the customer's expressed preferences
- Do not ask any more questions
- No explanations, just the recommendation
"""

        # Apply prompting tricks if enabled
        unified_prompt = self._apply_prompting_tricks(base_prompt)

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=32000
            )
            
            self.last_response = response
            
            # Parse recommendation
            if "RECOMMEND:" in response:
                try:
                    product_idx = int(response.split("RECOMMEND:")[-1].strip())
                    if 0 <= product_idx < num_products:
                        return product_idx
                except (ValueError, IndexError):
                    pass
            
            # Fallback: return first product if parsing fails
            return 0
            
        except Exception as e:
            print(f"Error in forced recommendation: {e}")
            return 0
    
    def _llm_decide_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        products = self._get_product_info(obs, info, num_products)
        
        import time
        context_start_time = time.time()
        context = self._build_llm_context(products, dialog_history, category)
        context_time = time.time() - context_start_time
        print(f"[TIMING] Context building: {context_time:.2f}s")
        
        feedback_start_time = time.time()
        feedback_context = self._build_feedback_context(category)
        feedback_time = time.time() - feedback_start_time
        print(f"[TIMING] Feedback context building: {feedback_time:.2f}s")
        
        base_prompt = f"""You are a product recommendation agent. Your goal is to find the best product for this user.

Context:
{context}

{feedback_context}

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
- CRITICAL: Do not ask questions that are similar to ones already asked in the conversation
- Build upon previous answers rather than re-asking the same type of question
- If you've gathered enough information, make a recommendation instead of asking more questions
"""

        # Apply prompting tricks if enabled
        unified_prompt = self._apply_prompting_tricks(base_prompt)

        try:
            import time
            llm_start_time = time.time()
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=200
            )
            llm_elapsed = time.time() - llm_start_time
            print(f"[TIMING] LLM agent decision: {llm_elapsed:.2f}s")
            
            self.last_response = response.strip()
            
            if response.strip().startswith("QUESTION:"):
                return num_products  # Ask question action
            elif response.strip().startswith("RECOMMEND:"):
                try:
                    product_index = int(response.strip().split(":")[1].strip())
                    if 0 <= product_index < num_products:
                        return product_index
                    else:
                        print(f"Invalid product index {product_index}, defaulting to ask question")
                        return num_products
                except (ValueError, IndexError):
                    print(f"Could not parse recommendation from: {response.strip()}")
                    return num_products
            else:
                print(f"Unexpected response format: {response.strip()}")
                return num_products
                
        except Exception as e:
            print(f"Error in LLM decision: {e}")
            return num_products  # Default to asking question
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """Extract product information from observation."""
        products = []
        
        for i in range(num_products):
            if i < obs['product_features'].shape[0]:
                features = obs['product_features'][i]
                product = {
                    'id': info.get('product_ids', [])[i] if i < len(info.get('product_ids', [])) else i,
                    'price': float(features[0]) if not np.isnan(features[0]) else 0.0,
                    'store_hash': int(features[1]) if not np.isnan(features[1]) else 0,
                    'title_length': int(features[2]) if not np.isnan(features[2]) else 0,
                    'features': [float(f) for f in features[3:] if not np.isnan(f)]
                }
                products.append(product)
        
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
    
    def _build_feedback_context(self, category: str) -> str:
        """Build context string from previous episodes."""
        if self.context_mode == "none" or not self.episode_history:
            return ""
        
        context_parts = ["Previous episodes:"]
        
        # Choose context mode
        if self.context_mode == "summary" and self.episode_summaries:
            for i, (episode_data, summary) in enumerate(zip(self.episode_history, self.episode_summaries)):
                episode_num = episode_data.get('episode', i + 1)
                episode_category = episode_data.get('category', 'unknown')
                persona = episode_data.get('persona', 'unknown')
                context_parts.append(f"Episode {episode_num}: [{episode_category}/Persona {persona}] Summary:")
                context_parts.append(f"  {summary}")
                context_parts.append("")  # Empty line for readability
        elif self.context_mode == "raw":
            # Show all episodes for context
            for episode_data in self.episode_history:
                episode_num = episode_data.get('episode', 0)
                episode_category = episode_data.get('category', 'unknown')
                persona = episode_data.get('persona', 'unknown')
                dialog = episode_data.get('dialog', [])
                selected_product_id = episode_data.get('selected_product_id', None)
                selected_product_name = episode_data.get('selected_product_name', 'Unknown Product')
                feedback = episode_data.get('feedback', '')
                
                context_parts.append(f"Episode {episode_num}: [{episode_category}/Persona {persona}]")
                
                # Add dialog
                if dialog:
                    for i, (question, answer) in enumerate(dialog):
                        context_parts.append(f"  Q{i+1}: {question}")
                        context_parts.append(f"  A{i+1}: {answer}")
                
                # Add selected product
                if selected_product_id is not None:
                    context_parts.append(f"  Selected Product: {selected_product_name}")
                
                # Add feedback
                if feedback:
                    context_parts.append(f"  Feedback: {feedback}")
                
                context_parts.append("")  # Empty line for readability
        # If context_mode is "summary" but no summaries available, fall back to raw
        
        return "\n".join(context_parts)
    
    def _apply_prompting_tricks(self, base_prompt: str) -> str:
        """Apply prompting tricks to enhance the base prompt."""
        if self.prompting_tricks == "none":
            return base_prompt
        
        elif self.prompting_tricks == "all":
            return f"""{base_prompt}

Let me think through this systematically:
- Customer preferences: [analyze what I know]
- Available products: [analyze the options]
- Best match: [reason about the best choice]
- Decision: [decide whether to ask or recommend]

Let's reason step by step:
1. What do I know about the customer so far?
2. What information am I still missing?
3. Based on this reasoning, what should I do next?

Before making your decision, think again: What are you unsure about regarding this customer? What questions should you ask next? Consider what additional information would help you make a better recommendation.

Think through each step carefully before responding."""
        
        else:
            # Unknown prompting trick, return original prompt
            return base_prompt
    
    def update_preferences(self, episode_result: Dict[str, Any]):
        """Store episode context: dialog + selected product + feedback."""
        self.episode_count += 1
        
        if 'final_info' in episode_result:
            category = episode_result.get('category', 'unknown')
            feedback = episode_result['final_info'].get('feedback', '')
            full_dialog = episode_result.get('full_dialog', [])
            chosen_product_id = episode_result['final_info'].get('chosen_product_id', None)
            persona_index = episode_result.get('persona_index', None)
            
            # Get product name from product_info
            product_name = "Unknown Product"
            if 'product_info' in episode_result and 'products_with_scores' in episode_result['product_info']:
                for product in episode_result['product_info']['products_with_scores']:
                    if product.get('id') == chosen_product_id:
                        product_name = product.get('name', 'Unknown Product')
                        break
            
            # Store episode context with persona and selected product
            episode_data = {
                'episode': self.episode_count,
                'category': category,
                'persona': persona_index,
                'dialog': full_dialog,
                'selected_product_id': chosen_product_id,
                'selected_product_name': product_name,
                'feedback': feedback
            }
            
            self.episode_history.append(episode_data)
            
            # Generate episode summary if context mode is "summary"
            if self.context_mode == "summary":
                summary = self._generate_episode_summary(episode_data)
                self.episode_summaries.append(summary)
    
    def _generate_episode_summary(self, episode_data: Dict[str, Any]) -> str:
        """Generate a summary of the episode that the agent wants to remember for future episodes."""
        episode_num = episode_data.get('episode', 0)
        category = episode_data.get('category', 'unknown')
        persona = episode_data.get('persona', 'unknown')
        dialog = episode_data.get('dialog', [])
        selected_product_id = episode_data.get('selected_product_id', None)
        feedback = episode_data.get('feedback', '')
        
        # Build episode context for summary
        dialog_text = ""
        if dialog:
            for i, (question, answer) in enumerate(dialog):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        
        summary_prompt = f"""You just completed Episode {episode_num} in the {category} category for Persona {persona}.

Episode Details:
{dialog_text}
Selected Product: {selected_product_id}
Feedback: {feedback}

Your task is to provide the context from this episode that you would want a future agent to know. Focus on:
- What worked or didn't work in your approach
- Key insights about user preferences or product selection
- Any patterns you noticed that could help in similar situations

Write only the summary, no additional commentary:"""

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": summary_prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=32000
            )
            return response.strip()
        except Exception as e:
            print(f"Error generating episode summary: {e}")
            return f"Episode {episode_num} completed in {category} category."
    
    
def save_checkpoint(all_results: List[Dict], category_results: Dict, agent: LLMAgent, 
                   output_dir: str, model: str, feedback_type: str, episode_num: int, seed: Optional[int] = None):
    """Save incremental checkpoint every 5 categories."""
    
    # Create checkpoint filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    completed_categories = len([cat for cat, results in category_results.items() if len(results) > 0])
    checkpoint_file = os.path.join(output_dir, f"checkpoint_categories_{completed_categories:02d}_episode_{episode_num:03d}_{model_safe_name}_{feedback_safe_name}.json")
    
    # Prepare checkpoint data
    checkpoint_data = {
        'experiment': 'Experiment 1: LLM Learning Across Categories (Checkpoint)',
        'timestamp': datetime.now().isoformat(),
        'episode_num': episode_num,
        'model': model,
        'feedback_type': feedback_type,
        'seed': seed,
        'agent_state': {
            'episode_count': agent.episode_count,
            'episode_history': agent.episode_history,
            'episode_summaries': agent.episode_summaries
        },
        'episodes_completed': len(all_results),
        'all_results': all_results,
        'category_results': category_results,
        'summary': {
            'categories_tested': list(category_results.keys()),
            'total_episodes': len(all_results),
            'episodes_by_category': {cat: len(results) for cat, results in category_results.items()},
            'product_counts_by_category': {
                cat: {
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'episodes': len(results)
                } for cat, results in category_results.items()
            }
        }
    }
    
    # Save checkpoint
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    print(f"  Checkpoint saved: {checkpoint_file}")
    return checkpoint_file


def load_checkpoint(checkpoint_file: str) -> Tuple[List[Dict], Dict, Dict]:
    """Load experiment from checkpoint file."""
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded checkpoint from episode {data['episode_num']}")
    print(f"  Categories tested: {data['summary']['categories_tested']}")
    print(f"  Total episodes: {data['summary']['total_episodes']}")
    
    return data['all_results'], data['category_results'], data['agent_state']


def run_experiment1(categories: List[str] = None,
                                   num_categories: int = 5,
                                   episodes_per_category: int = 5,
                                   max_questions: int = 8,
                                   model: str = "gpt-4o",
                                   feedback_type: str = "none",
                                   min_score_threshold: float = 50.0,
                                   output_dir: str = "experiment1_results",
                                   checkpoint_file: str = None,
                                   seed: Optional[int] = None,
                                   context_mode: str = "raw",
                                   prompting_tricks: str = "none"):
    """
    Run Experiment 1 with incremental checkpointing.
    
    Args:
        categories: List of categories to test (None = randomly choose)
        num_categories: Number of categories to randomly select (if categories is None)
        episodes_per_category: Number of episodes per category
        max_questions: Maximum questions per episode
        model: LLM model to use
        feedback_type: Type of feedback to provide ("none", "regret", "persona", "star_rating")
        min_score_threshold: Minimum score threshold for category relevance (default: 50.0)
        output_dir: Directory to save results
        checkpoint_file: Optional checkpoint file to resume from
        seed: Random seed for reproducible category selection and persona selection (None = no seeding)
        context_mode: How to carry context between episodes ("raw", "summary", "none") (default: "raw")
        prompting_tricks: Whether to use enhanced prompting tricks ("none", "all") (default: "none")
    """
    
    print(f"=== Experiment 1: LLM Learning Across Categories (With Checkpoints) ===")
    print(f"Episodes per category: {episodes_per_category}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    # Randomly select a persona based on the seed (consistent across episodes)
    persona_index = random.randint(0, 47000)  
    print(f"Selected persona: {persona_index}")
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Initialize or load from checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        all_results, category_results, agent_state = load_checkpoint(checkpoint_file)
        
        # Recreate agent with saved state
        agent = LLMAgent(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        agent.episode_count = agent_state['episode_count']
        agent.episode_history = agent_state.get('episode_history', [])
        agent.episode_summaries = agent_state.get('episode_summaries', [])
        
        # Calculate starting episode number
        start_episode = len(all_results) + 1
        print(f"Resuming from episode {start_episode}")
        
        # Track which categories are already completed
        completed_categories = set(category_results.keys())
        print(f"Already completed categories: {len(completed_categories)}")
    else:
        print("Starting fresh experiment")
        all_results = []
        category_results = {}
        agent = LLMAgent(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        start_episode = 1
        completed_categories = set()
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    from .core.user_model import UserModel
    
    if feedback_type == "persona":
        # Create persona agent for persona feedback
        persona_agent = UserModel(persona_index)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    from .core.simulate_interaction import list_categories, get_products_by_category
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
                # Convert to the format expected by RecoEnv: (product_id, score) tuples
                cached_scores = [(pid, score) for pid, score in scores]
                return max_score > min_score_threshold, max_score, cached_scores
            return False, 0.0, []
        except Exception as e:
            print(f"  Error checking category {category}: {e}")
            return False, 0.0, []
    
    # Category selection with proper seed-based randomization
    def get_categories_for_seed(available_categories, seed):
        """Get categories in randomized order based on seed."""
        # Set seed for reproducible category ordering
        random.seed(seed)
        categories = available_categories.copy()
        # Shuffle categories based on seed for proper randomization
        random.shuffle(categories)
        random.seed()  # Reset seed after use
        return categories

    # Initialize category selection with proper randomization
    if categories is None:
        # Get all categories in randomized order based on seed
        selected_categories = get_categories_for_seed(available_categories, seed)
        print(f"Categories selected with randomization from seed {seed}")
    else:
        # Use provided categories, filtered by availability
        selected_categories = [cat for cat in categories if cat in available_categories]
    
    # Filter out already completed categories when resuming from checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        remaining_categories = [cat for cat in selected_categories if cat not in completed_categories]
        print(f"Original categories: {len(selected_categories)} total")
        print(f"Already completed: {len(completed_categories)} categories")
        print(f"Remaining categories to test: {len(remaining_categories)}")
        selected_categories = remaining_categories
    else:
        print(f"Categories to test: {len(selected_categories)} total")
    
    used_categories = set()
    
    # Calculate target number of successful episodes (those with regret values)
    target_successful_episodes = num_categories * episodes_per_category
    
    # Calculate total episodes correctly for checkpoint resumption
    if checkpoint_file and os.path.exists(checkpoint_file):
        total_episodes = len(selected_categories) * episodes_per_category
        print(f"Remaining episodes to complete: {total_episodes}")
    else:
        total_episodes = len(selected_categories) * episodes_per_category
        print(f"Total episodes planned: {total_episodes}")
    
    episode_num = start_episode - 1  
    successful_episodes_count = 0  
        
    print(f"\n=== Starting Episode Execution ===")
    print(f"Target: {target_successful_episodes} successful episodes")
    print(f"Available categories: {len(selected_categories)}")
    
    # Simple approach: test categories one by one until we get enough successful episodes
    for category in selected_categories:
        if successful_episodes_count >= target_successful_episodes:
            break
            
        print(f"\n--- Testing Category: {category} ---")
        
        # Check if category is relevant for this persona
        is_relevant, max_score, cached_scores = is_category_relevant_for_persona(category, persona_index, min_score_threshold)
        
        if not is_relevant:
            print(f"  ✗ Category {category}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping")
            continue
            
        print(f"  ✓ Category {category}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        
        used_categories.add(category)
        if category not in category_results:
            category_results[category] = []
        
        for episode in range(episodes_per_category):
            if successful_episodes_count >= target_successful_episodes:
                break
                
            episode_num += 1
            
            # Show progress correctly
            print(f"Episode {episode_num} (Category: {category}) - {successful_episodes_count + 1}/{target_successful_episodes} planned episodes")
            
            import time
            episode_start_time = time.time()
            
            try:
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
                
                while not terminated and not truncated and step_count <= 20:
                    step_start_time = time.time()
                    action = agent.get_action(obs, current_info)
                    action_time = time.time() - step_start_time
                    print(f"[TIMING] Agent action decision: {action_time:.2f}s")
                    
                    step_start_time = time.time()
                    obs, reward, terminated, truncated, info = metrics_wrapper.step(action)
                    step_time = time.time() - step_start_time
                    print(f"[TIMING] Environment step: {step_time:.2f}s")
                    
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
                    'persona_index': persona_index,
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
                
                # Only increment after episode succeeds
                successful_episodes_count += 1
                
                # Log episode timing
                episode_elapsed = time.time() - episode_start_time
                print(f"[TIMING] Episode {episode_num} completed in {episode_elapsed:.2f}s")
                
                # Save checkpoint every 5 successful episodes
                if successful_episodes_count % 5 == 0:
                    save_checkpoint(all_results, category_results, agent, output_dir, model, feedback_type, episode_num, seed)
                
            except Exception as e:
                print(f"  Error in episode {episode_num}: {e}")
                # Save checkpoint even if episode failed (every 5 successful episodes)
                if successful_episodes_count % 5 == 0:
                    save_checkpoint(all_results, category_results, agent, output_dir, model, feedback_type, episode_num, seed)
                continue
    
    print(f"\n=== Results Analysis ===")
    print(f"Target successful episodes: {target_successful_episodes}")
    print(f"Actual successful episodes: {successful_episodes_count}")
    print(f"Categories tested: {len(used_categories)}")
    
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
        if len(results) > 1:
            first_score = results[0]['final_info'].get('chosen_score', 0)
            last_score = results[-1]['final_info'].get('chosen_score', 0)
            improvement = last_score - first_score
            print(f"  {category}: {first_score:.1f} → {last_score:.1f} (Δ{improvement:+.1f})")
    
    # Calculate regret progression with questions for each episode
    episode_regrets = []
    episode_data = []  # List of (regret, questions) tuples
    
    for result in all_results:
        if 'final_info' in result and 'regret' in result['final_info']:
            regret = result['final_info']['regret']
            questions = result.get('steps', 0)
            episode_regrets.append(regret)
            episode_data.append({'regret': regret, 'questions': questions})
    
    if episode_regrets:
        avg_regret = np.mean(episode_regrets)
        regret_trend = "improving" if len(episode_regrets) > 1 and episode_regrets[-1] < episode_regrets[0] else "stable"
        
        print(f"\nRegret Analysis:")
        print(f"  Average Regret: {avg_regret:.1f}")
        print(f"  Trend: {regret_trend}")
        print(f"  Episodes: {len(episode_regrets)}")
    
    # Calculate total questions asked across all episodes
    total_questions_asked = sum(episode.get('steps', 0) for episode in all_results)
    
    # Save final results
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    results_file = os.path.join(output_dir, f"experiment1_results_{model_safe_name}_{feedback_safe_name}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 1: LLM Learning Across Categories',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': {
                    'episode_data': episode_data,
                    'episode_regrets': episode_regrets,
                    'avg_regret': avg_regret if episode_regrets else 0.0,
                    'regret_trend': regret_trend if episode_regrets else "unknown"
                },
                'categories_tested': list(used_categories),
                'total_episodes': len(all_results),
                'successful_episodes': successful_episodes_count,
                'target_successful_episodes': target_successful_episodes,
                'total_questions_asked': total_questions_asked,
                'episodes_by_category': {cat: len(results) for cat, results in category_results.items()},
                'product_counts_by_category': {
                    cat: {
                        'num_products': results[0]['product_info']['num_products'] if results else 0,
                        'episodes': len(results)
                    } for cat, results in category_results.items()
                }
            },
            'config': {
                'persona_index': persona_index,
                'categories': categories,
                'episodes_per_category': episodes_per_category,
                'max_questions': max_questions,
                'model': model,
                'feedback_type': feedback_type,
                'seed': seed
            },
            'agent_episode_history': agent.episode_history,
            'category_results': {
                cat: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'products_with_scores': results[0]['product_info']['products_with_scores'] if results else []
                }
                for cat, results in category_results.items()
            },
            'all_episodes': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    print(f"Checkpoints saved to: {output_dir}/checkpoint_categories_*.json")
    
    return all_results, category_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 1 with checkpointing")
    parser.add_argument("--categories", nargs="+", help="Categories to test")
    parser.add_argument("--num_categories", type=int, default=5, help="Number of categories")
    parser.add_argument("--episodes_per_category", type=int, default=5, help="Episodes per category")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--feedback_type", default="none", help="Feedback type (none, regret, persona, star_rating)")
    parser.add_argument("--min_score_threshold", type=float, default=50.0, help="Min score threshold")
    parser.add_argument("--output_dir", default="experiment1_results", help="Output directory")
    parser.add_argument("--resume_from", help="Checkpoint file to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible category and persona selection")
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw", help="How to carry context between episodes: raw (full episode data), summary (agent-generated summaries), none (no context)")
    parser.add_argument("--prompting_tricks", choices=["none", "all"], default="none", help="Whether to use enhanced prompting tricks: none (standard prompting), all (includes chain-of-thought, ReAct, and think-again prompts)")
    
    args = parser.parse_args()
    
    run_experiment1(
        categories=args.categories,
        num_categories=args.num_categories,
        episodes_per_category=args.episodes_per_category,
        max_questions=args.max_questions,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir,
        checkpoint_file=args.resume_from,
        seed=args.seed,
        context_mode=args.context_mode,
        prompting_tricks=args.prompting_tricks
    )