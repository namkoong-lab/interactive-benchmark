#!/usr/bin/env python3
"""
Experiment 3: LLM Learning Across Both Personas and Categories.
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


class LLMAgentExperiment3:
    """
    LLM-based agent for Experiment 3 that maintains simple context:
    dialog + feedback from previous episodes across different personas and categories.
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
            # Get current persona from environment
            current_persona = getattr(self.current_env, 'persona_index', None) if hasattr(self, 'current_env') and self.current_env else None
        else:
            if self.current_episode_info is None:
                num_products = np.count_nonzero(np.any(obs['product_features'] != 0, axis=1))
                category = "unknown"
                current_persona = None
            else:
                num_products = self.current_episode_info['num_products']
                category = self.current_episode_info['category']
                current_persona = getattr(self.current_env, 'persona_index', None) if hasattr(self, 'current_env') and self.current_env else None
        
        dialog_history = []
        if hasattr(self, 'current_env') and self.current_env and hasattr(self.current_env, 'dialog_history'):
            dialog_history = self.current_env.dialog_history
        
        if len(dialog_history) >= self.max_questions:
            return self._force_recommendation(obs, info, dialog_history, category, num_products, current_persona)
        
        return self._llm_decide_action(obs, info, dialog_history, category, num_products, current_persona)
    
    def _force_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                             dialog_history: List[Tuple[str, str]], category: str, num_products: int, current_persona: int) -> int:
        """Force the agent to make a recommendation by prompting it to do so without revealing the limit."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category, current_persona)
        
        # Build previous episode context
        strategy_context = self._build_feedback_context(category, current_persona)
        
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
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int, current_persona: int) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category, current_persona)
        feedback_context = self._build_feedback_context(category, current_persona)
        
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
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=32000
            )
            
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
    
    def _build_llm_context(self, products: List[Dict], dialog_history: List[Tuple[str, str]], category: str, current_persona: int) -> str:
        """Build context string for LLM decision making."""
        product_list = f"Available {category} products:\n"
        for i, product in enumerate(products):
            description = product.get('description', 'No description available')
            product_list += f"{i}: Product ID {product['id']} - Price: ${product['price']:.2f}\n"
            product_list += f"   Description: {description}\n\n"
        
        dialog_text = "Conversation so far:\n"
        if dialog_history:
            for i, (question, answer) in enumerate(dialog_history):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        else:
            dialog_text += "No questions asked yet.\n"
        
        persona_context = f"Current Persona: {current_persona}\n" if current_persona is not None else ""
        
        return f"{persona_context}{product_list}\n{dialog_text}"
    
    def _build_feedback_context(self, category: str, current_persona: int) -> str:
        """Build context string from previous episodes."""
        if self.context_mode == "none" or not self.episode_history:
            return ""
        
        context_parts = ["Previous episodes:"]
        
        # Choose context mode
        if self.context_mode == "summary" and self.episode_summaries:
            for i, (episode_data, summary) in enumerate(zip(self.episode_history, self.episode_summaries)):
                episode_num = episode_data.get('episode', i + 1)
                episode_category = episode_data.get('category', 'unknown')
                episode_persona = episode_data.get('persona', 'unknown')
                context_parts.append(f"Episode {episode_num}: [{episode_category}/Persona {episode_persona}] Summary:")
                context_parts.append(f"  {summary}")
                context_parts.append("")  # Empty line for readability
        elif self.context_mode == "raw":
            # Show all episodes for context
            for episode_data in self.episode_history:
                episode_num = episode_data.get('episode', 0)
                episode_category = episode_data.get('category', 'unknown')
                episode_persona = episode_data.get('persona', 'unknown')
                dialog = episode_data.get('dialog', [])
                selected_product_id = episode_data.get('selected_product_id', None)
                selected_product_name = episode_data.get('selected_product_name', 'Unknown Product')
                feedback = episode_data.get('feedback', '')
                
                context_parts.append(f"Episode {episode_num}: [{episode_category}/Persona {episode_persona}]")
                
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
            return f"Episode {episode_num} completed in {category} category with Persona {persona}."
    

def save_checkpoint(all_results: List[Dict], persona_category_results: Dict, agent: LLMAgentExperiment3, 
                   output_dir: str, model: str, feedback_type: str, episode_num: int, seed: Optional[int] = None):
    """Save incremental checkpoint every 5 episodes."""
    
    # Create checkpoint filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    completed_episodes = len(all_results)
    checkpoint_file = os.path.join(output_dir, f"checkpoint_episode_{episode_num:03d}_{model_safe_name}_{feedback_safe_name}.json")
    
    # Prepare checkpoint data
    checkpoint_data = {
        'experiment': 'Experiment 3: LLM Learning Across Personas and Categories (Checkpoint)',
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
        'persona_category_results': persona_category_results,
        'summary': {
            'total_episodes': len(all_results),
            'episodes_by_persona_category': {key: len(results) for key, results in persona_category_results.items()},
            'unique_personas': len(set(result['persona_index'] for result in all_results)),
            'unique_categories': len(set(result['category'] for result in all_results))
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
    print(f"  Unique personas: {data['summary']['unique_personas']}")
    print(f"  Unique categories: {data['summary']['unique_categories']}")
    print(f"  Total episodes: {data['summary']['total_episodes']}")
    
    return data['all_results'], data['persona_category_results'], data['agent_state']


def run_experiment3(total_episodes: int = 50,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   feedback_type: str = "none",
                   min_score_threshold: float = 50.0,
                   output_dir: str = "experiment3_results",
                   checkpoint_file: str = None,
                   seed: Optional[int] = None,
                   context_mode: str = "raw",
                   prompting_tricks: str = "none"):
    """
    Run Experiment 3: LLM learning across both personas and categories.
    
    Args:
        total_episodes: Total number of episodes to run
        max_questions: Maximum questions per episode
        model: LLM model to use
        feedback_type: Type of feedback to provide ("none", "regret", "persona", "star_rating")
        min_score_threshold: Minimum score threshold for category relevance (default: 50.0)
        output_dir: Directory to save results
        checkpoint_file: Optional checkpoint file to resume from
        seed: Random seed for reproducible persona and category selection (None = no seeding)
        context_mode: How to carry context between episodes ("raw", "summary", "none") (default: "raw")
        prompting_tricks: Whether to use enhanced prompting tricks ("none", "all") (default: "none")
    """
    
    print(f"=== Experiment 3: LLM Learning Across Personas and Categories ===")
    print(f"Total episodes: {total_episodes}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Initialize or load from checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        all_results, persona_category_results, agent_state = load_checkpoint(checkpoint_file)
        
        # Recreate agent with saved state
        agent = LLMAgentExperiment3(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        agent.episode_count = agent_state['episode_count']
        agent.episode_history = agent_state.get('episode_history', [])
        agent.episode_summaries = agent_state.get('episode_summaries', [])
        
        # Calculate starting episode number
        start_episode = len(all_results) + 1
        print(f"Resuming from episode {start_episode}")
        
    else:
        print("Starting fresh experiment")
        all_results = []
        persona_category_results = {}
        agent = LLMAgentExperiment3(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        start_episode = 1
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    from .core.user_model import UserModel
    
    # For Experiment 3, we'll create feedback systems dynamically per episode
    # since personas vary between episodes
    def create_feedback_system_for_episode(episode_persona_index, feedback_type):
        """Create a feedback system for a specific episode's persona."""
        if feedback_type == "persona":
            persona_agent = UserModel(episode_persona_index)
            return FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
        else:
            return FeedbackSystem(feedback_type=feedback_type)
    
    from .core.simulate_interaction import list_categories, get_products_by_category
    available_categories = list_categories()
    
    # Function to check category relevance for persona
    def is_category_relevant_for_persona(category, persona_index, min_score_threshold):
        """Check if a category is relevant for a specific persona."""
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
            print(f"  Error checking category {category} for persona {persona_index}: {e}")
            return False, 0.0, []
    
    # Generate episode sequence with varying personas and categories
    def generate_episode_sequence(total_episodes, seed):
        """Generate a deterministic sequence of (persona_index, category) pairs."""
        if seed is not None:
            random.seed(seed)
        
        episodes = []
        for episode_num in range(1, total_episodes + 1):
            # Generate persona and category for this episode
            persona_index = random.randint(0, 47000)
            
            # Shuffle categories for this episode
            shuffled_categories = available_categories.copy()
            random.shuffle(shuffled_categories)
            
            # Find first relevant category for this persona
            category = None
            for cat in shuffled_categories:
                is_relevant, max_score, cached_scores = is_category_relevant_for_persona(cat, persona_index, min_score_threshold)
                if is_relevant:
                    category = cat
                    break
            
            if category is None:
                # Fallback to first category if no relevant category found
                category = available_categories[0]
                print(f"  Episode {episode_num}: No relevant category found for persona {persona_index}, using {category}")
            
            episodes.append((episode_num, persona_index, category))
        
        if seed is not None:
            random.seed()  # Reset seed
        
        return episodes
    
    # Get available categories and shuffle them
    available_categories = list_categories()
    if seed is not None:
        random.seed(seed)
        shuffled_categories = available_categories.copy()
        random.shuffle(shuffled_categories)
        random.seed()  # Reset seed
    else:
        shuffled_categories = available_categories
    
    # Run episodes directly instead of pre-generating sequence
    successful_episodes_count = 0
    episode_num = start_episode
    
    while successful_episodes_count < total_episodes:
        # Generate persona and category for this episode
        if seed is not None:
            random.seed(seed + episode_num)  # Use episode number for variation
        
        # Select persona for this episode
        persona_index = random.randint(0, 47000)  # Random persona
        
        # Find a random relevant category for this persona
        category = None
        max_attempts = len(shuffled_categories)
        attempts = 0
        
        while category is None and attempts < max_attempts:
            # Pick a random category from the shuffled list
            category_index = random.randint(0, len(shuffled_categories) - 1)
            test_category = shuffled_categories[category_index]
            
            # Check if this category is relevant for the persona
            is_relevant, max_score, cached_scores = is_category_relevant_for_persona(test_category, persona_index, min_score_threshold)
            if is_relevant:
                category = test_category
            else:
                # Remove this category from consideration for this episode
                shuffled_categories.pop(category_index)
            
            attempts += 1
        
        if category is None:
            # Fallback to first category if no relevant category found
            category = available_categories[0]
            print(f"  Episode {episode_num}: No relevant category found for persona {persona_index}, using {category}")
        
        print(f"\nEpisode {episode_num}/{total_episodes} (Persona: {persona_index}, Category: {category})")
        
        try:
            # We already found a relevant category above, so proceed directly
            print(f"  ✓ Category {category}: Proceeding with episode")
            
            # Create environment for this episode with episode-specific feedback system
            episode_feedback_system = create_feedback_system_for_episode(persona_index, feedback_type)
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=max_questions,
                categories=[category],  
                agent=agent,
                feedback_system=episode_feedback_system,
                cached_scores=cached_scores
            )
            
            metrics_wrapper = MetricsWrapper(env, 
                                           output_path=os.path.join(output_dir, f"episode_{episode_num}.jsonl"))
            
            obs, initial_info = metrics_wrapper.reset()
            
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
            
            # Only count episodes that successfully made a recommendation
            if info.get('action_type') == 'recommend' and 'chosen_score' in info:
                episode_result = {
                    'episode': episode_num,
                    'category': category,
                    'persona_index': persona_index,
                    'steps': step_count,
                    'terminated': terminated,
                    'truncated': truncated,
                    'final_info': info,
                    'full_dialog': full_dialog,
                    'product_info': product_info
                }
                
                all_results.append(episode_result)
                
                # Track by persona-category combination
                key = f"{persona_index}_{category}"
                if key not in persona_category_results:
                    persona_category_results[key] = []
                persona_category_results[key].append(episode_result)
                
                print(f"  Episode {episode_num}: Successfully completed (Score: {info.get('chosen_score', 0):.1f})")
                
                # Only increment counters and update agent for successful episodes
                agent.update_preferences(episode_result)
                successful_episodes_count += 1
            else:
                print(f"  Episode {episode_num}: Skipped - No recommendation made or missing score data")
            
            metrics_wrapper.close()
            
            # Move to next episode
            episode_num += 1
            
            # Save checkpoint every 5 episodes
            if episode_num % 5 == 0:
                save_checkpoint(all_results, persona_category_results, agent, output_dir, model, feedback_type, episode_num, seed)
            
        except Exception as e:
            print(f"  Error in episode {episode_num}: {e}")
            episode_num += 1
            # Save checkpoint even if episode failed
            if episode_num % 5 == 0:
                save_checkpoint(all_results, persona_category_results, agent, output_dir, model, feedback_type, episode_num, seed)
            continue
    
    print(f"\n=== Results Analysis ===")
    print(f"Total episodes completed: {len(all_results)}")
    print(f"Unique personas: {len(set(result['persona_index'] for result in all_results))}")
    print(f"Unique categories: {len(set(result['category'] for result in all_results))}")
    print(f"Unique persona-category combinations: {len(persona_category_results)}")
    
    # Calculate regret progression with questions for each episode
    episode_regrets = []
    episode_data = []  # List of (regret, questions) tuples
    
    for result in all_results:
        if 'final_info' in result and 'regret' in result['final_info']:
            regret = result['final_info']['regret']
            questions = result.get('steps', 0)
            episode_regrets.append(regret)
            episode_data.append({'regret': regret, 'questions': questions})
    
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
    results_file = os.path.join(output_dir, f"experiment3_results_{model_safe_name}_{feedback_safe_name}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 3: LLM Learning Across Personas and Categories',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': {
                    'episode_data': episode_data,
                    'episode_regrets': episode_regrets,
                    'avg_regret': avg_regret if episode_regrets else 0.0,
                    'regret_trend': regret_trend if episode_regrets else "unknown"
                },
                'total_episodes': len(all_results),
                'unique_personas': len(set(result['persona_index'] for result in all_results)),
                'unique_categories': len(set(result['category'] for result in all_results)),
                'unique_persona_category_combinations': len(persona_category_results),
                'total_questions_asked': total_questions_asked,
                'episodes_by_persona_category': {key: len(results) for key, results in persona_category_results.items()}
            },
            'config': {
                'total_episodes': total_episodes,
                'max_questions': max_questions,
                'model': model,
                'feedback_type': feedback_type,
                'min_score_threshold': min_score_threshold,
                'seed': seed,
                'context_mode': context_mode,
                'prompting_tricks': prompting_tricks
            },
            'agent_episode_history': agent.episode_history,
            'persona_category_results': {
                key: {
                    'persona': int(key.split('_')[0]),
                    'category': key.split('_', 1)[1],
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0
                }
                for key, results in persona_category_results.items()
            },
            'all_episodes': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    print(f"Checkpoints saved to: {output_dir}/checkpoint_episode_*.json")
    
    return all_results, persona_category_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 3 with checkpointing")
    parser.add_argument("--total_episodes", type=int, default=50, help="Total number of episodes")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", default="gpt-4o", help="LLM model")
    parser.add_argument("--feedback_type", default="none", help="Feedback type (none, regret, persona, star_rating)")
    parser.add_argument("--min_score_threshold", type=float, default=50.0, help="Min score threshold")
    parser.add_argument("--output_dir", default="experiment3_results", help="Output directory")
    parser.add_argument("--resume_from", help="Checkpoint file to resume from")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible persona and category selection")
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw", help="How to carry context between episodes: raw (full episode data), summary (agent-generated summaries), none (no context)")
    parser.add_argument("--prompting_tricks", choices=["none", "all"], default="none", help="Whether to use enhanced prompting tricks: none (standard prompting), all (includes chain-of-thought, ReAct, and think-again prompts)")
    
    args = parser.parse_args()
    
    run_experiment3(
        total_episodes=args.total_episodes,
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
