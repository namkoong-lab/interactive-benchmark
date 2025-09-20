#!/usr/bin/env python3
"""
Experiment 2: Cross-User Learning with Same Category
"""

import gymnasium as gym
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import argparse
from datetime import datetime
import random
from .envs.reco_env import RecoEnv
from .core.llm_client import chat_completion
from .wrappers.metrics_wrapper import MetricsWrapper


class LLMAgentExperiment2:
    """
    LLM-based agent for Experiment 2 that maintains simple context:
    dialog + feedback from previous episodes across different personas.
    """
    
    def __init__(self, model: str = "gpt-4o", max_questions: int = 8, context_mode: str = "raw", prompting_tricks: str = "none"):
        self.model = model
        self.max_questions = max_questions
        self.context_mode = context_mode  # Options: "raw", "summary", "none"
        self.prompting_tricks = prompting_tricks  # Options: "none", "all"
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
        
        # Force recommendation if max_questions have been asked
        if len(dialog_history) >= self.max_questions:
            # Force the agent to recommend a product by calling LLM with a prompt that encourages recommendation
            return self._force_recommendation(obs, info, dialog_history, category, num_products, current_persona)
        
        return self._llm_decide_action(obs, info, dialog_history, category, num_products, current_persona)
    
    def _force_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], 
                             dialog_history: List[Tuple[str, str]], category: str, num_products: int, current_persona: int = None) -> int:
        """Force the agent to make a recommendation by prompting it to do so without revealing the limit."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        # Add previous episode context
        strategy_context = self._build_episode_context()
        
        base_prompt = f"""You are a product recommendation agent learning optimal questioning strategies for {category} products.

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
                          dialog_history: List[Tuple[str, str]], category: str, num_products: int, current_persona: int = None) -> int:
        """Use LLM to decide whether to ask a question or make a recommendation."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        
        # Add previous episode context
        strategy_context = self._build_episode_context()
        
        base_prompt = f"""You are a product recommendation agent learning optimal questioning strategies for {category} products.

Context:
{context}

{strategy_context}

Task:
Based on the conversation so far and learned questioning patterns, either:
- Ask one short, consumer-friendly question to clarify user preferences, or
- If sufficiently confident, recommend one product by index. 

Output format (MUST be exactly one line, no extra text):
- To ask: QUESTION: <your question>
- To recommend: RECOMMEND: <number 0-{num_products-1}>

Rules:
- Do not include explanations, reasoning, bullets, or multiple questions
- Avoid jargon; use everyday language a shopper understands
- Keep questions specific and helpful for {category} products
- Consider what types of questions have worked well for this category before
- No meta commentary, only the question or recommendation
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
    
    def _build_episode_context(self) -> str:
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
    
    def update_strategies(self, episode_result: Dict[str, Any]):
        """Store episode context: dialog + selected product + feedback."""
        self.episode_count += 1
        
        if 'final_info' in episode_result:
            category = episode_result.get('category', 'unknown')
            persona_index = episode_result.get('persona_index', None)
            feedback = episode_result['final_info'].get('feedback', '')
            full_dialog = episode_result.get('full_dialog', [])
            chosen_product_id = episode_result['final_info'].get('chosen_product_id', None)
            
            product_name = "Unknown Product"
            if 'product_info' in episode_result and 'products_with_scores' in episode_result['product_info']:
                for product in episode_result['product_info']['products_with_scores']:
                    if product.get('id') == chosen_product_id:
                        product_name = product.get('name', 'Unknown Product')
                        break
            
            # Store episode context with selected product
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
            return f"Episode {episode_num} completed in {category} category for Persona {persona}."
    


def save_checkpoint(all_results: List[Dict], persona_results: Dict, agent: LLMAgentExperiment2, 
                   output_dir: str, model: str, feedback_type: str, episode_num: int, seed: Optional[int] = None):
    """Save incremental checkpoint every 5 personas."""
    
    # Create checkpoint filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace(" ", "_")
    completed_personas = len([pid for pid, results in persona_results.items() if len(results) > 0])
    checkpoint_file = os.path.join(output_dir, f"checkpoint_personas_{completed_personas:02d}_episode_{episode_num:03d}_{model_safe_name}_{feedback_safe_name}.json")
    
    # Prepare checkpoint data
    checkpoint_data = {
        'experiment': 'Experiment 2: LLM Learning Questioning Strategies Across Users (Checkpoint)',
        'timestamp': datetime.now().isoformat(),
        'episode_num': episode_num,
        'model': model,
        'feedback_type': feedback_type,
        'seed': seed,
        'agent_state': {
            'episode_count': agent.episode_count,
            'episode_history': agent.episode_history
        },
        'episodes_completed': len(all_results),
        'all_results': all_results,
        'persona_results': persona_results,
        'summary': {
            'personas_tested': list(persona_results.keys()),
            'total_episodes': len(all_results),
            'episodes_by_persona': {pid: len(results) for pid, results in persona_results.items()},
            'product_counts_by_persona': {
                pid: {
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'episodes': len(results)
                } for pid, results in persona_results.items()
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
    print(f"  Personas tested: {data['summary']['personas_tested']}")
    print(f"  Total episodes: {data['summary']['total_episodes']}")
    
    return data['all_results'], data['persona_results'], data['agent_state']


def run_experiment2(persona_indices: List[int] = None,
                   num_personas: int = 10,
                   episodes_per_persona: int = 3,
                   max_questions: int = 8,
                   model: str = "gpt-4o",
                   feedback_type: str = "none",
                   min_score_threshold: float = 50.0,
                   output_dir: str = "experiment2_results",
                   seed: Optional[int] = None,
                   checkpoint_file: str = None,
                   context_mode: str = "raw",
                   prompting_tricks: str = "none"):
    """
    Run Experiment 2: LLM learning questioning strategies across users in same category.
    
    Args:
        persona_indices: List of persona indices to use (None = randomly choose)
        num_personas: Number of personas to randomly select (if persona_indices is None)
        episodes_per_persona: Number of episodes per persona
        max_questions: Maximum questions per episode
        model: LLM model to use
        feedback_type: Type of feedback to provide ("none", "regret", "persona", "star_rating")
        min_score_threshold: Minimum score threshold for category relevance (default: 50.0)
        output_dir: Directory to save results
        seed: Random seed for reproducible category and persona selection (None = no seeding)
        checkpoint_file: Optional checkpoint file to resume from
        context_mode: How to carry context between episodes ("raw", "summary", "none") (default: "raw")
        prompting_tricks: Whether to use enhanced prompting tricks ("none", "all") (default: "none")
    """
    
    print(f"=== Experiment 2: LLM Learning Questioning Strategies Across Users ===")
    print(f"Episodes per persona: {episodes_per_persona}")
    print(f"Max questions: {max_questions}, Model: {model}, Feedback: {feedback_type}")
    if seed is not None:
        print(f"Random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
    
    # Deterministically select a category based on the seed
    from .core.simulate_interaction import list_categories
    available_categories = list_categories()
    
    # Use seed to deterministically select category (same seed = same category)
    random.seed(seed)
    category = random.choice(available_categories)
    random.seed()  # Reset seed after use
    print(f"Selected category (deterministic from seed {seed}): {category}")
    
    os.makedirs(output_dir, exist_ok=True)
    gym.register("RecoEnv-v0", entry_point=RecoEnv)
    
    # Initialize or load from checkpoint
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Resuming from checkpoint: {checkpoint_file}")
        all_results, persona_results, agent_state = load_checkpoint(checkpoint_file)
        
        # Recreate agent with saved state
        agent = LLMAgentExperiment2(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        agent.episode_count = agent_state['episode_count']
        agent.episode_history = agent_state.get('episode_history', [])
        
        # Calculate starting episode number
        start_episode = len(all_results) + 1
        print(f"Resuming from episode {start_episode}")
    else:
        print("Starting fresh experiment")
        all_results = []
        persona_results = {}
        agent = LLMAgentExperiment2(model=model, max_questions=max_questions, context_mode=context_mode, prompting_tricks=prompting_tricks)
        start_episode = 1
    
    # Create feedback system
    from .core.feedback_system import FeedbackSystem
    from .core.simulate_interaction import list_categories, get_products_by_category
    
    # Check if category is relevant for a specific persona (similar to Experiment 1)
    def check_category_relevance_with_persona(category, persona_index, min_score_threshold):
        """Check if a category is relevant for a specific persona. Returns (is_relevant, cached_scores)."""
        from .core.user_model import UserModel
        try:
            products = get_products_by_category(category)
            if not products:
                print(f"Category '{category}' has no products")
                return False, []
                
            user_model = UserModel(persona_index)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                print(f"Checking category relevance for '{category}' with persona {persona_index}...")
                print(f"Category '{category}': Max score {max_score:.1f} > {min_score_threshold}, proceeding")
                return max_score > min_score_threshold, scores
            return False, []
        except Exception as e:
            print(f"Error checking category {category} for persona {persona_index}: {e}")
            return False, []

    
    # NEW: Select exactly num_personas that pass the relevance filter
    def select_relevant_personas(available_personas, num_personas, category, min_score_threshold):
        """Select exactly num_personas that pass the relevance filter."""
        relevant_personas = []
        tested_personas = set()
        
        # Use personas in deterministic order (no shuffling for reproducibility)
        for persona_idx in available_personas:
            if len(relevant_personas) >= num_personas:
                break
                
            if persona_idx in tested_personas:
                continue
                
            tested_personas.add(persona_idx)
            
            # Check if this persona finds the category relevant
            is_relevant, max_score = check_persona_category_relevance(persona_idx, category, min_score_threshold)
            if is_relevant:
                relevant_personas.append(persona_idx)
                print(f"  Persona {persona_idx}: Max score {max_score:.1f} > {min_score_threshold}, selected")
            else:
                print(f"  Persona {persona_idx}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipped")
        
        if len(relevant_personas) < num_personas:
            print(f"Warning: Only found {len(relevant_personas)} relevant personas out of {num_personas} requested")
        
        return relevant_personas
    
    # Select personas first
    if persona_indices is None:
        # Use the new function to select exactly num_personas relevant personas
        available_personas = list(range(0, 1000))
        persona_indices = select_relevant_personas(available_personas, num_personas, category, min_score_threshold)
    else:
        # Use provided personas, but still check relevance
        print(f"Using provided personas: {persona_indices}")
        # Note: We'll still check relevance for each persona individually during testing
    
    # Check category relevance using the first persona that will actually be used
    first_persona = persona_indices[0] if persona_indices else None
    if first_persona is not None:
        sample_relevance, cached_scores_from_check = check_category_relevance_with_persona(category, first_persona, min_score_threshold)
        if not sample_relevance:
            print(f"Note: Category '{category}' may not be relevant to the first persona.")
            print("Individual personas will be checked and skipped if irrelevant.")
    
    print(f"Selected personas: {persona_indices}")
    
    from .core.user_model import UserModel
    if feedback_type == "persona":
        # For persona feedback, we'll use the first persona's agent
        first_persona = persona_indices[0]
        persona_agent = UserModel(first_persona)
        feedback_system = FeedbackSystem(feedback_type=feedback_type, persona_agent=persona_agent)
    else:
        feedback_system = FeedbackSystem(feedback_type=feedback_type)
    
    # Initialize results if not loaded from checkpoint
    if not checkpoint_file or not os.path.exists(checkpoint_file):
        all_results = []
        persona_results = {pid: [] for pid in persona_indices}
    
    # Calculate target number of successful episodes (those with regret values)
    target_successful_episodes = num_personas * episodes_per_persona
    
    total_episodes = len(persona_indices) * episodes_per_persona
    episode_num = 0
    successful_episodes_count = 0  # Count only episodes that actually run (have regret values)
    
    # Dynamic persona filtering - check relevance as we encounter personas
    def check_persona_category_relevance(persona_idx, category, min_score_threshold):
        """Check if a specific persona finds the category relevant."""
        from .core.user_model import UserModel
        try:
            products = get_products_by_category(category)
            if not products:
                return False, 0.0
                
            user_model = UserModel(persona_idx)
            scores = user_model.score_products(category, products)
            if scores:
                max_score = max(score for _, score in scores)
                return max_score > min_score_threshold, max_score
            return False, 0.0
        except Exception as e:
            print(f"  Error checking persona relevance: {e}")
            return False, 0.0
    
    # Test the pre-selected relevant personas
    for persona_index_idx, persona_index in enumerate(persona_indices):
        print(f"\n--- Testing Persona: {persona_index} ---")
        
        # For provided personas, still check relevance (for fresh experiments)
        if persona_indices is not None and not checkpoint_file:
            is_relevant, max_score = check_persona_category_relevance(persona_index, category, min_score_threshold)
            if not is_relevant:
                print(f"  Persona {persona_index}: Max score {max_score:.1f} ≤ {min_score_threshold}, skipping persona")
                continue
            print(f"  Persona {persona_index}: Max score {max_score:.1f} > {min_score_threshold}, proceeding")
        else:
            # For pre-selected personas or checkpoint resumption, assume they're relevant
            print(f"  Persona {persona_index}: Pre-selected as relevant, proceeding")
        
        for episode in range(episodes_per_persona):
            if successful_episodes_count >= target_successful_episodes:
                break
                
            episode_num += 1
            
            # Show progress correctly
            print(f"Episode {episode_num} (Persona: {persona_index}) - {successful_episodes_count + 1}/{target_successful_episodes} planned episodes")
            
            env = RecoEnv(
                persona_index=persona_index,
                max_questions=max_questions,
                categories=[category],  # Same category for all episodes
                agent=agent,
                feedback_system=feedback_system,
                cached_scores=cached_scores_from_check if persona_index == first_persona else None
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
                    'episode_in_persona': episode + 1,
                    'steps': step_count,
                    'terminated': terminated,
                    'truncated': truncated,
                    'final_info': info,
                    'full_dialog': full_dialog,
                    'product_info': product_info
                }
                
                all_results.append(episode_result)
                persona_results[persona_index].append(episode_result)
                agent.update_strategies(episode_result)
                
                print(f"  Episode {episode_num}: Successfully completed (Score: {info.get('chosen_score', 0):.1f})")
                
                # Increment successful episodes count after episode completes
                successful_episodes_count += 1
            else:
                print(f"  Episode {episode_num}: Skipped - No recommendation made or missing score data")
            
            metrics_wrapper.close()
            
            # Save checkpoint every 5 successful episodes
            if successful_episodes_count % 5 == 0:
                save_checkpoint(all_results, persona_results, agent, output_dir, model, feedback_type, episode_num, seed)
        
        # Move to next persona (automatic with the for loop)
    
    print(f"\n=== Results Analysis ===")
    print(f"Target successful episodes: {target_successful_episodes}")
    print(f"Actual successful episodes: {successful_episodes_count}")
    print(f"Personas tested: {len([pid for pid, results in persona_results.items() if results])}")
    
    print("\nPerformance by Persona:")
    for persona_index, results in persona_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        top1_rates = [r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]
        
        if scores:
            avg_score = np.mean(scores)
            top1_rate = np.mean(top1_rates)
            print(f"  Persona {persona_index}: Avg Score: {avg_score:.1f}, Top1 Rate: {top1_rate:.1%}, Episodes: {len(scores)}")
    
    print("\nLearning Progression (Questioning Strategy Effectiveness):")
    for persona_index, results in persona_results.items():
        scores = [r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]
        if len(scores) >= 2:
            first_half = np.mean(scores[:len(scores)//2])
            second_half = np.mean(scores[len(scores)//2:])
            improvement = second_half - first_half
            print(f"  Persona {persona_index}: {first_half:.1f} → {second_half:.1f} (Δ{improvement:+.1f})")
    
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
    
    # Create episode data combining regret and questions for each episode
    episode_data = []
    for i in range(len(episode_regrets)):
        episode_data.append({
            'regret': episode_regrets[i],
            'questions': episode_questions[i] if i < len(episode_questions) else 0
        })
    
    regret_progression = {
        'episode_data': episode_data,
        'episode_regrets': episode_regrets,
        'avg_regret': np.mean(episode_regrets),
        'regret_trend': 'improving' if len(episode_regrets) > 1 and episode_regrets[-1] < episode_regrets[0] else 'stable'
    }
    
    questions_progression = {
        'episode_questions': episode_questions,
        'avg_questions': np.mean(episode_questions),
        'total_questions': sum(episode_questions)
    }
    
    # Persona information
    persona_info = {}
    for pid, results in persona_results.items():
        if results:
            persona_info[pid] = {
                'num_products': results[0]['product_info']['num_products'],
                'episodes': len(results)
            }
    
    # Create model-specific filename
    model_safe_name = model.replace("/", "_").replace(":", "_")
    feedback_safe_name = feedback_type.replace("/", "_").replace(":", "_")
    results_file = os.path.join(output_dir, f"experiment2_results_{model_safe_name}_{feedback_safe_name}.json")
    with open(results_file, 'w') as f:
        json.dump({
            'experiment': 'Experiment 2: LLM Learning Questioning Strategies Across Users',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'regret_progression': regret_progression,
                'questions_progression': questions_progression,
                'persona_info': persona_info,
                'agent_episode_history': agent.episode_history,
                'overall_performance': {
                    'avg_score': np.mean(episode_scores) if episode_scores else 0,
                    'total_episodes': len(all_results),
                    'successful_episodes': successful_episodes_count,
                    'target_successful_episodes': target_successful_episodes,
                    'total_questions_asked': sum(episode_questions) if episode_questions else 0
                }
            },
            'config': {
                'category': category,
                'persona_indices': persona_indices,
                'episodes_per_persona': episodes_per_persona,
                'max_questions': max_questions,
                'model': model,
                'feedback_type': feedback_type,
                'seed': seed
            },
            'results': all_results,
            'persona_summary': {
                pid: {
                    'avg_score': np.mean([r['final_info'].get('chosen_score', 0) for r in results if 'chosen_score' in r['final_info']]),
                    'top1_rate': np.mean([r['final_info'].get('top1', False) for r in results if 'top1' in r['final_info']]),
                    'episode_count': len(results),
                    'num_products': results[0]['product_info']['num_products'] if results else 0,
                    'products_with_scores': results[0]['product_info']['products_with_scores'] if results else []
                }
                for pid, results in persona_results.items()
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Individual episode metrics saved to: {output_dir}/episode_*.jsonl")
    print(f"Checkpoints saved to: {output_dir}/checkpoint_personas_*.json")
    
    return all_results, persona_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 2: LLM Learning Questioning Strategies Across Users")
    parser.add_argument("--persona_indices", nargs="+", type=int, default=None, help="Persona indices to use")
    parser.add_argument("--num_personas", type=int, default=10, help="Number of personas to randomly select")
    parser.add_argument("--episodes_per_persona", type=int, default=3, help="Episodes per persona")
    parser.add_argument("--max_questions", type=int, default=8, help="Max questions per episode")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    parser.add_argument("--feedback_type", type=str, default="none", choices=["none", "regret", "persona", "star_rating"], help="Type of feedback to provide")
    parser.add_argument("--min_score_threshold", type=float, default=50.0, help="Minimum score threshold for category relevance")
    parser.add_argument("--output_dir", type=str, default="experiment2_results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible persona selection")
    parser.add_argument("--resume_from", help="Checkpoint file to resume from")
    parser.add_argument("--context_mode", choices=["raw", "summary", "none"], default="raw", help="How to carry context between episodes: raw (full episode data), summary (agent-generated summaries), none (no context)")
    parser.add_argument("--prompting_tricks", choices=["none", "all"], default="none", help="Whether to use enhanced prompting tricks: none (standard prompting), all (includes chain-of-thought, ReAct, and think-again prompts)")
    
    args = parser.parse_args()
    
    run_experiment2(
        persona_indices=args.persona_indices,
        num_personas=args.num_personas,
        episodes_per_persona=args.episodes_per_persona,
        max_questions=args.max_questions,
        model=args.model,
        feedback_type=args.feedback_type,
        min_score_threshold=args.min_score_threshold,
        output_dir=args.output_dir,
        seed=args.seed,
        checkpoint_file=args.resume_from,
        context_mode=args.context_mode,
        prompting_tricks=args.prompting_tricks
    )