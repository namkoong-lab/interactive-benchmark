"""
Unified Agent Architecture

This module contains the UnifiedAgent class that replaces all 6 existing agent classes:
- LLMAgent (variable_category)
- LLMAgentExperiment2 (variable_persona) 
- LLMAgentExperiment3 (variable_settings)
- FixedQuestionsAgent (planning_no_strat)
- GreedyFixedQuestionsAgent (planning_greedy)
- DPFixedQuestionsAgent (planning_dp)

The UnifiedAgent handles all experiment types through configuration parameters.
"""

from typing import Dict, List, Tuple, Optional, Any, Literal
import numpy as np
import re
from .llm_providers import chat_completion


class UnifiedAgent:
    """
    Single agent that handles ALL experiment types through configuration.
    
    Replaces:
    - LLMAgent (variable_category)
    - LLMAgentExperiment2 (variable_persona)
    - LLMAgentExperiment3 (variable_settings)
    - FixedQuestionsAgent (planning_no_strat)
    - GreedyFixedQuestionsAgent (planning_greedy)
    - DPFixedQuestionsAgent (planning_dp)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_questions: int = 8,
        context_mode: str = "raw",
        prompting_tricks: str = "none",
        
        # PLANNING CONFIGURATION
        force_all_questions: bool = False,  # True = planning mode (must ask all)
        strategy: str = "none",  # "none", "greedy", "pomdp"
        track_regret_progression: bool = False,  # For planning tracking episodes
        
        # EXPERIMENT CONFIGURATION (for feedback headers)
        vary_persona: bool = True,     # Does persona change across episodes?
        vary_category: bool = True,    # Does category change across episodes?
        
        # LLM PARAMETERS
        temperature: float = 0.7,
        max_tokens: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize unified agent.
        
        Args:
            model: LLM model to use (gpt-4o, claude-3-5-sonnet, etc.)
            max_questions: Maximum questions allowed per episode
            context_mode: How to carry context between episodes
                - "raw": Full episode details (dialog + feedback)
                - "summary": LLM-generated summaries
                - "none": No cross-episode context
            prompting_tricks: Prompting enhancements to apply
                - "none": Standard prompting
                - "all": Chain-of-thought + ReAct + reflection
            force_all_questions: True = planning mode (must ask all questions)
            strategy: Planning strategy ("none", "greedy", "pomdp")
            track_regret_progression: Enable regret tracking for planning experiments
            vary_persona: Does persona change across episodes?
            vary_category: Does category change across episodes?
            temperature: LLM temperature (0.0-2.0)
            max_tokens: Maximum tokens for LLM responses
            verbose: Enable verbose logging
        """
        self.model = model
        self.max_questions = max_questions
        self.context_mode = context_mode
        self.prompting_tricks = prompting_tricks
        
        # Episode tracking
        self.episode_count = 0
        self.episode_history: List[Dict[str, Any]] = []
        self.episode_summaries: List[str] = []
        
        # Current episode state
        self.current_episode_info: Optional[Dict[str, Any]] = None
        self.current_env = None
        self.last_response: Optional[str] = None
        
        # Configuration
        self.force_all_questions = force_all_questions
        self.strategy = strategy
        self.track_regret_progression = track_regret_progression
        self.vary_persona = vary_persona
        self.vary_category = vary_category
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Planning-specific
        self.is_tracking_episode = False
        self.current_question_count = 0
    
    def get_action(self, obs, info):
        """Decide action based on configuration."""
        
        # Get current state - handle missing info gracefully
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
        
        # PLANNING MODE: Must ask all questions
        if self.force_all_questions:
            if len(dialog_history) >= self.max_questions:
                return self._force_recommendation(obs, info, dialog_history, category, num_products)
            else:
                # Planning mode: always ask question
                if self.is_tracking_episode:
                    return self._ask_question_only(obs, info, dialog_history, category, num_products)
                else:
                    return self._llm_decide_action(obs, info, dialog_history, category, num_products)
        
        # VARIABLE MODE: Can stop early
        else:
            if len(dialog_history) >= self.max_questions:
                return self._force_recommendation(obs, info, dialog_history, category, num_products)
            else:
                return self._llm_decide_action(obs, info, dialog_history, category, num_products)
    
    def _ask_question_only(self, obs, info, dialog_history, category, num_products):
        """Ask question without recommendation option (for regret tracking)."""
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category)
        persona, episode_num = self._get_session_info()
        
        # Build strategy-specific prompt
        strategy_section = self._build_strategy_section()
        
        prompt = f"""You are a product recommendation agent.

SESSION: Customer #{persona} | Episode {episode_num} | Category: {category}

=== AVAILABLE PRODUCTS ===
{context}

{feedback_context}

{strategy_section}

=== OUTPUT FORMAT ===
QUESTION: [your question]

=== RULES ===
1. Ask about preferences, not specific products
2. Questions must be consumer-friendly
3. No explanations, just the format

Your question:"""
        
        response = chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if self.verbose:
            print(f"Question-only response: {response}")
        
        return num_products  # Always ask question
    
    def _llm_decide_action(self, obs, info, dialog_history, category, num_products, current_persona=None):
        """Main decision with strategy-specific prompts."""
        
        # Build context (inherited)
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category, current_persona)
        persona, episode_num = self._get_session_info()
        
        # Build strategy-specific prompt
        strategy_section = self._build_strategy_section()
        
        # Unified prompt
        base_prompt = f"""You are a product recommendation agent.

SESSION: Customer #{persona} | Episode {episode_num} | Category: {category}

=== AVAILABLE PRODUCTS ===
{context}

{feedback_context}

{strategy_section}

=== OUTPUT FORMAT ===
QUESTION: [your question]
RECOMMEND: [number 0-{num_products-1}]

=== RULES ===
1. Ask about preferences, not specific products
2. Questions must be consumer-friendly
3. No explanations, just the format

Your choice:"""
        
        # Apply prompting tricks if enabled
        if self.prompting_tricks == "all":
            base_prompt = self._apply_prompting_tricks(base_prompt)
        
        # Call LLM and parse response
        response = chat_completion(
            model=self.model,
            messages=[{"role": "user", "content": base_prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        if self.verbose:
            print(f"LLM response: {response}")
        
        if "QUESTION:" in response:
            return num_products  # Ask question
        elif "RECOMMEND:" in response:
            # Parse product number
            try:
                recommend_line = [line for line in response.split('\n') if 'RECOMMEND:' in line][0]
                product_index = int(recommend_line.split('RECOMMEND:')[1].strip())
                return product_index
            except (ValueError, IndexError):
                if self.verbose:
                    print(f"Failed to parse recommendation, defaulting to question")
                return num_products
        else:
            if self.verbose:
                print(f"No valid action found, defaulting to question")
            return num_products
    
    def _build_strategy_section(self):
        """Build strategy-specific prompt section."""
        if self.strategy == "greedy":
            return """
=== GREEDY STRATEGY ===
INTERNAL REASONING:
1. List all possible products the customer might like
2. Ask the question that ELIMINATES THE MOST products

Your question should maximize immediate information gain.
"""
        
        elif self.strategy == "pomdp":
            return """
=== POMDP PLANNING STRATEGY ===
Think like a planner solving a POMDP:
1. Maintain a belief state over customer preferences
2. Compute expected value of information (EVI) for each question
3. Choose the question with highest EVI

Consider future question value, not just immediate gain.
"""
        
        else:  # strategy == "none"
            return ""
    
    def _build_feedback_context_header(self, category, current_persona=None):
        """Build header based on what varies."""
        
        # Same customer, different categories
        if not self.vary_persona and self.vary_category:
            if self.episode_history:
                persona = self.episode_history[0].get('persona', 'unknown')
                return f"=== WHAT YOU'VE LEARNED ABOUT CUSTOMER #{persona} ==="
            else:
                return "=== WHAT YOU'VE LEARNED ABOUT THIS CUSTOMER ==="
        
        # Different customers, same category
        elif self.vary_persona and not self.vary_category:
            if self.episode_history:
                cat = self.episode_history[0].get('category', 'unknown')
                return f"=== QUESTIONING STRATEGIES FOR {cat.upper()} ==="
            else:
                return "=== QUESTIONING STRATEGIES LEARNED ACROSS CUSTOMERS ==="
        
        # Both vary
        else:
            if current_persona and category:
                return f"=== PREVIOUS EPISODES ===\nCurrent: Customer #{current_persona}, {category}"
            return "=== PREVIOUS EPISODES ==="
    
    def _get_summary_focus_instructions(self):
        """Summary focus based on what varies."""
        
        # Same customer, different categories
        if not self.vary_persona and self.vary_category:
            if self.episode_history:
                persona = self.episode_history[0].get('persona', 'unknown')
                return f"""Focus on Customer #{persona}:
- This customer's preferences across categories
- Budget ranges and decision factors that transfer
- Patterns in how THIS CUSTOMER responds"""
            else:
                return """Focus on this customer:
- Customer's preferences across categories
- Budget ranges and decision factors that transfer
- Patterns in how this customer responds"""
        
        # Different customers, same category
        elif self.vary_persona and not self.vary_category:
            return """Focus on questioning strategies:
- What question types work well for this category
- Patterns across different customers
- Effective question sequences"""
        
        # Both vary
        else:
            return """Focus on general strategies:
- Approaches that work broadly
- Transferable patterns across contexts"""
    
    def set_tracking_episode(self, is_tracking: bool):
        """Enable regret tracking (planning experiments)."""
        self.is_tracking_episode = is_tracking
    
    def update_preferences(self, episode_result):
        """Update episode history (add regret if planning mode)."""
        self.update_episode_history(episode_result)
        
        # Add regret if planning experiment
        if self.force_all_questions and 'final_info' in episode_result:
            regret = episode_result['final_info'].get('regret', 0.0)
            self.episode_history[-1]['regret'] = regret
    
    def _apply_prompting_tricks(self, base_prompt: str) -> str:
        """
        Apply prompting enhancement tricks to base prompt.
        
        Args:
            base_prompt: The base prompt to enhance
            
        Returns:
            Enhanced prompt with structured reasoning scaffolding
        """
        if self.prompting_tricks == "none":
            return base_prompt
        
        elif self.prompting_tricks == "all":
            # Structured reasoning with required completion
            return f"""{base_prompt}

=== REQUIRED REASONING (Complete before responding) ===

STEP 1 - What I know about the customer:
[Write 2-3 specific observations from the conversation]

STEP 2 - What I'm uncertain about:
[List 2 key uncertainties]

STEP 3 - Current best guess:
- If I had to recommend now: Product #___
- My confidence (0-100): ___
- Why I'm unsure: ___

STEP 4 - Information value:
- Most helpful question: "___"
- Why it helps: ___
- Expected products remaining after answer: ___

STEP 5 - Final decision logic:
- If confidence ≥70% → RECOMMEND
- If confidence <70% → ASK the question from Step 4

Complete the analysis above, then provide ONLY your decision:
QUESTION: [your question] OR RECOMMEND: [number]
"""
        else:
            return base_prompt
    
    def _build_episode_data(self, episode_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and structure episode data for history tracking.
        
        Args:
            episode_result: Raw episode result from environment
            
        Returns:
            Structured episode data dictionary
        """
        category = episode_result.get('category', 'unknown')
        feedback = episode_result['final_info'].get('feedback', '')
        full_dialog = episode_result.get('full_dialog', [])
        chosen_product_id = episode_result['final_info'].get('chosen_product_id', None)
        persona_index = episode_result.get('persona_index', None)
        
        # Extract product name
        product_name = "Unknown Product"
        if 'product_info' in episode_result and 'products_with_scores' in episode_result['product_info']:
            for product in episode_result['product_info']['products_with_scores']:
                if product.get('id') == chosen_product_id:
                    product_name = product.get('name', 'Unknown Product')
                    break
        
        return {
            'episode': self.episode_count,
            'category': category,
            'persona': persona_index,
            'dialog': full_dialog,
            'selected_product_id': chosen_product_id,
            'selected_product_name': product_name,
            'feedback': feedback
        }
    
    def _generate_episode_summary(self, episode_data: Dict[str, Any]) -> str:
        """
        Generate LLM-based summary of episode for compact context.
        
        Args:
            episode_data: Structured episode data
            
        Returns:
            Summary string generated by LLM
        """
        episode_num = episode_data.get('episode', 0)
        category = episode_data.get('category', 'unknown')
        persona = episode_data.get('persona', 'unknown')
        dialog = episode_data.get('dialog', [])
        selected_product_id = episode_data.get('selected_product_id', None)
        feedback = episode_data.get('feedback', '')
        
        # Format dialog
        dialog_text = ""
        if dialog:
            for i, (question, answer) in enumerate(dialog):
                dialog_text += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"
        
        # Get experiment-specific focus
        focus_instructions = self._get_summary_focus_instructions()
        
        summary_prompt = f"""You just completed Episode {episode_num} in the {category} category for Customer {persona}.

Episode Details:
{dialog_text}
Selected Product: {selected_product_id}
Customer Feedback: {feedback}

Your task is to provide the context from this episode that you would want a future agent to know.

{focus_instructions}

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
            return f"Episode {episode_num} completed in {category} category for Customer {persona}."
    
    def update_episode_history(self, episode_result: Dict[str, Any]):
        """
        Update episode history after an episode completes.
        
        Args:
            episode_result: Episode result dictionary from environment
        """
        self.episode_count += 1
        
        if 'final_info' not in episode_result:
            return
        
        # Build structured episode data
        episode_data = self._build_episode_data(episode_result)
        self.episode_history.append(episode_data)
        
        # Generate summary if in summary mode
        if self.context_mode == "summary":
            summary = self._generate_episode_summary(episode_data)
            self.episode_summaries.append(summary)
    
    def _get_product_info(self, obs: Dict[str, np.ndarray], info: Dict[str, Any], num_products: int) -> List[Dict]:
        """
        Extract product information from observation.
        
        Args:
            obs: Observation dictionary with product features
            info: Info dictionary with product metadata
            num_products: Number of products in episode
            
        Returns:
            List of product dictionaries with extracted info
        """
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
        """
        Build context string showing available products and conversation.
        
        Args:
            products: List of product dictionaries
            dialog_history: List of (question, answer) tuples
            category: Product category
            
        Returns:
            Formatted context string
        """
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
        
        return f"{product_list}\n{dialog_text}"
    
    def _build_raw_feedback_context(self, episode_history: List[Dict[str, Any]]) -> List[str]:
        """
        Build raw feedback context from episode history.
        
        Args:
            episode_history: List of episode data dictionaries
            
        Returns:
            List of context strings (one per episode)
        """
        context_parts = []
        
        for episode_data in episode_history:
            episode_num = episode_data.get('episode', 0)
            episode_category = episode_data.get('category', 'unknown')
            dialog = episode_data.get('dialog', [])
            selected_product_name = episode_data.get('selected_product_name', 'Unknown Product')
            feedback = episode_data.get('feedback', '')
            
            context_parts.append(f"Episode {episode_num} [{episode_category}]:")
            
            if dialog:
                for i, (question, answer) in enumerate(dialog):
                    context_parts.append(f"  Q{i+1}: {question}")
                    context_parts.append(f"  A{i+1}: {answer}")
            
            context_parts.append(f"  Recommended: {selected_product_name}")
            
            if feedback:
                context_parts.append(f"  Customer Feedback: {feedback}")
            
            context_parts.append("")
        
        return context_parts
    
    def _build_summary_feedback_context(self, episode_history: List[Dict[str, Any]], 
                                       episode_summaries: List[str]) -> List[str]:
        """
        Build summary-based feedback context from episode history.
        
        Args:
            episode_history: List of episode data dictionaries
            episode_summaries: List of LLM-generated summaries
            
        Returns:
            List of context strings (one per episode)
        """
        context_parts = []
        
        for episode_data, summary in zip(episode_history, episode_summaries):
            episode_num = episode_data.get('episode', 0)
            episode_category = episode_data.get('category', 'unknown')
            context_parts.append(f"Episode {episode_num} [{episode_category}]:")
            context_parts.append(f"  {summary}")
            context_parts.append("")
        
        return context_parts
    
    def _get_session_info(self) -> Tuple[str, int]:
        """
        Get current session information (customer ID, episode number).
        
        Returns:
            Tuple of (persona_id, episode_number)
        """
        persona = self.episode_history[0].get('persona', 'unknown') if self.episode_history else 'new'
        episode_num = self.episode_count + 1
        return str(persona), episode_num
    
    def _llm_decide_action(self, obs: Dict[str, np.ndarray], info: Dict[str, Any],
                          dialog_history: List[Tuple[str, str]], category: str, 
                          num_products: int, current_persona: Optional[int] = None) -> int:
        """
        Use LLM to decide whether to ask a question or make a recommendation.
        
        Args:
            obs: Observation from environment
            info: Info dictionary from environment
            dialog_history: List of (question, answer) tuples
            category: Product category
            num_products: Number of products available
            current_persona: Current persona ID (optional, for multi-persona experiments)
            
        Returns:
            Action index (0 to num_products-1 for recommend, num_products for ask)
        """
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category, current_persona)
        
        # Get session info
        persona, episode_num = self._get_session_info()
        
        base_prompt = f"""You are a product recommendation agent. Find the best product for this user while asking the fewest questions.

SESSION: Customer #{persona} | Episode {episode_num} | Category: {category}

=== AVAILABLE PRODUCTS ===
{context}

{feedback_context}

=== OUTPUT FORMAT ===
You must respond with EXACTLY ONE of these formats:

To ask a question:
QUESTION: [your question]

To recommend a product:
RECOMMEND: [number from 0 to {num_products-1}]

Examples:
QUESTION: What is your budget?
RECOMMEND: 5

=== RULES ===
1. Start your response with "QUESTION:" or "RECOMMEND:" - nothing else
2. No explanations, reasoning, or extra text
3. Questions must be short and consumer-friendly
4. Never repeat questions already asked
5. Use previous answers to inform your decision
6. Recommend when you have sufficient information

Your response:"""

        unified_prompt = self._apply_prompting_tricks(base_prompt)

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.4,
                max_tokens=200
            )
            
            self.last_response = response.strip()
            response_text = response.strip()
            
            if "QUESTION:" in response_text:
                return num_products
            elif "RECOMMEND:" in response_text:
                try:
                    rec_part = response_text.split("RECOMMEND:")[-1].strip()
                    numbers = re.findall(r'\d+', rec_part)
                    if numbers:
                        product_index = int(numbers[0])
                        if 0 <= product_index < num_products:
                            return product_index
                    return num_products
                except (ValueError, IndexError):
                    return num_products
            else:
                return num_products
                
        except Exception as e:
            print(f"Error in LLM decision: {e}")
            return num_products
    
    def _force_recommendation(self, obs: Dict[str, np.ndarray], info: Dict[str, Any],
                             dialog_history: List[Tuple[str, str]], category: str, 
                             num_products: int, current_persona: Optional[int] = None) -> int:
        """
        Force agent to make a recommendation (when max questions reached).
        
        Args:
            obs: Observation from environment
            info: Info dictionary from environment
            dialog_history: List of (question, answer) tuples
            category: Product category
            num_products: Number of products available
            current_persona: Current persona ID (optional, for multi-persona experiments)
            
        Returns:
            Product index to recommend (0 to num_products-1)
        """
        products = self._get_product_info(obs, info, num_products)
        context = self._build_llm_context(products, dialog_history, category)
        feedback_context = self._build_feedback_context(category, current_persona)
        
        # Get session info
        persona, episode_num = self._get_session_info()
        
        base_prompt = f"""You are a product recommendation agent for {category} products.

SESSION: Customer #{persona} | Episode {episode_num} | Category: {category}

=== AVAILABLE PRODUCTS ===
{context}

{feedback_context}

=== TASK ===
You have gathered sufficient information. Make your final recommendation now.

=== OUTPUT FORMAT ===
RECOMMEND: [number from 0 to {num_products-1}]

Example:
RECOMMEND: 5

=== RULES ===
1. Choose the product that best matches the user's preferences
2. No questions allowed - you must recommend
3. No explanations - only the format above

Your response:"""

        unified_prompt = self._apply_prompting_tricks(base_prompt)

        try:
            response = chat_completion(
                messages=[{"role": "user", "content": unified_prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=32000
            )
            
            self.last_response = response
            
            if "RECOMMEND:" in response:
                try:
                    product_idx = int(response.split("RECOMMEND:")[-1].strip())
                    if 0 <= product_idx < num_products:
                        return product_idx
                except (ValueError, IndexError):
                    pass
            return 0
            
        except Exception as e:
            print(f"Error in forced recommendation: {e}")
            return 0
    
    def _build_feedback_context(self, category: str, current_persona: Optional[int] = None) -> str:
        """
        Build feedback context from previous episodes.
        
        Args:
            category: Current product category
            current_persona: Current persona ID (for multi-persona experiments)
            
        Returns:
            Formatted feedback context string
        """
        if self.context_mode == "none" or not self.episode_history:
            return ""
        
        # Get header
        header = self._build_feedback_context_header(category, current_persona)
        
        context_parts = [header, ""]
        
        # Use appropriate context building method
        if self.context_mode == "summary":
            context_parts.extend(
                self._build_summary_feedback_context(self.episode_history, self.episode_summaries)
            )
        elif self.context_mode == "raw":
            context_parts.extend(
                self._build_raw_feedback_context(self.episode_history)
            )
        
        return "\n".join(context_parts)
