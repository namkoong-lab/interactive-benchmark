#!/usr/bin/env python3
"""
Unified configuration for all experiment types.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal, Dict, Any
import random
import yaml
import json


@dataclass
class ExperimentConfig:
    """
    Unified configuration for all experiment types.
    
    Base Experiment Types:
    - variable_category: Same customer, different categories
    - variable_persona: Different customers, same category
    - variable_settings: Both vary
    
    Planning Mode (Optional):
    Can be added to any base experiment type to track regret progression.
    Every Nth episode (planning_interval) runs in planning mode:
    - Agent asks all max_questions questions
    - After each question, agent recommends (hidden from agent)
    - Regret is tracked after each question
    - Only final recommendation gets feedback
    
    Planning strategies:
    - none: No planning (all episodes are regular)
    - no_strat: Planning mode without strategy hints
    - greedy: Planning with greedy information gain prompts
    - pomdp: Planning with POMDP-style forward-thinking prompts
    """
    
    # === EXPERIMENT TYPE ===
    experiment_type: Literal[
        "variable_category",      # Same customer, different categories
        "variable_persona",        # Different customers, same category
        "variable_settings"        # Both vary
    ]
    
    # === MODEL SETTINGS ===
    model: str = "gpt-4o"
    
    # === EXPERIMENT PARAMETERS ===
    max_questions: int = 8
    
    # === LEARNING SETTINGS ===
    context_mode: Literal["raw", "summary", "none"] = "raw"
    prompting_tricks: Literal["none", "all"] = "none"
    feedback_type: Literal["none", "regret", "persona", "star_rating"] = "none"
    
    # === SEED SETTINGS ===
    seeds: Optional[List[int]] = None  # Optional list of seeds (if provided, first total_trajectories are used)
    
    # === DATA SETTINGS ===
    categories: Optional[List[List[str]]] = None  # List of lists: [[cats for traj1], [cats for traj2], ...]
    persona_indices: Optional[List[List[int]]] = None  # List of lists: [[personas for traj1], [personas for traj2], ...]
    min_score_threshold: float = 60.0
    max_products_per_category: int = 100  
    
    # === OUTPUT ===
    output_dir: str = "experiment_results"
    experiment_name: Optional[str] = None
    checkpoint_file: Optional[str] = None
    debug_mode: bool = False  # If True, show detailed output; if False, simplified output
    
    # === EXPERIMENT-SPECIFIC PARAMETERS ===
    # Trajectory-based parameters (replaces episodes_per_category/episodes_per_persona/total_episodes)
    episodes_per_trajectory: int = 5  # Episodes per trajectory (continuous context/feedback)
    total_trajectories: int = 10      # Number of trajectories (agent runs)
    
    # === PLANNING MODE PARAMETERS ===
    # Can be applied to any experiment type to track regret progression
    planning_mode: Literal["none", "no_strat", "greedy", "pomdp"] = "none"
    planning_interval: int = 5  # Run planning episode every N episodes (e.g., 5 = episodes 5, 10, 15, ...)
                                 # Only applies when planning_mode != "none"
    
    def validate(self):
        """
        Validate all configuration parameters before running experiment.
        Raises ValueError with clear message if any parameter is invalid.
        """
        # Validate experiment_type
        valid_experiment_types = ["variable_category", "variable_persona", "variable_settings"]
        if self.experiment_type not in valid_experiment_types:
            raise ValueError(
                f"Invalid experiment_type: '{self.experiment_type}'. "
                f"Must be one of {valid_experiment_types}"
            )
        
        # Validate planning_mode
        valid_planning_modes = ["none", "no_strat", "greedy", "pomdp"]
        if self.planning_mode not in valid_planning_modes:
            raise ValueError(
                f"Invalid planning_mode: '{self.planning_mode}'. "
                f"Must be one of {valid_planning_modes}"
            )
        
        # Validate feedback_type
        valid_feedback_types = ["none", "regret", "persona", "star_rating"]
        if self.feedback_type not in valid_feedback_types:
            raise ValueError(
                f"Invalid feedback_type: '{self.feedback_type}'. "
                f"Must be one of {valid_feedback_types}\n"
                f"Note: 'no_strat', 'greedy', 'pomdp' are planning_mode values, not feedback_type!"
            )
        
        # Validate context_mode
        valid_context_modes = ["raw", "summary", "none"]
        if self.context_mode not in valid_context_modes:
            raise ValueError(
                f"Invalid context_mode: '{self.context_mode}'. "
                f"Must be one of {valid_context_modes}"
            )
        
        # Validate prompting_tricks
        valid_prompting_tricks = ["none", "all"]
        if self.prompting_tricks not in valid_prompting_tricks:
            raise ValueError(
                f"Invalid prompting_tricks: '{self.prompting_tricks}'. "
                f"Must be one of {valid_prompting_tricks}"
            )
        
        # Validate planning_interval
        if self.planning_mode != "none":
            if self.planning_interval < 1:
                raise ValueError(
                    f"planning_interval must be >= 1, got {self.planning_interval}"
                )
            if self.planning_interval > self.episodes_per_trajectory:
                raise ValueError(
                    f"planning_interval ({self.planning_interval}) cannot be greater than "
                    f"episodes_per_trajectory ({self.episodes_per_trajectory})"
                )
        
        # Validate trajectory parameters
        if self.total_trajectories < 1:
            raise ValueError(f"total_trajectories must be >= 1, got {self.total_trajectories}")
        
        if self.episodes_per_trajectory < 1:
            raise ValueError(f"episodes_per_trajectory must be >= 1, got {self.episodes_per_trajectory}")
        
        # Validate max_questions
        if self.max_questions < 1:
            raise ValueError(f"max_questions must be >= 1, got {self.max_questions}")
    
    def get_categories(self) -> List[List[str]]:
        """
        Get list of category lists to use in the experiment.
        
        Returns:
            List of lists: [[cats for traj1], [cats for traj2], ...]
        """
        from pipeline.core.simulate_interaction import list_categories
        
        # Case 1: Categories explicitly provided
        if self.categories is not None:
            return self.categories
        
        # Case 2: Generate categories based on personas structure
        if self.persona_indices is not None:
            all_categories = list_categories()
            result = []
            
            if self.experiment_type == "variable_category":
                # Need episodes_per_trajectory to determine how many categories per trajectory
                if self.episodes_per_trajectory is None:
                    raise ValueError("episodes_per_trajectory required when only personas provided in variable_category mode")
                
                for traj_idx in range(len(self.persona_indices)):
                    sampled = random.sample(all_categories, min(self.episodes_per_trajectory, len(all_categories)))
                    result.append(sampled)
            
            elif self.experiment_type == "variable_persona":
                # Generate 1 category per trajectory (constant across episodes)
                for traj_idx in range(len(self.persona_indices)):
                    sampled = random.sample(all_categories, 1)
                    result.append(sampled)
            
            else:  # variable_settings
                # Match structure of personas
                for persona_list in self.persona_indices:
                    sampled = random.sample(all_categories, min(len(persona_list), len(all_categories)))
                    result.append(sampled)
            
            return result
        
        # Case 3: Need to generate random categories (neither provided)
        all_categories = list_categories()
        result = []
        
        for traj_idx in range(self.total_trajectories):
            if self.experiment_type == "variable_category":
                cats_per_traj = self.episodes_per_trajectory
            elif self.experiment_type == "variable_persona":
                cats_per_traj = 1
            else:  # variable_settings
                cats_per_traj = self.episodes_per_trajectory
            
            sampled = random.sample(all_categories, min(cats_per_traj, len(all_categories)))
            result.append(sampled)
        
        return result
    
    def get_persona_indices(self) -> List[List[int]]:
        """
        Get list of persona index lists to use in the experiment.
        
        Returns:
            List of lists: [[personas for traj1], [personas for traj2], ...]
        """
        from pipeline.core.personas import get_persona_description
        
        # Case 1: Persona indices explicitly provided
        if self.persona_indices is not None:
            return self.persona_indices
        
        # Get total number of available personas
        max_persona_index = 0
        while True:
            try:
                get_persona_description(max_persona_index)
                max_persona_index += 1
            except:
                break
        
        all_persona_indices = list(range(max_persona_index))
        
        # Case 2: Generate personas based on categories structure
        if self.categories is not None:
            result = []
            
            if self.experiment_type == "variable_persona":
                # Need episodes_per_trajectory to determine how many personas per trajectory
                if self.episodes_per_trajectory is None:
                    raise ValueError("episodes_per_trajectory required when only categories provided in variable_persona mode")
                
                for traj_idx in range(len(self.categories)):
                    sampled = random.sample(all_persona_indices, min(self.episodes_per_trajectory, len(all_persona_indices)))
                    result.append(sampled)
            
            elif self.experiment_type == "variable_category":
                # Generate 1 persona per trajectory (constant across episodes)
                for traj_idx in range(len(self.categories)):
                    sampled = random.sample(all_persona_indices, 1)
                    result.append(sampled)
            
            else:  # variable_settings
                # Match structure of categories
                for category_list in self.categories:
                    sampled = random.sample(all_persona_indices, min(len(category_list), len(all_persona_indices)))
                    result.append(sampled)
            
            return result
        
        # Case 3: Need to generate random personas (neither provided)
        result = []
        
        for traj_idx in range(self.total_trajectories):
            if self.experiment_type == "variable_persona":
                personas_per_traj = self.episodes_per_trajectory
            elif self.experiment_type == "variable_category":
                personas_per_traj = 1
            else:  # variable_settings
                personas_per_traj = self.episodes_per_trajectory
            
            sampled = random.sample(all_persona_indices, min(personas_per_traj, len(all_persona_indices)))
            result.append(sampled)
        
        return result
    
    def get_seeds(self) -> List[int]:
        """
        Get list of seeds for trajectories.
        
        Logic:
        - total_trajectories defines how many seeds are needed
        - If seeds provided: use first total_trajectories seeds
        - If seeds has fewer than total_trajectories: pad with random seeds
        - If seeds not provided: generate total_trajectories random seeds
        
        Returns:
            List of seeds (length = total_trajectories)
        """
        if self.seeds is not None:
            # Seeds provided: use first N, or pad with random if not enough
            if len(self.seeds) >= self.total_trajectories:
                return self.seeds[:self.total_trajectories]
            else:
                # Pad with random seeds
                additional_seeds = [random.randint(1, 1000000) for _ in range(self.total_trajectories - len(self.seeds))]
                return self.seeds + additional_seeds
        else:
            # No seeds provided: generate random seeds
            return [random.randint(1, 1000000) for _ in range(self.total_trajectories)]
    
    def validate_experiment_constraints(self):
        """
        Validate experiment-specific constraints with flexible inference.
        
        New Logic:
        1. variable_settings: If either provided → infer missing one (no counts needed)
        2. variable_category: 
           - If categories provided → infer personas (1 per traj)
           - If personas provided → need episodes_per_trajectory
        3. variable_persona:
           - If personas provided → infer categories (1 per traj)
           - If categories provided → need episodes_per_trajectory
        """
        both_provided = self.categories is not None and self.persona_indices is not None
        neither_provided = self.categories is None and self.persona_indices is None
        only_categories = self.categories is not None and self.persona_indices is None
        only_personas = self.categories is None and self.persona_indices is not None
        
        # Validate list-of-lists structure if provided
        if self.categories is not None:
            if not isinstance(self.categories, list) or not all(isinstance(c, list) for c in self.categories):
                raise ValueError("categories must be List[List[str]] when provided")
        
        if self.persona_indices is not None:
            if not isinstance(self.persona_indices, list) or not all(isinstance(p, list) for p in self.persona_indices):
                raise ValueError("persona_indices must be List[List[int]] when provided")
        
        # CASE 1: Both provided
        if both_provided:
            # Must have same number of trajectories
            if len(self.categories) != len(self.persona_indices):
                raise ValueError(
                    f"categories and persona_indices must have same number of trajectories. "
                    f"Got {len(self.categories)} category trajectories and {len(self.persona_indices)} persona trajectories"
                )
            
            # Derive total_trajectories from structure
            self.total_trajectories = len(self.categories)
            
            # Validate mode-specific constraints
            if self.experiment_type == "variable_category":
                # Each persona list should have exactly 1 persona (constant across episodes)
                for i, personas in enumerate(self.persona_indices):
                    if len(personas) != 1:
                        raise ValueError(
                            f"variable_category mode requires exactly 1 persona per trajectory. "
                            f"Trajectory {i} has {len(personas)} personas: {personas}"
                        )
            
            elif self.experiment_type == "variable_persona":
                # Each category list should have exactly 1 category (constant across episodes)
                for i, cats in enumerate(self.categories):
                    if len(cats) != 1:
                        raise ValueError(
                            f"variable_persona mode requires exactly 1 category per trajectory. "
                            f"Trajectory {i} has {len(cats)} categories: {cats}"
                        )
            
            # For variable_settings, both can vary freely
        
        # CASE 2: Only categories provided
        elif only_categories:
            # Derive total_trajectories from categories structure
            self.total_trajectories = len(self.categories)
            
            if self.experiment_type == "variable_category":
                # Will generate 1 persona per trajectory (validated in get_persona_indices)
                pass
            
            elif self.experiment_type == "variable_persona":
                # Need episodes_per_trajectory to generate personas
                if self.episodes_per_trajectory is None or self.episodes_per_trajectory <= 0:
                    raise ValueError(
                        "episodes_per_trajectory required when only categories provided in variable_persona mode"
                    )
                
                # Each category list should have exactly 1 category
                for i, cats in enumerate(self.categories):
                    if len(cats) != 1:
                        raise ValueError(
                            f"variable_persona mode requires exactly 1 category per trajectory. "
                            f"Trajectory {i} has {len(cats)} categories: {cats}"
                        )
            
            else:  # variable_settings
                # Will generate personas to match categories structure (no additional params needed)
                pass
        
        # CASE 3: Only personas provided
        elif only_personas:
            # Derive total_trajectories from personas structure
            self.total_trajectories = len(self.persona_indices)
            
            if self.experiment_type == "variable_persona":
                # Will generate 1 category per trajectory (validated in get_categories)
                pass
            
            elif self.experiment_type == "variable_category":
                # Need episodes_per_trajectory to generate categories
                if self.episodes_per_trajectory is None or self.episodes_per_trajectory <= 0:
                    raise ValueError(
                        "episodes_per_trajectory required when only personas provided in variable_category mode"
                    )
                
                # Each persona list should have exactly 1 persona
                for i, personas in enumerate(self.persona_indices):
                    if len(personas) != 1:
                        raise ValueError(
                            f"variable_category mode requires exactly 1 persona per trajectory. "
                            f"Trajectory {i} has {len(personas)} personas: {personas}"
                        )
            
            else:  # variable_settings
                # Will generate categories to match personas structure (no additional params needed)
                pass
        
        # CASE 4: Neither provided
        else:
            # total_trajectories and episodes_per_trajectory must be provided
            if self.total_trajectories is None or self.total_trajectories <= 0:
                raise ValueError("total_trajectories must be provided when categories and persona_indices are not specified")
            if self.episodes_per_trajectory is None or self.episodes_per_trajectory <= 0:
                raise ValueError("episodes_per_trajectory must be provided when categories and persona_indices are not specified")
    
    def get_used_categories(self) -> Optional[List[List[str]]]:
        """Get categories that will actually be used in the experiment."""
        # This will be populated by the experiment runner based on actual usage
        return getattr(self, '_used_categories', self.categories)
    
    def get_used_persona_indices(self) -> Optional[List[List[int]]]:
        """Get persona indices that will actually be used in the experiment."""
        # This will be populated by the experiment runner based on actual usage
        return getattr(self, '_used_persona_indices', self.persona_indices)
    
    def is_batch(self) -> bool:
        """Check if this is a batch run (multiple trajectories)."""
        return self.total_trajectories > 1
    
    def get_output_path(self, seed: Optional[int] = None) -> str:
        """
        Generate output path based on config.
        
        Args:
            seed: If provided, includes seed in path (for batch runs)
            
        Returns:
            Output directory path
        """
        if self.experiment_name:
            name = self.experiment_name
        else:
            name = f"{self.experiment_type}_{self.model.replace('/', '_')}_{self.feedback_type}"
        
        if seed is not None and self.is_batch():
            name = f"{name}_seed{seed}"
        
        return os.path.join(self.output_dir, name)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExperimentConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ValueError: If configuration parameters are invalid
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config.validate()  # Validate immediately after loading
        return config
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ExperimentConfig':
        """
        Load configuration from JSON file.
        
        Raises:
            ValueError: If configuration parameters are invalid
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        config = cls(**data)
        config.validate()  # Validate immediately after loading
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_dict_complete(self) -> Dict[str, Any]:
        """Convert to dictionary with all parameters, showing used values."""
        from dataclasses import fields
        
        result = {}
        
        # Get field defaults
        field_defaults = {}
        for field in fields(self):
            if field.default is not field.default_factory:
                field_defaults[field.name] = field.default
            elif field.default_factory is not field.default_factory:
                field_defaults[field.name] = field.default_factory()
            else:
                field_defaults[field.name] = None
        
        # Compare current values with defaults
        for key, value in asdict(self).items():
            default_value = field_defaults.get(key)
            
            # Special handling for actual used values
            if key == "categories":
                # Show used categories (list-of-lists)
                result[key] = self.get_used_categories()
            elif key == "persona_indices":
                # Show used persona indices (list-of-lists)
                result[key] = self.get_used_persona_indices()
            elif key == "seeds":
                # Show all seeds used (list)
                result[key] = self.get_seeds()
            elif value != default_value:
                result[key] = value
            else:
                # Set unused parameters to null
                result[key] = None
        
        return result
    
    @classmethod
    def quick_test(cls, experiment_type: str = "variable_category", 
                   model: str = "gpt-4o") -> 'ExperimentConfig':
        """
        Create quick test configuration (minimal episodes for testing).
        
        Args:
            experiment_type: Type of experiment
            model: Model to use
            
        Returns:
            ExperimentConfig for quick testing
        """
        return cls(
            experiment_type=experiment_type,
            model=model,
            num_categories=1,
            episodes_per_category=2,
            num_personas=2,
            episodes_per_persona=2,
            total_episodes=5,
            max_questions=8,
            context_mode="none",
            feedback_type="none",
            seed=42
        )


import os

