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
    
    Supports:
    - variable_category: Same customer, different categories
    - variable_persona: Different customers, same category  
    - variable_settings: Both vary
    """
    
    # === EXPERIMENT TYPE ===
    experiment_type: Literal["variable_category", "variable_persona", "variable_settings"]
    
    # === MODEL SETTINGS ===
    model: str = "gpt-4o"
    
    # === EXPERIMENT PARAMETERS ===
    max_questions: int = 8
    
    # === LEARNING SETTINGS ===
    context_mode: Literal["raw", "summary", "none"] = "raw"
    prompting_tricks: Literal["none", "all"] = "none"
    feedback_type: Literal["none", "regret", "persona", "star_rating"] = "none"
    
    # === SEED SETTINGS ===
    seeds: Optional[List[int]] = None  # List of seeds (1 or many)
    num_seeds: Optional[int] = None  # Auto-generate N seeds
    
    # === DATA SETTINGS ===
    categories: Optional[List[str]] = None  # None = random selection
    persona_indices: Optional[List[int]] = None  # None = random selection
    min_score_threshold: float = 60.0
    max_products_per_category: int = 100  
    
    # === OUTPUT ===
    output_dir: str = "experiment_results"
    experiment_name: Optional[str] = None
    checkpoint_file: Optional[str] = None
    
    # === EXPERIMENT-SPECIFIC PARAMETERS ===
    # Trajectory-based parameters (replaces episodes_per_category/episodes_per_persona/total_episodes)
    episodes_per_trajectory: int = 5  # Episodes per trajectory (continuous context/feedback)
    total_trajectories: int = 10      # Number of trajectories (agent runs)
    
    def get_categories(self) -> List[str]:
        """
        Get list of categories to use in the experiment.
        
        Logic:
        - If categories provided: use them (pad with random if needed)
        - If num_categories derived: generate random categories
        - If both provided: use categories, pad/truncate to num_categories
        - If neither provided: use all available categories
        
        Returns:
            List of categories
        """
        from pipeline.core.simulate_interaction import list_categories
        
        all_categories = list_categories()
        num_categories = self.get_num_categories()
        
        if self.categories is not None:
            # Categories provided: use them, pad/truncate to num_categories
            if len(self.categories) >= num_categories:
                return self.categories[:num_categories]
            else:
                # Pad with random categories
                remaining_categories = [cat for cat in all_categories if cat not in self.categories]
                random.shuffle(remaining_categories)
                additional_categories = remaining_categories[:num_categories - len(self.categories)]
                return self.categories + additional_categories
        else:
            # No categories provided: generate random categories
            random.shuffle(all_categories)
            return all_categories[:num_categories]
    
    def get_persona_indices(self) -> List[int]:
        """
        Get list of persona indices to use in the experiment.
        
        Logic:
        - If persona_indices provided: use them (pad with random if needed)
        - If num_personas derived: generate random persona indices
        - If both provided: use persona_indices, pad/truncate to num_personas
        - If neither provided: use all available personas
        
        Returns:
            List of persona indices
        """
        from pipeline.core.personas import get_persona_description
        
        # Get total number of available personas
        max_persona_index = 0
        while True:
            try:
                get_persona_description(max_persona_index)
                max_persona_index += 1
            except:
                break
        
        all_persona_indices = list(range(max_persona_index))
        num_personas = self.get_num_personas()
        
        if self.persona_indices is not None:
            # Persona indices provided: use them, pad/truncate to num_personas
            if len(self.persona_indices) >= num_personas:
                return self.persona_indices[:num_personas]
            else:
                # Pad with random persona indices
                remaining_indices = [idx for idx in all_persona_indices if idx not in self.persona_indices]
                random.shuffle(remaining_indices)
                additional_indices = remaining_indices[:num_personas - len(self.persona_indices)]
                return self.persona_indices + additional_indices
        else:
            # No persona indices provided: generate random persona indices
            random.shuffle(all_persona_indices)
            return all_persona_indices[:num_personas]
    
    def get_seeds(self) -> List[int]:
        """
        Get list of seeds to run experiments with.
        
        Logic:
        - If seeds provided: use them (pad with random if num_seeds > len(seeds))
        - If num_seeds provided: generate random seeds
        - If both provided: use seeds, pad/truncate to num_seeds
        - If neither provided: error
        
        Returns:
            List of seeds
        """
        if self.seeds is not None and self.num_seeds is not None:
            # Both provided: use seeds, pad/truncate to num_seeds
            if len(self.seeds) >= self.num_seeds:
                return self.seeds[:self.num_seeds]
            else:
                # Pad with random seeds
                additional_seeds = [random.randint(1, 1000000) for _ in range(self.num_seeds - len(self.seeds))]
                return self.seeds + additional_seeds
                
        elif self.seeds is not None:
            # Only seeds provided: derive num_seeds from length
            return self.seeds
            
        elif self.num_seeds is not None:
            # Only num_seeds provided: generate random seeds
            return [random.randint(1, 1000000) for _ in range(self.num_seeds)]
            
        else:
            # Neither provided: error
            raise ValueError("Either 'seeds' or 'num_seeds' must be provided")
    
    def get_num_seeds(self) -> int:
        """Get the number of seeds (derived from seeds length or num_seeds)."""
        return len(self.get_seeds())
    
    def get_num_categories(self) -> int:
        """Get the number of categories (derived from experiment type and episodes_per_trajectory)."""
        if self.experiment_type == "variable_category":
            return self.episodes_per_trajectory
        elif self.experiment_type == "variable_persona":
            return 1
        else:  # variable_settings or planning
            return self.episodes_per_trajectory
    
    def get_num_personas(self) -> int:
        """Get the number of personas (derived from experiment type and episodes_per_trajectory)."""
        if self.experiment_type == "variable_category":
            return 1
        elif self.experiment_type == "variable_persona":
            return self.episodes_per_trajectory
        else:  # variable_settings or planning
            return self.episodes_per_trajectory
    
    def validate_experiment_constraints(self):
        """Validate experiment-specific constraints."""
        if self.experiment_type == "variable_category":
            # Must have exactly 1 persona
            if self.persona_indices is not None and len(self.persona_indices) > 1:
                raise ValueError(f"variable_category experiment can only use 1 persona, got {len(self.persona_indices)}")
            if self.get_num_personas() > 1:
                raise ValueError(f"variable_category experiment can only use 1 persona, got {self.get_num_personas()} personas")
                
        elif self.experiment_type == "variable_persona":
            # Must have exactly 1 category
            if self.categories is not None and len(self.categories) > 1:
                raise ValueError(f"variable_persona experiment can only use 1 category, got {len(self.categories)}")
            if self.get_num_categories() > 1:
                raise ValueError(f"variable_persona experiment can only use 1 category, got {self.get_num_categories()} categories")
    
    def get_used_categories(self) -> Optional[List[str]]:
        """Get categories that will actually be used in the experiment."""
        # This will be populated by the experiment runner based on actual usage
        return getattr(self, '_used_categories', self.categories)
    
    def get_used_persona_indices(self) -> Optional[List[int]]:
        """Get persona indices that will actually be used in the experiment."""
        # This will be populated by the experiment runner based on actual usage
        return getattr(self, '_used_persona_indices', self.persona_indices)
    
    def is_batch(self) -> bool:
        """Check if this is a batch run (multiple seeds)."""
        return self.get_num_seeds() > 1
    
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
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
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
            
            # Special handling for derived values
            if key == "num_seeds":
                # Show derived num_seeds value
                result[key] = self.get_num_seeds()
            elif key == "num_categories":
                # Show derived num_categories value
                result[key] = self.get_num_categories()
            elif key == "num_personas":
                # Show derived num_personas value
                result[key] = self.get_num_personas()
            elif key == "categories":
                # Show used categories
                result[key] = self.get_used_categories()
            elif key == "persona_indices":
                # Show used persona indices
                result[key] = self.get_used_persona_indices()
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

