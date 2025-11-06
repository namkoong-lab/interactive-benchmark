# AIR: Agentic In-context Experiential Reasoning

**Benchmarking In-context Experiential Reasoning Through Repeated Product Recommendations**

This repository implements AIR, a benchmark for evaluating how LLM agents learn and adapt through experiential reasoning in multi-episode product recommendation scenarios. AIR challenges agents to improve performance across episodes by learning through natural language interactions rather than through explicit parameter updates. 

## Overview

AIR evaluates an agent's ability to perform **in-context experiential reasoning**. Specifically, agents must:
- Elicit latent user preferences through strategic questioning
- Navigate evolving product landscapes and user needs
- Leverage cross-episode memory to improve recommendations
- Manage uncertainty in incomplete information environments

### Core Components

1. **Real-world Products**: 71K+ Amazon items across 2K+ categories with rich metadata
2. **Diverse Personas**: 40K+ user profiles with varied, latent preferences and demographics
3. **LLM User Simulator**: Realistic interaction trajectories powered by persona-driven response generation

## Project Structure

```
├── pipeline/                 # Core framework
│   ├── core/                # Personas, agents, LLM providers, scoring
│   │   └── llm_providers/  # OpenAI, Claude, Gemini integrations
│   ├── envs/               # Recommendation environment (Gymnasium)
│   └── wrappers/           # Metrics, feedback, logging
├── experiments/             # Experiment orchestration and baselines
├── experiment_runners/      # Configuration and launch scripts
│   └── configs/            # YAML configuration files
├── config/                 # Configuration dataclasses (Python code)
├── database/               # Product database, caching, HuggingFace sync
├── database_creation/      # Scripts for categorizing/processing products
├── data/                   # Personas, product mappings, trajectories
├── graphing/               # Visualization and analysis tools
├── webpage/                # Interactive leaderboard and submission interface
```

## Key Features

### Configuration System
All experiments use YAML configs with 31 parameters covering:
- Experiment setup (type, episodes, trajectories, seeds)
- Agent parameters (model, temperature, max questions)
- Context modes (raw, summary, planning)
- Feedback types (persona, oracle, LLM-based)
- Checkpointing and resumption
- Interactive trajectory generation

**Example**: `experiment_runners/config_reference.yaml` documents all parameters.


### Experiment Types

AIR supports three experimental paradigms to isolate different adaptation challenges:

- **`variable_category`**: Fixed persona, varying product categories (preference generalization)
- **`variable_persona`**: Fixed category, varying user personas (user adaptation)
- **`variable_settings`**: Both persona and category vary (full adaptation)

### Planning Modes

Planning modes force the agent to give a recommendation after each question within an episode, enabling analysis of within-episode improvement and whether this learning rate increases across later episodes. 
- **`planning_no_strat`**: Non-modified experiment 
- **`planning_greedy`**: Greedy question selection
- **`planning_dp`**: Dynamic programming-style lookahead

```yaml
planning_mode: "planning_dp"
planning_interval: 5
```

### Interactive Mode
Generate multiple trajectory variants for manual curation:
1. System produces N variants of Episode 1
2. User selects preferred variant
3. System generates N variants of Episode 2 from selected Episode 1
4. Repeat until trajectory complete

```yaml
interactive_mode: true
interactive_variants: 10
interactive_input_file: "episode_01_variant_003.json"  # For continuation
```
## Setup & Installation

### Prerequisites

- Python 3.9+
- ~1GB disk space (500MB database + dependencies)
- API keys for LLM providers (at least OpenAI and Google for scoring):
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)

### Installation Steps

**1. Clone Repository**
```bash
git clone https://github.com/namkoong-lab/personas.git
cd personas
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure API Keys**

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
```

**4. Database Setup**

The product database is hosted on HuggingFace and will **automatically download** on first run. 

#### Automatic Setup (Recommended)
```bash
# Just run any experiment - database downloads automatically
cd experiment_runners
python run_experiment.py --config configs/basic_variable_category.yaml
```

On first run, you'll see:
```
🔄 Database not found. Downloading from HuggingFace...
📦 Downloading products_part1.parquet (4/4)...
✅ Database setup complete!
```

#### Manual Pre-download (Optional)
```bash
# Pre-download database before running experiments
cd database
python setup_database.py
```

This downloads 4 Parquet files (~500MB total) from HuggingFace and builds a local SQLite database.

### Database Contents

The AIR database contains:
- **71,088 products** from Amazon with rich metadata
- **2,030 product categories** organized into substitute sets
- **Product attributes**: titles, prices, ratings, descriptions, images
- **Score cache**: Stores persona-product scores to avoid re-computation

**Database Structure**:
```
database/
├── personas.db              # SQLite database (auto-generated)
├── setup_database.py        # Download script
└── cache/                   # Downloaded Parquet files
    ├── products_part1.parquet
    ├── products_part2.parquet
    ├── products_part3.parquet
    └── products_part4.parquet
```

**Score Caching**: The database includes a `persona_scores` table that grows during experiments. Cached scores are reused across runs, speeding up repeated experiments with the same personas/categories.

## Quick Start

### Basic Experiments

```bash
cd experiment_runners

# Run with example config
python run_experiment.py --config configs/basic_variable_category.yaml

# Interactive trajectory building
python run_experiment.py --config configs/interactive_example.yaml
```

### Resuming from Checkpoint

```yaml
checkpoint_enabled: true
resume_from_checkpoint: "experiment_results/checkpoint_traj2_ep8.json"
```

## Integrating Custom Models

AIR's modular architecture makes it easy to benchmark your own LLM models and agents.

### Option 1: Add a New LLM Provider

Integrate a new LLM API (e.g., Cohere, Mistral, local models) in 4 steps:

1. **Copy the template**: Use `pipeline/core/llm_providers/custom_provider_template.py` as a starting point
2. **Implement two methods**:
   - `__init__()`: Load API key and initialize client
   - `chat_completion()`: Make API calls with retry logic
3. **Register your provider** in `pipeline/core/llm_providers/__init__.py`
4. **Add API key** to `.env` and use in your config

**See**: `custom_provider_template.py` for detailed implementation guide and examples from `openai_provider.py`, `claude_provider.py`, `gemini_provider.py`.

**Test your provider**:
```python
from pipeline.core.llm_providers import chat_completion
response = chat_completion(
    model="my-model-v1",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Option 2: Custom Agent Logic

For advanced agent behavior (custom prompting, tool use, RAG), extend `UnifiedAgent`:

1. **Create custom agent** in `pipeline/core/my_custom_agent.py`
2. **Override methods**:
   - `decide_action()`: Custom decision logic
   - `_build_llm_context()`: Custom prompt construction
   - Add pre/post-processing (tool calls, retrieval, etc.)
3. **Modify experiment runner** to use your agent class
4. **Test on small scale** before full experiments

**Key extension points**:
- `decide_action()`: Control when to ask vs recommend
- `_build_llm_context()`: Customize product/dialog presentation
- `_llm_decide_action()`: Override core LLM prompting
- Add external knowledge, tools, or multi-step reasoning

## Citation

```bibtex
@inproceedings{yang2025air,
  title={Benchmarking In-context Experiential Reasoning Through Repeated Product Recommendations},
  author={Yang, Gilbert and Chen, Yaqin and Yen, Thomson and Namkoong, Hongseok},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.