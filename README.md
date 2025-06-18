# Amazon Product Categorization and Visualization

This project provides tools for categorizing Amazon products and visualizing their relationships in 3D space. It uses AI to create a hierarchical category structure and allows interactive exploration of product data.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── amazon_dataset/          # Product categorization and visualization
│   ├── benchmark_metadata/  # Raw product metadata
│   ├── categorized_products/# Output directory for categorized products
│   ├── categorize.py       # Product categorization script
│   ├── item_index.py      # Data indexing and retrieval
│   ├── visualize.py       # 3D visualization tool
│   └── run_pipeline.py    # Main pipeline runner
│
└── pipeline/               # Persona-based recommender system
    ├── personas.py        # Persona generation and management
    ├── simulate_interaction.py  # Core simulation logic
    └── simulation.py      # Simulation runner
```

## Running the Pipeline

### 1. Categorize Products

```bash
python amazon_dataset/run_pipeline.py
```

This will:
- Load product metadata from `benchmark_metadata/`
- Generate or load category hierarchy
- Categorize products and save results to `categorized_products/`
- Each product is saved as `item_N.json` with its metadata and category

### 2. Visualize Products

To visualize products in 3D:

```bash
python amazon_dataset/visualize.py
```

The visualization tool allows you to:
1. Select a category to visualize (or view all categories)
2. Choose three dimensions to plot:
   - Price
   - Year of Production
   - Average Rating
   - Number of Ratings
   - Size of Bundle
   - Other metadata fields

The visualization is interactive:
- Rotate: Click and drag
- Zoom: Scroll
- Pan: Right-click and drag
- Hover: See exact values


### 4. Persona-Based Recommender System

The `pipeline/` directory contains a persona-based recommender system that simulates interactions between users and an AI recommender:

```bash
# Generate persona benchmark entries
python pipeline/personas.py <comma_separated_indices>

# Run simulation with a specific persona
python pipeline/simulation.py <persona_index> [llm_b_model] [num_questions]
```

The recommender system:
- Generates persona profiles with preferences and attributes
- Simulates user-recommender interactions
- Evaluates recommendation accuracy
- Supports different LLM models for the recommender

Key components:
- `personas.py`: Generates benchmark entries for personas with:
  - Product preferences
  - User attributes
  - Correct product matches
  - Noise traits for evaluation
- `simulate_interaction.py`: Core simulation functions:
  - `llm_a_respond()`: Simulates user answers
  - `llm_b_interact()`: Runs dynamic questioning and recommendations
- `simulation.py`: Command-line interface for running simulations

## Data Format

### Input Data
Products are stored in `benchmark_metadata/` as JSON files with:
- Title
- Description
- Brand
- Price
- Features
- Other metadata

### Output Data
Categorized products are saved in `categorized_products/` as JSON files with:
- Original metadata
- Category path
- Confidence score

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`