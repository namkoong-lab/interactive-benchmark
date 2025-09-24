# DARE: Interactive Recommendation Benchmark

This repository contains the codebase for the paper *"Benchmarking Multi-Episode Learning for Adaptive Agents: An Interactive Recommendation Dataset"* It provides an environment, dataset, and evaluation framework for studying how language model agents learn and adapt through multi-turn, multi-episode interactions.

## Overview

DARE evaluates an agent’s ability to:
- Ask strategic questions to elicit preferences.
- Recommend products across diverse categories.
- Learn from feedback over multiple episodes.
- Manage uncertainty and calibrate confidence.

The benchmark integrates:
- **Personas**: 1,000+ richly described user profiles.
- **Products**: 70K+ Amazon items organized into ~2K substitute sets.
- **Feedback**: Regret values, star ratings, and free-form text.

## Project Structure

- `pipeline/`: Core framework (personas, environments, feedback, LLM clients).
- `experiments/`: Baselines (oracle, popularity, random) and planning strategies (DP, greedy).
- `experiment_runners/`: Scripts to launch experiments.
- `database/`: Product database and population utilities.
- `database_creation/`: Category mapping and preprocessing.
- `graphing/`: Visualization tools (ECE, regret, calibration plots).

## Experiment Types

- **Baselines**: Oracle, random, and popularity recommenders.
- **Planning**: Dynamic programming–style and greedy questioning.
- **Variable Experiments**: Vary personas, categories, or both to test adaptability.

## Metrics

- Regret (vs. optimal recommendation).
- Ranking of chosen product.
- Calibration and Expected Calibration Error (ECE).
- Question counts and interaction efficiency.

## Installation

```bash
git clone <repo_url>
pip install -r requirements.txt
```

Set up a `.env` file with API keys (OpenAI, Google Gemini, Claude).

## Quick Start 

```bash
# Oracle baseline
python experiment_runners/run_baseline_oracle.py

# Planning (DP-style) experiment
python experiment_runners/run_planning_dp.py

# Variable persona experiment
python experiment_runners/run_variable_persona.py
```

For visualization:

```bash
python graphing/graphing_ece.py --input results.csv
```

## Research Applications

DARE enables research in:
- Multi-episode learning and adaptation.
- Preference elicitation and personalization.
- Strategic question-asking.
- Calibration of confidence estimates.