# Kernel Benchmarking Testing

This directory contains tools for testing and benchmarking CUDA kernel implementations using LLM-generated code.

## Setup

1. Create and activate a Python virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in this directory with your API keys:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

The main script supports two modes of operation:

1. Using Claude API (recommended):
```bash
python simplified_test.py --use-claude --verbose --time-generation
```

2. Using custom inference (if available):
```bash
python simplified_test.py --verbose --time-generation
```

### Command Line Arguments

- `--use-claude`: Use Claude API instead of custom inference
- `--verbose`: Print verbose output during generation
- `--time-generation`: Time the generation process

## Project Structure

- `simplified_test.py`: Main script for generating CUDA kernels
- `inference_utils.py`: Utilities for different inference backends
- `requirements.txt`: Python package dependencies
- `.env`: Environment variables (API keys)

## Generated Code

The generated CUDA kernels will be output directly to the console. The code includes:
- Custom CUDA kernel implementation
- PyTorch wrapper class
- Necessary imports and setup code

## Requirements

- Python 3.8+
- CUDA toolkit (for running generated kernels)
- PyTorch with CUDA support
- Anthropic API key (for Claude)
# KernelBench Testing

This directory contains code and resources for testing the DeepSeek-R1-Distill-Qwen-32B model on KernelBench, a benchmark for evaluating LLMs' ability to generate efficient GPU CUDA kernels.

## Contents

- `scrape_leaderboard.py`: Script to scrape the KernelBench leaderboard and create Q&A pairs
- `qa_pairs/`: Directory containing Q&A pairs from the KernelBench leaderboard
  - `problems/`: Individual problem files
  - `solutions/`: Individual solution files (rank 1 submissions)
  - `kernelbench_qa_pairs.json`: JSON file containing all Q&A pairs
  - `kernelbench_qa_pairs.md`: Markdown file containing all Q&A pairs
  - `kernelbench_qa_pairs.csv`: CSV file containing all Q&A pairs

## Usage

### Scraping the Leaderboard

To scrape the KernelBench leaderboard and create Q&A pairs:

```bash
cd ~/repos/bASIC
python -m kernelbench_testing.scrape_leaderboard
```

This will:
1. Fetch the KernelBench leaderboard from https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/
2. Extract problem statements and rank 1 solutions
3. Create Q&A pairs with the problem statement as the question and the rank 1 solution as the answer
4. Save the Q&A pairs in multiple formats (JSON, Markdown, CSV, individual files)

### Using the Q&A Pairs

The Q&A pairs can be used to:
- Train or fine-tune an LLM to generate efficient CUDA kernels
- Evaluate an LLM's ability to generate CUDA kernels
- Compare different LLMs' performance on CUDA kernel generation

## Resources

- KernelBench GitHub Repository: https://github.com/ScalingIntelligence/KernelBench
- KernelBench Hugging Face Dataset: https://huggingface.co/datasets/ScalingIntelligence/KernelBench
- KernelBench Leaderboard: https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/
