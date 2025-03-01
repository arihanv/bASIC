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
