# bASIC

A Python implementation of speculative tree decoding with adaptive draft tree structure, based on the OPT-Tree approach.

## Overview

This project implements speculative tree decoding with adaptive draft tree structure for the DeepSeek-R1-Distill-Qwen-32B model. The implementation is based on the [OPT-Tree paper](https://arxiv.org/pdf/2406.17276), which proposes an algorithm to construct adaptive and scalable draft trees for speculative decoding.

Speculative decoding is a technique that allows generating multiple tokens in a single decoding step, significantly improving inference efficiency. The OPT-Tree approach constructs an optimal tree structure that maximizes the mathematical expectation of the acceptance length in each decoding step.

## Features

- **Adaptive Tree Structure**: Dynamically constructs and prunes a tree structure for speculative decoding
- **Parallel Processing**: Leverages multiple GPUs (up to 8 H100s) for efficient tree construction and verification
- **Streaming Generation**: Supports streaming output for real-time text generation
- **Benchmarking**: Includes tools to measure performance improvements over standard decoding

## Installation

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/arihanv/bASIC.git
cd bASIC
```

## Usage

### Basic Usage

```python
from main import generate_with_standard_model, generate_with_speculative_decoding

# Generate text using standard decoding
standard_output = generate_with_standard_model("What is the capital of France?")

# Generate text using speculative tree decoding
speculative_output = generate_with_speculative_decoding("What is the capital of France?")

# Generate text using parallel speculative tree decoding
parallel_output = generate_with_speculative_decoding("What is the capital of France?", use_parallel=True)
```

### Streaming Output

```python
from main import generate_with_streaming_speculative_decoding

# Generate text with streaming output
generate_with_streaming_speculative_decoding("What is the capital of France?")
```

### Benchmarking

```bash
python benchmark.py --prompt "What is the capital of France?" --max-tokens 50 --runs 5
```

## Implementation Details

The implementation consists of several key components:

1. **Tree Class**: Manages the tree structure for speculative decoding, including node addition, path tracking, and attention mask generation.

2. **SPModel Class**: Implements the speculative decoding model, handling token drafting and verification.

3. **ParallelSPModel Class**: Extends SPModel to leverage multiple GPUs for parallel processing.

4. **Utility Functions**: Provides helper functions for tree decoding, verification, and logits processing.

## Performance

The speculative tree decoding implementation can generate more than 10 tokens in a single decoding step, achieving a significant speedup compared to standard autoregressive decoding. The parallel implementation further improves performance by distributing the workload across multiple GPUs.

## References

- [OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure](https://arxiv.org/pdf/2406.17276)
- [OPT-Tree GitHub Repository](https://github.com/Jikai0Wang/OPT-Tree)
