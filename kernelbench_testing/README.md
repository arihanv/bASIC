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
