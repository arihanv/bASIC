# KernelBench Dataset Loading Guide

This guide explains how to load and use the KernelBench dataset for testing LLMs' ability to generate efficient GPU CUDA kernels.

## Dataset Overview

The KernelBench dataset is hosted on Hugging Face and contains problems across four levels of increasing complexity:

1. **Level 1** (100 problems): Single-kernel operators (e.g., matrix multiplication, convolutions, normalization)
2. **Level 2** (100 problems): Simple fusion patterns (e.g., Conv + Bias + ReLU)
3. **Level 3** (50 problems): Full model architectures (e.g., MobileNet, VGG)
4. **Level 4** (20 problems): Hugging Face model architectures

Each problem includes:
- `code`: The PyTorch implementation of the operator/model
- `level`: The difficulty level (1-4)
- `name`: The name of the problem
- `problem_id`: The unique identifier for the problem

## Installation Requirements

To load the KernelBench dataset, you need to install the Hugging Face datasets library:

```bash
pip install datasets
```

## Loading the Dataset

### Loading the Full Dataset

```python
from datasets import load_dataset

# Load the full KernelBench dataset
dataset = load_dataset('ScalingIntelligence/KernelBench')

# The dataset is a DatasetDict with keys 'level_1', 'level_2', 'level_3', and 'level_4'
print(dataset.keys())  # Output: dict_keys(['level_1', 'level_2', 'level_3', 'level_4'])
```

### Accessing a Specific Level

```python
# Access problems from a specific level (e.g., Level 1)
level = 1
level_dataset = dataset[f'level_{level}']

# Print the number of problems in this level
print(f"Number of problems in Level {level}: {len(level_dataset)}")
```

### Accessing a Specific Problem

```python
# Get a specific problem by its ID
problem_id = 1  # Problem IDs start from 1
problem = level_dataset.filter(lambda x: x['problem_id'] == problem_id)

# Access the problem's code and name
code = problem['code'][0]
name = problem['name'][0]

print(f"Problem name: {name}")
print(f"Problem code:\n{code}")
```

### Iterating Through Problems

```python
# Iterate through all problems in a level
for i, problem in enumerate(level_dataset):
    print(f"Problem {i+1}: {problem['name']}")
    
    # Access the problem's code
    code = problem['code']
    
    # Process the problem...
```

## Example: Loading and Using a Problem for CUDA Kernel Generation

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('ScalingIntelligence/KernelBench')

# Select a level and problem
level = 1
problem_id = 1

# Get the problem
level_dataset = dataset[f'level_{level}']
problem = level_dataset.filter(lambda x: x['problem_id'] == problem_id)
code = problem['code'][0]
name = problem['name'][0]

print(f"Processing problem: {name} (Level {level}, ID {problem_id})")

# Create a prompt for the LLM to generate a CUDA kernel
prompt = f"""You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.

You are given the following architecture:

```
{code}
```

Optimize the architecture named Model with custom CUDA kernels! Name your optimized output architecture ModelNew.
"""

# Send the prompt to your LLM endpoint and get the response
# response = your_llm_endpoint(prompt)

# Extract the generated CUDA kernel from the response
# kernel = extract_code_from_response(response)

# Save the kernel to a file
# with open(f"level_{level}_problem_{problem_id}_kernel.py", "w") as f:
#     f.write(kernel)
```

## Dataset Structure

The dataset has the following structure:

```
DatasetDict({
    level_1: Dataset({
        features: ['code', 'level', 'name', 'problem_id'],
        num_rows: 100
    })
    level_2: Dataset({
        features: ['code', 'level', 'name', 'problem_id'],
        num_rows: 100
    })
    level_3: Dataset({
        features: ['code', 'level', 'name', 'problem_id'],
        num_rows: 50
    })
    level_4: Dataset({
        features: ['code', 'level', 'name', 'problem_id'],
        num_rows: 20
    })
})
```

## Additional Resources

- KernelBench GitHub Repository: [https://github.com/ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench)
- KernelBench Hugging Face Dataset: [https://huggingface.co/datasets/ScalingIntelligence/KernelBench](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)
- KernelBench Paper: [https://arxiv.org/html/2502.10517v1](https://arxiv.org/html/2502.10517v1)
