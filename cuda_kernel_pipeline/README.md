# CUDA Kernel Collection and Preprocessing Pipeline

This pipeline collects, preprocesses, filters, and creates training datasets from CUDA kernels for training an LLM that generates optimized CUDA code.

## Overview

The pipeline consists of four main components:

1. **Collector**: Collects CUDA kernels from GitHub repositories
2. **Preprocessor**: Cleans and preprocesses the collected kernels
3. **Quality Filter**: Filters kernels based on quality metrics
4. **Dataset Creator**: Creates training datasets from preprocessed kernels

## Installation

```bash
# Clone the repository
git clone https://github.com/arihanv/bASIC.git
cd bASIC

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the complete pipeline
python -m cuda_kernel_pipeline.main --output-dir ./data --github-token YOUR_GITHUB_TOKEN

# Skip specific steps
python -m cuda_kernel_pipeline.main --output-dir ./data --skip-collection --skip-preprocessing

# Adjust quality threshold
python -m cuda_kernel_pipeline.main --output-dir ./data --min-quality 0.8
```

## Pipeline Components

### Collector

The collector searches for high-quality CUDA kernel implementations in both official NVIDIA repositories and community repositories.

```python
from cuda_kernel_pipeline.collector.collector import CUDAKernelCollector

collector = CUDAKernelCollector(
    output_dir="./data/raw",
    min_stars=50,
    max_repos=100,
    github_token="YOUR_GITHUB_TOKEN"
)
stats = collector.collect_all()
```

### Preprocessor

The preprocessor cleans, normalizes, and extracts features from CUDA kernel code to prepare it for model training.

```python
from cuda_kernel_pipeline.preprocessor.preprocessor import CUDAKernelPreprocessor

preprocessor = CUDAKernelPreprocessor(
    input_dir="./data/raw",
    output_dir="./data/preprocessed"
)
stats = preprocessor.preprocess_all_kernels()
```

### Quality Filter

The quality filter assesses kernel quality based on compilation success, code quality metrics, performance indicators, and complexity/diversity.

```python
from cuda_kernel_pipeline.quality_filter.quality_filter import KernelQualityFilter

quality_filter = KernelQualityFilter(
    kernels_dir="./data/preprocessed",
    output_dir="./data/filtered"
)
stats = quality_filter.filter_kernels(min_quality_score=0.7)
```

### Dataset Creator

The dataset creator generates various types of training examples for different training scenarios, including code completion and performance optimization.

```python
from cuda_kernel_pipeline.dataset_creator.dataset_creator import CUDADatasetCreator

dataset_creator = CUDADatasetCreator(
    processed_kernels_dir="./data/filtered",
    output_dir="./data/datasets"
)
stats = dataset_creator.create_training_dataset()
```

## Data Sources

The pipeline collects CUDA kernels from the following sources:

1. **NVIDIA Official Repositories**:
   - [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
   - [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
   - [NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)
   - [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
   - [NVIDIA/cccl](https://github.com/NVIDIA/cccl)

2. **High-Quality Community Repositories**:
   - Repositories with topics: `cuda-kernels`, `cuda-programming`, `gpu-acceleration`
   - Repositories with minimum star count (default: 50)

## Output Data Format

The pipeline produces the following outputs:

1. **Raw Kernels**: Original CUDA kernel files collected from repositories
2. **Preprocessed Kernels**: Cleaned and normalized kernel files with extracted features
3. **Filtered Kernels**: High-quality kernels that pass the quality filter
4. **Training Datasets**: JSONL files containing training examples for different scenarios:
   - `train_completion.jsonl`: Code completion examples for training
   - `val_completion.jsonl`: Code completion examples for validation
   - `test_completion.jsonl`: Code completion examples for testing
   - `train_perf.jsonl`: Performance-annotated examples for RL training
   - `val_perf.jsonl`: Performance-annotated examples for RL validation

## Requirements

- Python 3.8+
- PyGithub
- NVIDIA CUDA Toolkit (for compilation checking)

## License

This project is licensed under the terms of the license included with this repository.
