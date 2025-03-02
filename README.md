# bASIC: Tree Speculative Decoding for GPU Kernel Generation

bASIC is a high-performance framework that leverages tree speculative decoding to generate optimized GPU kernels. Our approach significantly outperforms state-of-the-art models on [KernelBench](https://github.com/ScalingIntelligence/KernelBench), delivering superior accuracy and generation speed.

## 🌟 Features

- **Tree Speculative Decoding**: Advanced decoding strategy for faster and more accurate kernel generation
- **Custom Endpoint Integration**: Flexible architecture supporting various LLM backends
- **KernelBench Performance**: State-of-the-art results on the KernelBench benchmark suite
- **CUDA Optimization**: Specialized in generating highly optimized CUDA kernels

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA Toolkit
- PyTorch with CUDA support
- OpenAI Python package

### Installation

1. Clone the repository:
```bash
git clone https://github.com/arihanv/bASIC.git
cd bASIC
```

2. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📁 Project Structure

```
bASIC/
├── test_inference.py       # Inference testing with custom endpoints
├── inference_utils.py      # Tree speculative decoding implementation
├── KernelBench/           # Benchmark suite (gitignored)
│   ├── prompt_X_Y.txt     # Problem-specific prompts
│   └── ...
└── requirements.txt       # Project dependencies
```

## 🔧 Usage

### Generating Optimized Kernels

Run the main script with your desired configuration:
```bash
python test
```

### Working with KernelBench Problems

The framework is designed to solve various KernelBench problems. Each problem is addressed through specific prompts:

- **Level 1**:
  - `prompt_1_1.txt`: Matrix multiplication optimization
  - `prompt_1_12.txt`: Diagonal matrix operations
- **Additional Levels**: Follow the naming convention `prompt_X_Y.txt` for level X, problem Y

### Custom Endpoint Configuration

Customize the inference endpoint in `inference_utils.py` to use your preferred LLM backend:

```python
create_custom_inference_server(
    model="your-model",
    endpoint="your-endpoint",
    parameters={"your": "config"}
)
```

## 📈 Performance

Our framework achieves state-of-the-art performance on KernelBench:

- Faster generation times compared to traditional approaches
- Higher accuracy in generated kernel implementations
- Improved memory efficiency through optimized decoding

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

## 🔗 Links

- [KernelBench Repository](https://github.com/ScalingIntelligence/KernelBench)
- [Project Documentation](docs/)
