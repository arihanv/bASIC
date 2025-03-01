from inference_utils import create_custom_inference_server, create_claude_inference_server
import re
import os
from dotenv import load_dotenv

load_dotenv()

def extract_code_block(text):
    """
    Extract code block from text, even if it's not properly formatted with backticks.
    """
    # First try the standard extraction with backticks
    code_match = re.search(r"```(?:python|cpp)?\s*(.*?)```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # If that fails, look for class ModelNew definition
    model_new_match = re.search(r"class ModelNew\(.*?\):(.*?)(?:$|class|def\s+\w+\s*\([^)]*\)\s*:)", text, re.DOTALL)
    if model_new_match:
        # Extract the class and try to reconstruct a proper Python class
        class_code = model_new_match.group(0).strip()
        return f"import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n{class_code}"
    
    # If all else fails, return the entire text
    print("WARNING: Could not extract code block with standard methods. Using entire response.")
    return text

def main():
    # Parse command line arguments for inference type
    import argparse
    parser = argparse.ArgumentParser(description='Generate CUDA kernel code using LLM')
    parser.add_argument('--use-claude', action='store_true', help='Use Claude API instead of custom inference')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    parser.add_argument('--time-generation', action='store_true', help='Time the generation process')
    args = parser.parse_args()

    # Define a simple matrix multiplication problem
    problem = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    """Simple model that performs a single square matrix multiplication (C = A * B)"""
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''
    
    # Create a prompt for the model
    prompt = f"""Given this PyTorch model that performs matrix multiplication:

```python
{problem}
```

Create a new version of this model called ModelNew that uses a custom CUDA kernel for matrix multiplication. The custom implementation should:
1. Use a tiled matrix multiplication approach
2. Include proper CUDA kernel code
3. Handle the matrix dimensions N=2048
4. Be compilable and functional

Only output the code, no explanations needed."""
    
    print("Creating inference server...")
    if args.use_claude:
        inference_server = create_claude_inference_server(
            verbose=args.verbose,
            time_generation=args.time_generation
        )
    else:
        inference_server = create_custom_inference_server(
            verbose=args.verbose,
            time_generation=args.time_generation
        )
    
    print(f"Generating CUDA kernel using {'Claude' if args.use_claude else 'custom'} inference...")
    print(prompt)
    response = inference_server(prompt)
    
    # Print the full response for debugging
    print("\nFull response from the model:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    # Try to extract code
    custom_cuda = extract_code_block(response)
    
    if custom_cuda is None or custom_cuda.strip() == "":
        print("Failed to extract code from the model's response")
        return
    
    # Save generated kernel
    kernel_path = 'matrix_multiplication_kernel.py'
    with open(kernel_path, 'w') as f:
        f.write(custom_cuda)
    print(f"Generated kernel saved to {kernel_path}")
    
    # Print the extracted code
    print("\nExtracted code:")
    print("=" * 80)
    print(custom_cuda)
    print("=" * 80)
    
    print("Note: Skipping evaluation since CUDA is not available on this machine.")
    print("To evaluate the generated kernel, you would need a machine with a GPU and CUDA support.")

if __name__ == "__main__":
    main()
