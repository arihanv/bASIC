from custom_inference import create_custom_inference_server
import re

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
    prompt = f"""You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.

You have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.

Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is:

```
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y
```

The example new arch with custom CUDA kernels looks like this:
```
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

# Define the CUDA kernel
cuda_source = '''
extern "C" __global__ void add_kernel(float* x, float* y, float* out, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        out[idx] = x[idx] + y[idx];
    }}
}}
'''

# Compile the CUDA kernel
module_path = os.path.dirname(os.path.abspath(__file__))
add_op = load(name="add_op",
              sources=[],
              extra_cuda_cflags=[],
              extra_cflags=[],
              verbose=True)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Get tensor dimensions
        n = x.numel()
        
        # Allocate output tensor
        out = torch.empty_like(x)
        
        # Launch CUDA kernel
        threads_per_block = 1024
        blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
        
        add_op.add_kernel(
            blocks_per_grid, threads_per_block,
            x.contiguous().data_ptr(), y.contiguous().data_ptr(),
            out.data_ptr(), n
        )
        
        return out
```

You are given the following architecture:

```
{problem}
```

Optimize the architecture named Model with custom CUDA kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""
    
    print("Creating custom inference server...")
    inference_server = create_custom_inference_server(verbose=True, time_generation=True)
    
    print("Generating CUDA kernel...")
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
