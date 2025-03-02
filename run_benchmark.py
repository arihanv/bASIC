import os
import sys
import argparse
import re
from datasets import load_dataset
from custom_inference import create_custom_inference_server

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

def create_prompt(ref_arch_src):
    """
    Create a prompt for the model to generate a CUDA kernel.
    """
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
{ref_arch_src}
```

Optimize the architecture named Model with custom CUDA kernels! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code!
"""
    return prompt

def main():
    parser = argparse.ArgumentParser(description='Run KernelBench on custom LLM endpoint')
    parser.add_argument('--level', type=int, default=1, help='Benchmark level (1-4)')
    parser.add_argument('--start', type=int, default=1, help='Starting problem ID')
    parser.add_argument('--end', type=int, default=10, help='Ending problem ID')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset for level {args.level}...")
    dataset = load_dataset('ScalingIntelligence/KernelBench')
    curr_level_dataset = dataset[f'level_{args.level}']
    
    # Create custom inference server
    print("Creating custom inference server...")
    inference_server = create_custom_inference_server(verbose=args.verbose, time_generation=True)
    
    results = []
    
    for problem_id in range(args.start, args.end + 1):
        try:
            print(f"\nProcessing level {args.level} problem {problem_id}")
            
            # Get problem
            curr_problem_row = curr_level_dataset.filter(lambda x: x['problem_id'] == problem_id)
            if len(curr_problem_row) == 0:
                print(f"Problem {problem_id} not found in level {args.level}")
                continue
                
            ref_arch_src = curr_problem_row['code'][0]
            problem_name = curr_problem_row['name'][0]
            
            print(f"Problem name: {problem_name}")
            
            # Generate CUDA kernel
            print("Generating CUDA kernel...")
            custom_cuda_prompt = create_prompt(ref_arch_src)
            
            # Get response from the model
            response = inference_server(custom_cuda_prompt)
            
            # Try to extract code
            custom_cuda = extract_code_block(response)
            
            if custom_cuda is None or custom_cuda.strip() == "":
                print("Failed to extract code from the model's response")
                results.append({
                    'problem_id': problem_id,
                    'problem_name': problem_name,
                    'generation_success': False
                })
                continue
            
            # Save generated kernel
            kernel_path = os.path.join(args.output_dir, f'level_{args.level}_problem_{problem_id}_kernel.py')
            with open(kernel_path, 'w') as f:
                f.write(custom_cuda)
            
            print(f"Generated kernel saved to {kernel_path}")
            
            # Save the full response
            response_path = os.path.join(args.output_dir, f'level_{args.level}_problem_{problem_id}_response.txt')
            with open(response_path, 'w') as f:
                f.write(response)
            
            results.append({
                'problem_id': problem_id,
                'problem_name': problem_name,
                'generation_success': True
            })
            
        except Exception as e:
            print(f"Error processing level {args.level} problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'problem_id': problem_id,
                'problem_name': problem_name if 'problem_name' in locals() else 'Unknown',
                'generation_success': False,
                'error': str(e)
            })
    
    # Calculate metrics
    num_problems = len(results)
    num_success = sum(1 for r in results if r['generation_success'])
    
    success_rate = num_success / num_problems if num_problems > 0 else 0
    
    # Save summary
    summary_path = os.path.join(args.output_dir, f'level_{args.level}_summary.txt')
    with open(summary_path, 'w') as f: 
        f.write(f"Level {args.level} Summary (Problems {args.start}-{args.end}):\n")
        f.write(f"Total problems: {num_problems}\n")
        f.write(f"Successful generations: {num_success} ({success_rate:.2%})\n")
        f.write("\nDetailed Results:\n")
        for r in results:
            f.write(f"Problem {r['problem_id']} ({r['problem_name']}): ")
            f.write(f"Generation Success: {r['generation_success']}")
            if not r['generation_success'] and 'error' in r:
                f.write(f", Error: {r['error']}")
            f.write("\n")
    
    print(f"\nLevel {args.level} Summary (Problems {args.start}-{args.end}):")
    print(f"Total problems: {num_problems}")
    print(f"Successful generations: {num_success} ({success_rate:.2%})")
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    main()
