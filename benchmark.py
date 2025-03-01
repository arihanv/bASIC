import time
import argparse
from main import generate_with_standard_model, generate_with_speculative_decoding

def benchmark(prompt, max_tokens=50, runs=5):
    """
    Benchmark the performance of standard decoding vs. speculative tree decoding.
    
    Args:
        prompt (str): Input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        runs (int): Number of runs for each method.
        
    Returns:
        dict: Benchmark results.
    """
    results = {
        "standard": {
            "times": [],
            "tokens_per_second": []
        },
        "speculative": {
            "times": [],
            "tokens_per_second": []
        },
        "parallel_speculative": {
            "times": [],
            "tokens_per_second": []
        }
    }
    
    # Benchmark standard decoding
    print(f"Benchmarking standard decoding ({runs} runs)...")
    for i in range(runs):
        start_time = time.time()
        output = generate_with_standard_model(prompt, max_tokens)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        tokens_per_second = max_tokens / elapsed_time
        
        results["standard"]["times"].append(elapsed_time)
        results["standard"]["tokens_per_second"].append(tokens_per_second)
        
        print(f"  Run {i+1}: {elapsed_time:.2f}s, {tokens_per_second:.2f} tokens/s")
    
    # Benchmark speculative tree decoding (non-parallel)
    print(f"\nBenchmarking speculative tree decoding ({runs} runs)...")
    for i in range(runs):
        start_time = time.time()
        output = generate_with_speculative_decoding(prompt, max_tokens, use_parallel=False)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        tokens_per_second = max_tokens / elapsed_time
        
        results["speculative"]["times"].append(elapsed_time)
        results["speculative"]["tokens_per_second"].append(tokens_per_second)
        
        print(f"  Run {i+1}: {elapsed_time:.2f}s, {tokens_per_second:.2f} tokens/s")
    
    # Benchmark parallel speculative tree decoding
    print(f"\nBenchmarking parallel speculative tree decoding ({runs} runs)...")
    for i in range(runs):
        start_time = time.time()
        output = generate_with_speculative_decoding(prompt, max_tokens, use_parallel=True)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        tokens_per_second = max_tokens / elapsed_time
        
        results["parallel_speculative"]["times"].append(elapsed_time)
        results["parallel_speculative"]["tokens_per_second"].append(tokens_per_second)
        
        print(f"  Run {i+1}: {elapsed_time:.2f}s, {tokens_per_second:.2f} tokens/s")
    
    # Calculate averages
    for method in results:
        avg_time = sum(results[method]["times"]) / runs
        avg_tokens_per_second = sum(results[method]["tokens_per_second"]) / runs
        
        results[method]["avg_time"] = avg_time
        results[method]["avg_tokens_per_second"] = avg_tokens_per_second
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Standard Decoding: {results['standard']['avg_time']:.2f}s, {results['standard']['avg_tokens_per_second']:.2f} tokens/s")
    print(f"Speculative Tree Decoding: {results['speculative']['avg_time']:.2f}s, {results['speculative']['avg_tokens_per_second']:.2f} tokens/s")
    print(f"Parallel Speculative Tree Decoding: {results['parallel_speculative']['avg_time']:.2f}s, {results['parallel_speculative']['avg_tokens_per_second']:.2f} tokens/s")
    
    # Calculate speedup
    speedup = results["parallel_speculative"]["avg_tokens_per_second"] / results["standard"]["avg_tokens_per_second"]
    print(f"Speedup (Parallel Speculative vs. Standard): {speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark speculative tree decoding")
    parser.add_argument("--prompt", type=str, default="What is the capital of France?", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs for each method")
    
    args = parser.parse_args()
    
    benchmark(args.prompt, args.max_tokens, args.runs)
