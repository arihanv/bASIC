import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import importlib.util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KernelBenchRunner:
    def __init__(self, kernelbench_path):
        """Initialize the benchmark runner with path to KernelBench"""
        self.kernelbench_path = Path(kernelbench_path)
        self.src_path = self.kernelbench_path / "src"
        
        # Add KernelBench src to Python path
        sys.path.append(str(self.src_path))
        
        # Import KernelBench utilities
        try:
            from utils.benchmark import benchmark_model
            from utils.profiler import profile_model
            self.benchmark_model = benchmark_model
            self.profile_model = profile_model
        except ImportError as e:
            logger.error(f"Failed to import KernelBench utilities: {e}")
            raise

    def load_model(self, model_path, model_class_name="ModelNew"):
        """Load a model from a Python file"""
        try:
            spec = importlib.util.spec_from_file_location("model_module", model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the model class and create instance
            model_class = getattr(module, model_class_name)
            model = model_class()
            
            # Get input generation functions
            get_inputs = getattr(module, "get_inputs", None)
            get_init_inputs = getattr(module, "get_init_inputs", None)
            
            if not get_inputs or not get_init_inputs:
                raise AttributeError("Model file must define get_inputs() and get_init_inputs()")
            
            return model, get_inputs, get_init_inputs
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def evaluate_kernel(self, model_path, num_warmup=10, num_iters=100, profile=True):
        """Evaluate a kernel implementation"""
        logger.info(f"Evaluating kernel from {model_path}")
        
        # Load the model and reference implementation
        model, get_inputs, get_init_inputs = self.load_model(model_path)
        ref_model, _, _ = self.load_model(model_path, "Model")  # Original implementation
        
        # Move models to GPU
        model = model.cuda()
        ref_model = ref_model.cuda()
        
        # Initialize models
        init_inputs = get_init_inputs()
        if init_inputs:
            model(*init_inputs)
            ref_model(*init_inputs)
        
        # Get test inputs
        inputs = get_inputs()
        
        # Warmup
        logger.info("Warming up...")
        for _ in range(num_warmup):
            model(*inputs)
            ref_model(*inputs)
        torch.cuda.synchronize()
        
        # Benchmark
        logger.info(f"Running benchmark ({num_iters} iterations)...")
        optimized_time = self.benchmark_model(model, inputs, num_iters)
        baseline_time = self.benchmark_model(ref_model, inputs, num_iters)
        
        speedup = baseline_time / optimized_time
        logger.info(f"Results:")
        logger.info(f"- Baseline time: {baseline_time:.4f} ms")
        logger.info(f"- Optimized time: {optimized_time:.4f} ms")
        logger.info(f"- Speedup: {speedup:.2f}x")
        
        # Profile if requested
        if profile:
            logger.info("\nProfiling optimized kernel...")
            profile_results = self.profile_model(model, inputs)
            logger.info("Profile results:")
            for metric, value in profile_results.items():
                logger.info(f"- {metric}: {value}")
        
        return {
            "baseline_time": baseline_time,
            "optimized_time": optimized_time,
            "speedup": speedup,
            "profile_results": profile_results if profile else None
        }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate and profile CUDA kernels')
    parser.add_argument('--kernelbench-path', type=str, required=True,
                       help='Path to KernelBench repository')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the model file containing kernel implementation')
    parser.add_argument('--num-warmup', type=int, default=10,
                       help='Number of warmup iterations')
    parser.add_argument('--num-iters', type=int, default=100,
                       help='Number of benchmark iterations')
    parser.add_argument('--no-profile', action='store_true',
                       help='Disable kernel profiling')
    args = parser.parse_args()
    
    runner = KernelBenchRunner(args.kernelbench_path)
    runner.evaluate_kernel(
        args.model_path,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
        profile=not args.no_profile
    )

if __name__ == "__main__":
    main()
