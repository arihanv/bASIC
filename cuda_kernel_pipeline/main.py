#!/usr/bin/env python3
"""
CUDA Kernel Collection and Preprocessing Pipeline

This script orchestrates the entire pipeline for collecting, preprocessing,
filtering, and creating datasets from CUDA kernels for training an LLM
that generates optimized CUDA code.
"""

import os
import argparse
import logging
from typing import Dict, Any

from cuda_kernel_pipeline.collector.collector import CUDAKernelCollector
from cuda_kernel_pipeline.preprocessor.preprocessor import CUDAKernelPreprocessor
from cuda_kernel_pipeline.quality_filter.quality_filter import KernelQualityFilter
from cuda_kernel_pipeline.dataset_creator.dataset_creator import CUDADatasetCreator

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("cuda_pipeline.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CUDA Kernel Collection and Preprocessing Pipeline")
    
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Base directory for all pipeline outputs")
    parser.add_argument("--github-token", type=str, default=None,
                        help="GitHub API token for authentication")
    parser.add_argument("--min-stars", type=int, default=50,
                        help="Minimum number of stars for community repositories")
    parser.add_argument("--max-repos", type=int, default=100,
                        help="Maximum number of repositories to collect from")
    parser.add_argument("--min-quality", type=float, default=0.7,
                        help="Minimum quality score for kernel filtering (0.0 to 1.0)")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip the collection step (use existing data)")
    parser.add_argument("--skip-preprocessing", action="store_true",
                        help="Skip the preprocessing step (use existing data)")
    parser.add_argument("--skip-filtering", action="store_true",
                        help="Skip the filtering step (use existing data)")
    parser.add_argument("--skip-dataset-creation", action="store_true",
                        help="Skip the dataset creation step")
    
    return parser.parse_args()

def run_pipeline(args) -> Dict[str, Any]:
    """
    Run the complete CUDA kernel pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with pipeline statistics
    """
    logger = setup_logging()
    logger.info("Starting CUDA Kernel Collection and Preprocessing Pipeline")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    raw_dir = os.path.join(args.output_dir, "raw")
    preprocessed_dir = os.path.join(args.output_dir, "preprocessed")
    filtered_dir = os.path.join(args.output_dir, "filtered")
    dataset_dir = os.path.join(args.output_dir, "datasets")
    
    stats = {}
    
    # Step 1: Collect CUDA kernels
    if not args.skip_collection:
        logger.info("Step 1: Collecting CUDA kernels")
        collector = CUDAKernelCollector(
            output_dir=raw_dir,
            min_stars=args.min_stars,
            max_repos=args.max_repos,
            github_token=args.github_token
        )
        collection_stats = collector.collect_all()
        stats["collection"] = collection_stats
    else:
        logger.info("Skipping collection step")
    
    # Step 2: Preprocess kernels
    if not args.skip_preprocessing:
        logger.info("Step 2: Preprocessing CUDA kernels")
        preprocessor = CUDAKernelPreprocessor(
            input_dir=raw_dir,
            output_dir=preprocessed_dir
        )
        preprocessing_stats = preprocessor.preprocess_all_kernels()
        stats["preprocessing"] = preprocessing_stats
    else:
        logger.info("Skipping preprocessing step")
    
    # Step 3: Filter kernels by quality
    if not args.skip_filtering:
        logger.info("Step 3: Filtering CUDA kernels by quality")
        quality_filter = KernelQualityFilter(
            kernels_dir=preprocessed_dir,
            output_dir=filtered_dir
        )
        filtering_stats = quality_filter.filter_kernels(min_quality_score=args.min_quality)
        stats["filtering"] = filtering_stats
    else:
        logger.info("Skipping filtering step")
    
    # Step 4: Create training datasets
    if not args.skip_dataset_creation:
        logger.info("Step 4: Creating training datasets")
        dataset_creator = CUDADatasetCreator(
            processed_kernels_dir=filtered_dir,
            output_dir=dataset_dir
        )
        dataset_stats = dataset_creator.create_training_dataset()
        stats["dataset_creation"] = dataset_stats
    else:
        logger.info("Skipping dataset creation step")
    
    logger.info("Pipeline completed successfully")
    return stats

if __name__ == "__main__":
    args = parse_args()
    stats = run_pipeline(args)
    
    print("\nPipeline Statistics:")
    for step, step_stats in stats.items():
        print(f"\n{step.capitalize()} Statistics:")
        for key, value in step_stats.items():
            print(f"  {key}: {value}")
