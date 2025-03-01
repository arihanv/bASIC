import os
import json
import random
import logging
from typing import List, Dict, Any, Tuple

class CUDADatasetCreator:
    """
    A class for creating training datasets from preprocessed CUDA kernels.
    
    This dataset creator generates various types of training examples for
    different training scenarios, including code completion and performance optimization.
    """
    
    def __init__(self, processed_kernels_dir: str, output_dir: str, split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):
        """
        Initialize the CUDA dataset creator.
        
        Args:
            processed_kernels_dir: Directory containing preprocessed kernels
            output_dir: Directory to store the created datasets
            split_ratio: Ratio for train/validation/test splits (default: 80/10/10)
        """
        self.processed_kernels_dir = processed_kernels_dir
        self.output_dir = output_dir
        self.split_ratio = split_ratio
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_training_dataset(self) -> Dict[str, int]:
        """
        Create training dataset from processed kernels.
        
        Returns:
            Dictionary with statistics about created datasets
        """
        # Load processed kernels
        kernels = self._load_processed_kernels()
        self.logger.info(f"Loaded {len(kernels)} processed kernels")
        
        # Split into train/validation/test sets
        train, val, test = self._split_dataset(kernels)
        self.logger.info(f"Split dataset: {len(train)} train, {len(val)} validation, {len(test)} test")
        
        # Create input-output pairs for different training scenarios
        train_pairs = self._create_completion_pairs(train)
        val_pairs = self._create_completion_pairs(val)
        test_pairs = self._create_completion_pairs(test)
        
        # Save datasets
        self._save_dataset(train_pairs, os.path.join(self.output_dir, 'train_completion.jsonl'))
        self._save_dataset(val_pairs, os.path.join(self.output_dir, 'val_completion.jsonl'))
        self._save_dataset(test_pairs, os.path.join(self.output_dir, 'test_completion.jsonl'))
        
        # Create performance-annotated dataset for RL training
        train_perf = self._create_performance_annotated_dataset(train)
        val_perf = self._create_performance_annotated_dataset(val)
        
        # Save performance-annotated datasets
        self._save_dataset(train_perf, os.path.join(self.output_dir, 'train_perf.jsonl'))
        self._save_dataset(val_perf, os.path.join(self.output_dir, 'val_perf.jsonl'))
        
        stats = {
            'total_kernels': len(kernels),
            'train_kernels': len(train),
            'val_kernels': len(val),
            'test_kernels': len(test),
            'train_completion_pairs': len(train_pairs),
            'val_completion_pairs': len(val_pairs),
            'test_completion_pairs': len(test_pairs),
            'train_perf_examples': len(train_perf),
            'val_perf_examples': len(val_perf)
        }
        
        self.logger.info(f"Dataset creation complete. Stats: {stats}")
        
        return stats
    
    def _load_processed_kernels(self) -> List[Dict[str, Any]]:
        """
        Load processed kernels from the processed kernels directory.
        
        Returns:
            List of dictionaries containing kernel code and features
        """
        kernels = []
        
        for root, _, files in os.walk(self.processed_kernels_dir):
            for file in files:
                if file.endswith('.cu') or file.endswith('.cuh'):
                    file_path = os.path.join(root, file)
                    features_path = f"{file_path}.features.json"
                    
                    if os.path.exists(features_path):
                        try:
                            # Load kernel code
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code = f.read()
                            
                            # Load features
                            with open(features_path, 'r', encoding='utf-8') as f:
                                features = json.load(f)
                            
                            kernels.append({
                                'file_path': file_path,
                                'code': code,
                                'features': features
                            })
                        except Exception as e:
                            self.logger.error(f"Error loading kernel {file_path}: {e}")
        
        return kernels
    
    def _split_dataset(self, kernels: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split kernels into train/validation/test sets.
        
        Args:
            kernels: List of kernel dictionaries
            
        Returns:
            Tuple of (train, validation, test) kernel lists
        """
        # Shuffle kernels
        random.shuffle(kernels)
        
        # Calculate split indices
        train_end = int(len(kernels) * self.split_ratio[0])
        val_end = train_end + int(len(kernels) * self.split_ratio[1])
        
        # Split kernels
        train = kernels[:train_end]
        val = kernels[train_end:val_end]
        test = kernels[val_end:]
        
        return train, val, test
    
    def _create_completion_pairs(self, kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create code completion pairs for training.
        
        Args:
            kernels: List of kernel dictionaries
            
        Returns:
            List of completion pair dictionaries
        """
        pairs = []
        
        for kernel in kernels:
            code = kernel['code']
            features = kernel['features']
            
            # Create different types of completion tasks
            pairs.extend(self._generate_completion_pairs(code, features))
        
        return pairs
    
    def _generate_completion_pairs(self, code: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate different types of completion pairs from a kernel.
        
        Args:
            code: Kernel code
            features: Kernel features
            
        Returns:
            List of completion pair dictionaries
        """
        pairs = []
        
        # 1. Function signature → implementation
        signature_match = self._extract_kernel_signature(code)
        if signature_match:
            signature, implementation = signature_match
            pairs.append({
                'type': 'signature_to_implementation',
                'input': signature,
                'output': implementation,
                'features': features
            })
        
        # 2. Partial implementation → complete implementation
        if len(code) > 100:  # Only for reasonably sized kernels
            # Take first 30% of the code as input
            split_point = len(code) // 3
            partial_code = code[:split_point]
            complete_code = code
            
            pairs.append({
                'type': 'partial_to_complete',
                'input': partial_code,
                'output': complete_code,
                'features': features
            })
        
        # 3. Problem description → implementation
        # This would require extracting or generating problem descriptions
        # For now, we'll use a placeholder based on kernel features
        problem_desc = self._generate_problem_description(features)
        pairs.append({
            'type': 'description_to_implementation',
            'input': problem_desc,
            'output': code,
            'features': features
        })
        
        # 4. Unoptimized kernel → optimized kernel
        # This would require pairs of unoptimized/optimized kernels
        # For now, we'll skip this type
        
        return pairs
    
    def _extract_kernel_signature(self, code: str) -> Tuple[str, str]:
        """
        Extract kernel signature and implementation from code.
        
        Args:
            code: Kernel code
            
        Returns:
            Tuple of (signature, implementation) or None if not found
        """
        import re
        
        # Match global kernel function
        pattern = r'(__global__\s+void\s+\w+\s*\([^)]*\))\s*(\{[^}]*\})'
        match = re.search(pattern, code, re.DOTALL)
        
        if match:
            signature = match.group(1)
            implementation = match.group(2)
            return signature, implementation
        
        return None
    
    def _generate_problem_description(self, features: Dict[str, Any]) -> str:
        """
        Generate a problem description from kernel features.
        
        Args:
            features: Kernel features
            
        Returns:
            Generated problem description
        """
        kernel_type = features.get('kernel_type', 'unknown')
        optimization_level = features.get('optimization_level', 'unknown')
        uses_shared_memory = features.get('uses_shared_memory', False)
        uses_tensor_cores = features.get('uses_tensor_cores', False)
        
        description = f"Implement a CUDA kernel for {kernel_type} operation with {optimization_level} optimization. "
        
        if uses_shared_memory:
            description += "The kernel should use shared memory for better performance. "
        
        if uses_tensor_cores:
            description += "The kernel should leverage tensor cores for matrix operations. "
        
        description += "Ensure efficient memory access patterns and minimal thread divergence."
        
        return description
    
    def _create_performance_annotated_dataset(self, kernels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create dataset with performance metrics for RL training.
        
        Args:
            kernels: List of kernel dictionaries
            
        Returns:
            List of performance-annotated kernel dictionaries
        """
        annotated_kernels = []
        
        for kernel in kernels:
            # In a real implementation, we would compile and run the kernel
            # to get performance metrics using NVIDIA Nsight
            # For now, we'll use placeholder metrics based on features
            
            performance_metrics = self._generate_performance_metrics(kernel['features'])
            
            annotated_kernels.append({
                'code': kernel['code'],
                'features': kernel['features'],
                'performance': performance_metrics
            })
        
        return annotated_kernels
    
    def _generate_performance_metrics(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate placeholder performance metrics based on kernel features.
        
        Args:
            features: Kernel features
            
        Returns:
            Dictionary of performance metrics
        """
        # In a real implementation, this would run Nsight and extract metrics
        # For now, we'll generate placeholder metrics
        
        optimization_level = features.get('optimization_level', 'low')
        uses_shared_memory = features.get('uses_shared_memory', False)
        uses_tensor_cores = features.get('uses_tensor_cores', False)
        
        # Generate placeholder metrics based on features
        if optimization_level == 'high':
            throughput_base = 0.8
        elif optimization_level == 'medium':
            throughput_base = 0.5
        else:
            throughput_base = 0.3
        
        if uses_shared_memory:
            throughput_base += 0.1
        
        if uses_tensor_cores:
            throughput_base += 0.2
        
        # Add some randomness
        throughput = throughput_base + random.uniform(-0.1, 0.1)
        throughput = max(0.1, min(1.0, throughput))  # Clamp between 0.1 and 1.0
        
        metrics = {
            'throughput': throughput,
            'gpu_utilization': throughput * 0.9 + random.uniform(0, 0.1),
            'memory_bandwidth': throughput * 0.8 + random.uniform(0, 0.2),
            'compute_occupancy': throughput * 0.7 + random.uniform(0, 0.3)
        }
        
        return metrics
    
    def _save_dataset(self, data: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save dataset to a JSONL file.
        
        Args:
            data: List of data dictionaries
            output_path: Path to save the dataset
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Saved dataset with {len(data)} examples to {output_path}")
