import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

class CUDAKernelPreprocessor:
    """
    A class for preprocessing collected CUDA kernels.
    
    This preprocessor cleans, normalizes, and extracts features from CUDA kernel code
    to prepare it for model training.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the CUDA kernel preprocessor.
        
        Args:
            input_dir: Directory containing collected kernels
            output_dir: Directory to store preprocessed kernels
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def preprocess_all_kernels(self) -> Dict[str, int]:
        """
        Preprocess all collected kernels.
        
        Returns:
            Dictionary with statistics about preprocessed kernels
        """
        kernel_files = self._find_kernel_files()
        
        processed_count = 0
        for file_path in kernel_files:
            try:
                self.logger.info(f"Preprocessing kernel file: {file_path}")
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Process the kernel file
                cleaned_code = self._clean_kernel_code(content)
                tokenized_code = self._tokenize_code(cleaned_code)
                normalized_code = self._normalize_code(tokenized_code)
                features = self._extract_kernel_features(normalized_code)
                
                # Store processed kernel
                rel_path = os.path.relpath(file_path, self.input_dir)
                output_path = os.path.join(self.output_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                self._store_processed_kernel(output_path, normalized_code, features)
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"Error preprocessing {file_path}: {e}")
        
        stats = {
            'total_files': len(kernel_files),
            'processed_files': processed_count,
            'failed_files': len(kernel_files) - processed_count
        }
        
        self.logger.info(f"Preprocessing complete. Stats: {stats}")
        
        return stats
    
    def _find_kernel_files(self) -> List[str]:
        """
        Find all CUDA kernel files in the input directory.
        
        Returns:
            List of paths to CUDA kernel files
        """
        kernel_files = []
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith((".cu", ".cuh")):
                    kernel_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(kernel_files)} CUDA kernel files")
        
        return kernel_files
    
    def _clean_kernel_code(self, code: str) -> str:
        """
        Clean kernel code by removing comments and normalizing whitespace.
        
        Args:
            code: Raw kernel code
            
        Returns:
            Cleaned kernel code
        """
        # Remove C-style comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove C++-style comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        
        # Normalize whitespace (but preserve newlines)
        lines = code.split('\n')
        cleaned_lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
        
        return '\n'.join(cleaned_lines)
    
    def _tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize CUDA code into meaningful tokens.
        
        Args:
            code: Cleaned kernel code
            
        Returns:
            List of tokens
        """
        # Simple tokenization by whitespace and special characters
        # In a real implementation, this would use a code-specific tokenizer
        
        # Replace special characters with spaces around them
        for char in '()[]{}<>+-*/=;,.&|^~!?:':
            code = code.replace(char, f' {char} ')
        
        # Split by whitespace
        tokens = code.split()
        
        return tokens
    
    def _normalize_code(self, tokens: List[str]) -> str:
        """
        Normalize code structure and variable names.
        
        Args:
            tokens: Tokenized code
            
        Returns:
            Normalized code
        """
        # In a real implementation, this would:
        # - Standardize variable names
        # - Normalize indentation
        # - Standardize function calls
        
        # For now, just join tokens with spaces
        return ' '.join(tokens)
    
    def _extract_kernel_features(self, code: str) -> Dict[str, Any]:
        """
        Extract features from kernel code for classification.
        
        Args:
            code: Normalized kernel code
            
        Returns:
            Dictionary of kernel features
        """
        features = {
            'uses_shared_memory': '__shared__' in code,
            'uses_tensor_cores': any(x in code for x in ['wmma', 'mma']),
            'uses_atomics': any(x in code for x in ['atomicAdd', 'atomicCAS', 'atomicExch']),
            'kernel_complexity': self._estimate_complexity(code),
            'memory_access_pattern': self._classify_memory_access(code),
            'thread_organization': self._extract_thread_organization(code)
        }
        
        # Classify kernel type
        kernel_type, optimization_level = self._classify_kernel_type(code, features)
        features['kernel_type'] = kernel_type
        features['optimization_level'] = optimization_level
        
        return features
    
    def _estimate_complexity(self, code: str) -> str:
        """
        Estimate the complexity of a kernel.
        
        Args:
            code: Normalized kernel code
            
        Returns:
            Complexity level (low, medium, high)
        """
        # Count loops and conditionals as a simple proxy for complexity
        loop_count = code.count('for') + code.count('while')
        conditional_count = code.count('if') + code.count('switch')
        
        if loop_count > 5 or conditional_count > 10:
            return 'high'
        elif loop_count > 2 or conditional_count > 5:
            return 'medium'
        else:
            return 'low'
    
    def _classify_memory_access(self, code: str) -> str:
        """
        Classify the memory access pattern of a kernel.
        
        Args:
            code: Normalized kernel code
            
        Returns:
            Memory access pattern classification
        """
        if 'threadIdx.x + blockIdx.x * blockDim.x' in code:
            return 'coalesced'
        elif 'threadIdx.y + blockIdx.y * blockDim.y' in code:
            return 'strided'
        else:
            return 'unknown'
    
    def _extract_thread_organization(self, code: str) -> Dict[str, Any]:
        """
        Extract thread organization information from kernel code.
        
        Args:
            code: Normalized kernel code
            
        Returns:
            Dictionary with thread organization information
        """
        # Extract block and grid dimensions
        block_dim_pattern = r'dim3\s+block\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
        grid_dim_pattern = r'dim3\s+grid\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)'
        
        block_match = re.search(block_dim_pattern, code)
        grid_match = re.search(grid_dim_pattern, code)
        
        thread_org = {
            'uses_dynamic_parallelism': 'cudaLaunchKernel' in code,
            'uses_shared_memory': '__shared__' in code,
            'uses_cooperative_groups': 'cooperative_groups' in code
        }
        
        if block_match:
            thread_org['block_dim'] = (
                int(block_match.group(1)),
                int(block_match.group(2)),
                int(block_match.group(3))
            )
        
        if grid_match:
            thread_org['grid_dim'] = (
                int(grid_match.group(1)),
                int(grid_match.group(2)),
                int(grid_match.group(3))
            )
        
        return thread_org
    
    def _classify_kernel_type(self, code: str, features: Dict[str, Any]) -> Tuple[str, str]:
        """
        Classify kernel by its purpose and optimization level.
        
        Args:
            code: Normalized kernel code
            features: Extracted kernel features
            
        Returns:
            Tuple of (kernel_type, optimization_level)
        """
        # Kernel type classification based on patterns
        kernel_types = [
            ('matrix_multiplication', ['matrixMul', 'gemm', 'sgemm', 'dgemm']),
            ('convolution', ['conv', 'convolution']),
            ('reduction', ['reduce', 'reduction']),
            ('attention', ['attention', 'softmax']),
            ('element_wise', ['elementWise', 'element_wise']),
            ('scan', ['scan', 'prefix_sum']),
            ('sort', ['sort', 'radix_sort', 'merge_sort'])
        ]
        
        for kernel_type, patterns in kernel_types:
            if any(pattern in code for pattern in patterns):
                break
        else:
            kernel_type = 'other'
        
        # Optimization level classification
        if features['uses_tensor_cores'] or 'wmma' in code or 'mma' in code:
            optimization_level = 'high'
        elif features['uses_shared_memory'] and features['memory_access_pattern'] == 'coalesced':
            optimization_level = 'medium'
        else:
            optimization_level = 'low'
        
        return kernel_type, optimization_level
    
    def _store_processed_kernel(self, output_path: str, normalized_code: str, features: Dict[str, Any]) -> None:
        """
        Store processed kernel and its features.
        
        Args:
            output_path: Path to store the processed kernel
            normalized_code: Normalized kernel code
            features: Extracted kernel features
        """
        # Store normalized code
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(normalized_code)
        
        # Store features in a separate JSON file
        features_path = f"{output_path}.features.json"
        import json
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2)
        
        self.logger.info(f"Stored processed kernel at {output_path}")
