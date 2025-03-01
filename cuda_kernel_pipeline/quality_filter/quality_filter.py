import os
import json
import logging
import subprocess
from typing import List, Dict, Any, Optional

class KernelQualityFilter:
    """
    A class for filtering CUDA kernels based on quality metrics.
    
    This filter assesses kernel quality based on compilation success,
    code quality metrics, performance indicators, and complexity/diversity.
    """
    
    def __init__(self, kernels_dir: str, output_dir: str, nvcc_path: Optional[str] = None):
        """
        Initialize the kernel quality filter.
        
        Args:
            kernels_dir: Directory containing kernels to filter
            output_dir: Directory to store filtered kernels
            nvcc_path: Path to NVIDIA CUDA compiler (default: search in PATH)
        """
        self.kernels_dir = kernels_dir
        self.output_dir = output_dir
        self.nvcc_path = nvcc_path or 'nvcc'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def filter_kernels(self, min_quality_score: float = 0.7) -> Dict[str, int]:
        """
        Filter kernels based on quality score.
        
        Args:
            min_quality_score: Minimum quality score threshold (0.0 to 1.0)
            
        Returns:
            Dictionary with statistics about filtered kernels
        """
        kernels = self._load_kernels()
        self.logger.info(f"Loaded {len(kernels)} kernels for quality assessment")
        
        high_quality_kernels = []
        quality_scores = []
        
        for kernel in kernels:
            quality_score = self._assess_quality(kernel)
            quality_scores.append(quality_score)
            
            if quality_score >= min_quality_score:
                high_quality_kernels.append(kernel)
                self._save_filtered_kernel(kernel)
        
        stats = {
            'total_kernels': len(kernels),
            'high_quality_kernels': len(high_quality_kernels),
            'low_quality_kernels': len(kernels) - len(high_quality_kernels),
            'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'min_quality_score': min(quality_scores) if quality_scores else 0.0,
            'max_quality_score': max(quality_scores) if quality_scores else 0.0
        }
        
        self.logger.info(f"Filtering complete. Stats: {stats}")
        
        return stats
    
    def _load_kernels(self) -> List[Dict[str, Any]]:
        """
        Load kernels from the kernels directory.
        
        Returns:
            List of kernel dictionaries
        """
        kernels = []
        
        for root, _, files in os.walk(self.kernels_dir):
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
    
    def _assess_quality(self, kernel: Dict[str, Any]) -> float:
        """
        Assess the quality of a kernel.
        
        Args:
            kernel: Kernel dictionary
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Check compilation
        compilation_score = self._check_compilation(kernel)
        
        # Assess code quality
        code_quality_score = self._assess_code_quality(kernel)
        
        # Assess performance characteristics
        performance_score = self._assess_performance(kernel)
        
        # Assess complexity and uniqueness
        diversity_score = self._assess_diversity(kernel)
        
        # Weighted average of scores
        weights = [0.3, 0.2, 0.4, 0.1]
        total_score = sum(score * weight for score, weight in 
                          zip([compilation_score, code_quality_score, 
                               performance_score, diversity_score], weights))
        
        self.logger.debug(f"Quality scores for {os.path.basename(kernel['file_path'])}: "
                         f"compilation={compilation_score:.2f}, "
                         f"code_quality={code_quality_score:.2f}, "
                         f"performance={performance_score:.2f}, "
                         f"diversity={diversity_score:.2f}, "
                         f"total={total_score:.2f}")
        
        return total_score
    
    def _check_compilation(self, kernel: Dict[str, Any]) -> float:
        """
        Check if a kernel compiles successfully.
        
        Args:
            kernel: Kernel dictionary
            
        Returns:
            Compilation score (0.0 to 1.0)
        """
        # In a real implementation, this would compile the kernel with nvcc
        # For now, we'll use a placeholder based on kernel features
        
        # Check for common syntax issues
        code = kernel['code']
        syntax_issues = 0
        
        # Missing semicolons
        if ') {' in code and not ');' in code:
            syntax_issues += 1
        
        # Unbalanced braces
        if code.count('{') != code.count('}'):
            syntax_issues += 2
        
        # Unbalanced parentheses
        if code.count('(') != code.count(')'):
            syntax_issues += 2
        
        # Score based on syntax issues
        if syntax_issues == 0:
            return 1.0
        elif syntax_issues <= 2:
            return 0.5
        else:
            return 0.0
    
    def _assess_code_quality(self, kernel: Dict[str, Any]) -> float:
        """
        Assess the code quality of a kernel.
        
        Args:
            kernel: Kernel dictionary
            
        Returns:
            Code quality score (0.0 to 1.0)
        """
        code = kernel['code']
        
        # Check for comments
        comment_lines = code.count('//') + code.count('/*')
        code_lines = len(code.split('\n'))
        comment_ratio = min(1.0, comment_lines / max(1, code_lines) * 5)  # Aim for 20% comments
        
        # Check for consistent naming conventions
        naming_score = 0.0
        if '_' in code and not any(c.isupper() for c in code):
            # snake_case
            naming_score = 1.0
        elif not '_' in code and any(c.isupper() for c in code):
            # camelCase or PascalCase
            naming_score = 1.0
        else:
            # Mixed conventions
            naming_score = 0.5
        
        # Check for error handling
        error_handling_score = 0.0
        if 'cudaError' in code or 'cudaStatus' in code:
            error_handling_score = 1.0
        
        # Average the scores
        return (comment_ratio + naming_score + error_handling_score) / 3.0
    
    def _assess_performance(self, kernel: Dict[str, Any]) -> float:
        """
        Assess the performance characteristics of a kernel.
        
        Args:
            kernel: Kernel dictionary
            
        Returns:
            Performance score (0.0 to 1.0)
        """
        features = kernel['features']
        code = kernel['code']
        
        # Check for memory access patterns
        memory_score = 0.0
        if features.get('memory_access_pattern') == 'coalesced':
            memory_score = 1.0
        elif features.get('memory_access_pattern') == 'strided':
            memory_score = 0.5
        
        # Check for shared memory usage
        shared_memory_score = 1.0 if features.get('uses_shared_memory', False) else 0.0
        
        # Check for thread divergence minimization
        divergence_score = 0.0
        if 'if (threadIdx' in code:
            # Potential thread divergence
            divergence_score = 0.5
        else:
            divergence_score = 1.0
        
        # Check for register usage optimization
        register_score = 0.0
        if 'register' in code or '__restrict__' in code:
            register_score = 1.0
        
        # Average the scores
        return (memory_score + shared_memory_score + divergence_score + register_score) / 4.0
    
    def _assess_diversity(self, kernel: Dict[str, Any]) -> float:
        """
        Assess the complexity and diversity of a kernel.
        
        Args:
            kernel: Kernel dictionary
            
        Returns:
            Diversity score (0.0 to 1.0)
        """
        features = kernel['features']
        
        # Check algorithm complexity
        complexity_score = 0.0
        if features.get('kernel_complexity') == 'high':
            complexity_score = 1.0
        elif features.get('kernel_complexity') == 'medium':
            complexity_score = 0.7
        else:
            complexity_score = 0.4
        
        # Check kernel purpose diversity
        kernel_type = features.get('kernel_type', 'other')
        type_score = 0.0
        if kernel_type in ['matrix_multiplication', 'convolution', 'attention']:
            # Common kernel types
            type_score = 0.7
        elif kernel_type in ['reduction', 'scan', 'sort']:
            # Less common kernel types
            type_score = 0.9
        else:
            # Rare or other kernel types
            type_score = 1.0
        
        # Average the scores
        return (complexity_score + type_score) / 2.0
    
    def _save_filtered_kernel(self, kernel: Dict[str, Any]) -> None:
        """
        Save a filtered kernel to the output directory.
        
        Args:
            kernel: Kernel dictionary
        """
        rel_path = os.path.relpath(kernel['file_path'], self.kernels_dir)
        output_path = os.path.join(self.output_dir, rel_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save kernel code
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(kernel['code'])
        
        # Save features
        features_path = f"{output_path}.features.json"
        with open(features_path, 'w', encoding='utf-8') as f:
            json.dump(kernel['features'], f, indent=2)
        
        self.logger.info(f"Saved filtered kernel to {output_path}")
