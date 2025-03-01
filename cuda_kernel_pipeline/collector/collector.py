import os
import re
import subprocess
import logging
from typing import List, Dict, Any, Optional
from github import Github, Repository

class CUDAKernelCollector:
    """
    A class for collecting CUDA kernels from GitHub repositories.
    
    This collector searches for high-quality CUDA kernel implementations
    in both official NVIDIA repositories and community repositories.
    """
    
    def __init__(self, 
                 output_dir: str, 
                 min_stars: int = 50, 
                 max_repos: int = 100,
                 github_token: Optional[str] = None):
        """
        Initialize the CUDA kernel collector.
        
        Args:
            output_dir: Directory to store collected kernels
            min_stars: Minimum number of stars for community repositories
            max_repos: Maximum number of repositories to collect from
            github_token: GitHub API token for authentication
        """
        self.output_dir = output_dir
        self.min_stars = min_stars
        self.max_repos = max_repos
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize GitHub API client
        self.g = Github(self.github_token) if self.github_token else Github()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def collect_from_official_repos(self) -> List[str]:
        """
        Collect kernels from NVIDIA's official repositories.
        
        Returns:
            List of paths to collected kernel files
        """
        official_repos = [
            "NVIDIA/cuda-samples",
            "NVIDIA/cutlass",
            "NVIDIA/FasterTransformer",
            "NVIDIA/Megatron-LM",
            "NVIDIA/cccl"
        ]
        
        collected_files = []
        for repo_name in official_repos:
            self.logger.info(f"Collecting kernels from {repo_name}")
            files = self._clone_and_extract_kernels(repo_name)
            collected_files.extend(files)
            
        return collected_files
    
    def search_community_repos(self) -> List[str]:
        """
        Search for high-quality community repositories with CUDA kernels.
        
        Returns:
            List of paths to collected kernel files
        """
        search_queries = [
            "cuda kernel language:cuda stars:>50",
            "topic:cuda-kernels stars:>50",
            "topic:gpu-acceleration language:cuda stars:>50",
            "tensor cores cuda stars:>50"
        ]
        
        collected_files = []
        for query in search_queries:
            self.logger.info(f"Searching repositories with query: {query}")
            repos = self.g.search_repositories(query=query)
            
            count = 0
            for repo in repos:
                if count >= self.max_repos:
                    break
                    
                if repo.stargazers_count >= self.min_stars:
                    self.logger.info(f"Found repository: {repo.full_name} with {repo.stargazers_count} stars")
                    files = self._clone_and_extract_kernels(repo.full_name)
                    collected_files.extend(files)
                    count += 1
        
        return collected_files
    
    def _clone_and_extract_kernels(self, repo_name: str) -> List[str]:
        """
        Clone repository and extract CUDA kernel files.
        
        Args:
            repo_name: Full name of the repository (owner/repo)
            
        Returns:
            List of paths to extracted kernel files
        """
        repo_dir = os.path.join(self.output_dir, repo_name.replace("/", "_"))
        
        # Skip if already cloned
        if os.path.exists(repo_dir):
            self.logger.info(f"Repository {repo_name} already cloned, skipping")
            return self._find_cuda_files(repo_dir)
        
        os.makedirs(repo_dir, exist_ok=True)
        
        # Clone repository
        try:
            self.logger.info(f"Cloning repository {repo_name}")
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo_name}.git", repo_dir],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone repository {repo_name}: {e}")
            return []
        
        # Find CUDA files
        return self._find_cuda_files(repo_dir)
    
    def _find_cuda_files(self, repo_dir: str) -> List[str]:
        """
        Find CUDA kernel files in a repository.
        
        Args:
            repo_dir: Path to the repository directory
            
        Returns:
            List of paths to CUDA files
        """
        cuda_files = []
        
        for root, _, files in os.walk(repo_dir):
            for file in files:
                if file.endswith((".cu", ".cuh")):
                    cuda_files.append(os.path.join(root, file))
        
        self.logger.info(f"Found {len(cuda_files)} CUDA files in {repo_dir}")
        
        # Process and store kernel metadata
        for file_path in cuda_files:
            self._extract_kernel_metadata(file_path)
        
        return cuda_files
    
    def _extract_kernel_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from CUDA kernel file.
        
        Args:
            file_path: Path to the CUDA file
            
        Returns:
            Dictionary containing kernel metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return {}
        
        # Extract kernel functions
        kernel_pattern = r'__global__\s+void\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}'
        kernels = re.findall(kernel_pattern, content, re.DOTALL)
        
        # Store metadata
        metadata = {
            'file_path': file_path,
            'kernels': kernels,
            'size': len(content),
            'kernel_count': len(kernels)
        }
        
        if kernels:
            relative_path = os.path.relpath(file_path, self.output_dir)
            self._store_kernel_info(relative_path, kernels, content)
            
        return metadata
    
    def _store_kernel_info(self, file_path: str, kernels: List[str], content: str) -> None:
        """
        Store kernel information in database or file system.
        
        Args:
            file_path: Relative path to the kernel file
            kernels: List of kernel function names
            content: Content of the kernel file
        """
        # For now, just log the information
        self.logger.info(f"Storing kernel info for {file_path} with {len(kernels)} kernels")
        
        # In a real implementation, this would store the data in a database
        # such as SQLite or MongoDB
        
    def collect_all(self) -> Dict[str, int]:
        """
        Collect kernels from all sources.
        
        Returns:
            Dictionary with statistics about collected kernels
        """
        official_files = self.collect_from_official_repos()
        community_files = self.search_community_repos()
        
        stats = {
            'official_repos': len(set([os.path.dirname(f) for f in official_files])),
            'community_repos': len(set([os.path.dirname(f) for f in community_files])),
            'official_files': len(official_files),
            'community_files': len(community_files),
            'total_files': len(official_files) + len(community_files)
        }
        
        self.logger.info(f"Collection complete. Stats: {stats}")
        
        return stats
