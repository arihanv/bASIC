import torch
import numpy as np

class Tree:
    def __init__(self, nodes=10, device=None, threshold=0.5, max_depth=10):
        """
        Initialize a tree structure for speculative decoding.
        
        Args:
            nodes (int): Maximum number of nodes in the tree.
            device: Device to use for tensor operations.
            threshold (float): Probability threshold for accepting a token.
            max_depth (int): Maximum depth of the tree.
        """
        self.nodes = nodes
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.depth = 0
        self.max_depth = max_depth
        
        # Initialize tree structure
        self.nnodes = torch.tensor(0, device=self.device)
        self.weight = torch.zeros((nodes, nodes), device=self.device)  # Adjacency matrix
        self.weight_matrix = torch.zeros((max_depth, self.nodes), device=self.device)  # Weight matrix for each depth
        self.input_ids_matrix = torch.zeros((max_depth, self.nodes), dtype=torch.long, device=self.device)  # Token IDs for each depth
        self.parents_matrix = torch.zeros((max_depth, self.nodes), dtype=torch.long, device=self.device)  # Parent node indices
        self.kv_mask = torch.zeros((self.nodes, 1), device=self.device)  # KV cache mask
        self.tri = torch.eye(self.nodes, device=self.device)  # Triangular matrix for attention
        self.rows = torch.arange(self.nodes, device=self.device)  # Row indices
        self.position_id = torch.zeros((self.nodes), dtype=torch.long, device=self.device)  # Position IDs
        self.kv_cache_mask = torch.zeros((self.nodes, self.nodes), device=self.device)  # KV cache mask
    
    def initialize(self, logits):
        """
        Initialize the tree with logits.
        
        Args:
            logits: Logits from the model.
            
        Returns:
            dict: Output dictionary with tree state.
        """
        logits_id = torch.topk(logits[0][-1], k=self.nodes, dim=-1)
        
        # Update tree state
        self.weight_matrix[self.depth] = self.weight_matrix[self.depth].copy_(logits_id[0])
        self.input_ids_matrix[self.depth] = self.input_ids_matrix[self.depth].copy_(logits_id[1])
        self.parents_matrix[0] = self.parents_matrix[0].copy_(self.rows)
        
        output_dict = {
            "input_ids": self.input_ids_matrix[0],
            "position_ids": self.position_id+1,
            "attention_mask": self.tri,
            "parent_last": self.rows,
            "is_final": False
        }
        
        self.depth = 1
        
        return output_dict
    
    def add(self, logits):
        """
        Add nodes to the tree based on logits.
        
        Args:
            logits: Logits from the model.
            
        Returns:
            dict: Output dictionary with tree state.
        """
        if self.depth >= self.max_depth:
            output_dict = {
                "input_ids": self.input_ids_matrix[:self.depth].reshape(-1),
                "position_ids": self.position_id[:self.depth*self.nodes]+1,
                "attention_mask": self.kv_cache_mask[:self.depth*self.nodes, :self.depth*self.nodes],
                "parent_last": self.rows,
                "is_final": True
            }
            return output_dict
        
        # Get top-k tokens for each node
        logits_id = torch.topk(logits[0], k=self.nodes, dim=-1)
        
        # Update tree state
        self.weight_matrix[self.depth] = self.weight_matrix[self.depth].copy_(logits_id[0])
        self.input_ids_matrix[self.depth] = self.input_ids_matrix[self.depth].copy_(logits_id[1])
        
        # Update parent indices
        for i in range(self.nodes):
            self.parents_matrix[self.depth][i] = i
        
        # Update KV cache mask
        for i in range(self.nodes):
            self.kv_cache_mask[self.depth*self.nodes+i, :(self.depth+1)*self.nodes] = 1
        
        output_dict = {
            "input_ids": self.input_ids_matrix[:self.depth+1].reshape(-1),
            "position_ids": self.position_id[:(self.depth+1)*self.nodes]+1,
            "attention_mask": self.kv_cache_mask[:(self.depth+1)*self.nodes, :(self.depth+1)*self.nodes],
            "parent_last": self.rows,
            "is_final": False
        }
        
        self.depth += 1
        
        return output_dict
    
    def get_path_to_node(self, node_idx):
        """
        Get the path from root to a specific node.
        
        Args:
            node_idx (int): Index of the target node.
            
        Returns:
            list: List of node indices from root to the target node.
        """
        path = []
        current = node_idx
        depth = self.depth - 1
        
        while depth >= 0:
            path.append(current)
            current = self.parents_matrix[depth][current].item()
            depth -= 1
        
        return path[::-1]  # Reverse to get path from root to node
    
    def get_tokens_from_path(self, path):
        """
        Get the token IDs along a path.
        
        Args:
            path (list): List of node indices.
            
        Returns:
            list: List of token IDs along the path.
        """
        tokens = []
        for i, node_idx in enumerate(path):
            if i > 0:  # Skip root node
                tokens.append(self.input_ids_matrix[i-1][node_idx].item())
        return tokens
    
    def generate_attention_mask(self, input_len, max_len):
        """
        Generate attention mask for the tree structure.
        
        Args:
            input_len (int): Length of the input sequence.
            max_len (int): Maximum length of the sequence.
            
        Returns:
            torch.Tensor: Attention mask tensor.
        """
        mask = torch.zeros((max_len, max_len), device=self.device)
        
        # Set causal mask for input sequence
        for i in range(input_len):
            mask[i, :i+1] = 1
        
        # Set mask for tree structure
        for depth in range(self.depth):
            for node_idx in range(self.nodes):
                seq_idx = input_len + depth * self.nodes + node_idx
                
                # Allow attention to input sequence
                mask[seq_idx, :input_len] = 1
                
                # Allow attention to nodes in the path
                path = self.get_path_to_node(node_idx)
                for i, path_idx in enumerate(path):
                    if i > 0:  # Skip root node
                        mask[seq_idx, input_len + (i-1) * self.nodes + path_idx] = 1
        
        return mask
    
    def reset(self):
        """
        Reset the tree to its initial state.
        """
        self.depth = 0
        self.weight_matrix.zero_()
        self.input_ids_matrix.zero_()
        self.parents_matrix.zero_()
        self.kv_cache_mask.zero_()
