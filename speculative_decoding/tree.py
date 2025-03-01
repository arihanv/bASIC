import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        
        # For tracking tree structure
        self.node_count = 1  # Root node is always present
        self.leaf_nodes = [0]  # Start with root as the only leaf
        self.token_ids = [0] * nodes  # Token IDs for each node
        self.token_probs = [0.0] * nodes  # Probabilities for each node
        self.children = [[] for _ in range(nodes)]  # Child node indices for each node
        self.parent = [0] * nodes  # Parent node index for each node
        self.node_tokens = [""] * nodes  # Actual token strings for visualization
    
    def add_node(self, parent_idx, token_id, prob):
        """
        Add a new node to the tree.
        
        Args:
            parent_idx (int): Index of the parent node.
            token_id (int): Token ID for the new node.
            prob (float): Probability of the token.
            
        Returns:
            int: Index of the new node.
        """
        if self.node_count >= self.nodes:
            return -1  # Tree is full
        
        node_idx = self.node_count
        self.node_count += 1
        
        # Update tree structure
        self.weight[parent_idx, node_idx] = 1
        self.parent[node_idx] = parent_idx
        self.children[parent_idx].append(node_idx)
        self.token_ids[node_idx] = token_id
        self.token_probs[node_idx] = prob
        
        # Update leaf nodes
        if parent_idx in self.leaf_nodes:
            self.leaf_nodes.remove(parent_idx)
        self.leaf_nodes.append(node_idx)
        
        return node_idx
    
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
        
        # Reset tracking structures
        self.node_count = 1  # Root node is always present
        self.leaf_nodes = [0]  # Start with root as the only leaf
        self.token_ids = [0] * self.nodes  # Token IDs for each node
        self.token_probs = [0.0] * self.nodes  # Probabilities for each node
        self.children = [[] for _ in range(self.nodes)]  # Child node indices for each node
        self.parent = [0] * self.nodes  # Parent node index for each node
        self.node_tokens = [""] * self.nodes  # Actual token strings for visualization
    
    def set_node_token(self, node_idx, token_str):
        """
        Set the token string for a node (for visualization).
        
        Args:
            node_idx (int): Index of the node.
            token_str (str): Token string.
        """
        if 0 <= node_idx < self.nodes:
            self.node_tokens[node_idx] = token_str
    
    def visualize(self, output_path=None):
        """
        Visualize the tree structure and attention mask.
        
        Args:
            output_path (str, optional): Path to save the visualization. If None, display the plot.
            
        Returns:
            matplotlib.figure.Figure: The figure object.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(self.node_count):
            G.add_node(i, label=self.node_tokens[i] if self.node_tokens[i] else f"Node {i}")
        
        # Add edges
        for i in range(1, self.node_count):
            G.add_edge(self.parent[i], i)
        
        # Draw the tree
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
        labels = {node: G.nodes[node]["label"] for node in G.nodes()}
        
        nx.draw(G, pos, ax=ax1, with_labels=True, labels=labels, 
                node_color="lightgreen", node_size=2000, 
                font_size=10, font_weight="bold", 
                edge_color="brown", width=2, arrows=True)
        
        ax1.set_title("(a) Draft tree")
        
        # Draw the attention mask
        if self.node_count > 1:
            # Create attention mask for visualization
            tokens = [self.node_tokens[i] for i in range(self.node_count) if self.node_tokens[i]]
            if not tokens:
                tokens = [f"Node {i}" for i in range(self.node_count)]
            
            mask = np.zeros((len(tokens), len(tokens)))
            
            # Set mask values based on tree structure
            for i in range(len(tokens)):
                for j in range(i + 1):
                    # Check if j is in the path to i
                    node_i = i
                    path = [node_i]
                    while node_i > 0:
                        node_i = self.parent[node_i]
                        path.append(node_i)
                    
                    if j in path:
                        mask[i, j] = 1
            
            # Plot the mask
            im = ax2.imshow(mask, cmap="Greens", aspect="equal")
            
            # Set ticks and labels
            ax2.set_xticks(np.arange(len(tokens)))
            ax2.set_yticks(np.arange(len(tokens)))
            ax2.set_xticklabels(tokens)
            ax2.set_yticklabels(tokens)
            
            # Rotate the x labels for better readability
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            ax2.set_title("(b) Tree attention mask")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close(fig)
        else:
            plt.show()
        
        return fig
