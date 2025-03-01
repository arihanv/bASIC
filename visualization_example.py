import torch
from speculative_decoding.tree import Tree

def create_example_tree():
    """
    Create an example tree structure matching the visualization in the paper.
    """
    # Initialize tree
    tree = Tree(nodes=10, device=None, threshold=0.5, max_depth=5)
    
    # Set root node token
    tree.set_node_token(0, "it")
    
    # Add first level nodes
    is_node = tree.add_node(0, 1, 0.9)  # "is" node
    has_node = tree.add_node(0, 2, 0.8)  # "has" node
    
    # Set token strings for first level
    tree.set_node_token(is_node, "is")
    tree.set_node_token(has_node, "has")
    
    # Add second level nodes
    good_node = tree.add_node(is_node, 3, 0.7)  # "good" node
    a_node = tree.add_node(has_node, 4, 0.6)  # "a" node
    
    # Set token strings for second level
    tree.set_node_token(good_node, "good")
    tree.set_node_token(a_node, "a")
    
    return tree

def main():
    # Create example tree
    tree = create_example_tree()
    
    # Visualize tree and save to file
    tree.visualize(output_path="tree_visualization.png")
    
    print("Tree visualization saved to tree_visualization.png")
    
    # Print tree structure
    print("\nTree Structure:")
    print(f"Root: {tree.node_tokens[0]}")
    
    for i in range(1, tree.node_count):
        parent_idx = tree.parent[i]
        print(f"Node {i}: {tree.node_tokens[i]} (parent: {tree.node_tokens[parent_idx]})")
    
    # Print leaf nodes
    print("\nLeaf Nodes:")
    for leaf_idx in tree.leaf_nodes:
        print(f"Leaf: {tree.node_tokens[leaf_idx]}")

if __name__ == "__main__":
    main()
