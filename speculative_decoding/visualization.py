import matplotlib.pyplot as plt
import networkx as nx

def visualize_tree(tree, output_path=None):
    """
    Visualize a tree structure and its attention mask.
    
    Args:
        tree: Tree object to visualize.
        output_path (str, optional): Path to save the visualization. If None, display the plot.
        
    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    return tree.visualize(output_path)

def create_example_tree():
    """
    Create an example tree structure matching the visualization in the paper.
    """
    # Import here to avoid circular imports
    from .tree import Tree
    
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
