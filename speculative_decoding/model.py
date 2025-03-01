import torch
from openai import OpenAI
from .tree import Tree
from .utils import tree_decoding, verify, parallel_tree_decoding

class SPModel:
    def __init__(self, base_url, api_key, model_name, draft_model_name=None):
        """
        Initialize a speculative decoding model.
        
        Args:
            base_url (str): Base URL for the API.
            api_key (str): API key for authentication.
            model_name (str): Name of the base model.
            draft_model_name (str, optional): Name of the draft model. If None, use the base model.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.draft_model_name = draft_model_name or model_name
        
    def _generate_completion(self, prompt, model_name, max_tokens=1, temperature=0.0, stream=False):
        """
        Generate completion for a prompt using the specified model.
        
        Args:
            prompt (str): Input prompt.
            model_name (str): Name of the model to use.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            stream (bool): Whether to stream the response.
            
        Returns:
            dict: Model outputs.
        """
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream
        )
        
        return response
    
    def _tokenize(self, text):
        """
        Tokenize text using the model's tokenizer.
        
        Args:
            text (str): Text to tokenize.
            
        Returns:
            list: List of token IDs.
        """
        # This is a placeholder. In a real implementation, we would use the model's tokenizer.
        # For the OpenAI API, we would need to use a separate tokenizer or the API's tokenize endpoint.
        return [ord(c) for c in text]  # Simple character-level tokenization for demonstration
    
    def _detokenize(self, token_ids):
        """
        Detokenize token IDs to text.
        
        Args:
            token_ids (list): List of token IDs.
            
        Returns:
            str: Detokenized text.
        """
        # This is a placeholder. In a real implementation, we would use the model's tokenizer.
        # For the OpenAI API, we would need to use a separate tokenizer or the API's detokenize endpoint.
        return ''.join([chr(token_id) for token_id in token_ids])  # Simple character-level detokenization for demonstration
    
    def draft(self, prompt, tree, max_tokens=10, temperature=0.0):
        """
        Generate draft tokens using the draft model.
        
        Args:
            prompt (str): Input prompt.
            tree (Tree): Tree structure for speculative decoding.
            max_tokens (int): Maximum number of tokens to draft.
            temperature (float): Temperature for sampling.
            
        Returns:
            Tree: Updated tree structure.
        """
        # Generate completion
        response = self._generate_completion(
            prompt=prompt,
            model_name=self.draft_model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False
        )
        
        # Extract generated text
        generated_text = response.choices[0].message.content
        
        # Tokenize generated text
        token_ids = self._tokenize(generated_text)
        
        # Update tree with draft tokens
        for i, token_id in enumerate(token_ids[:max_tokens]):
            if i == 0:
                # Add first token as child of root
                tree.add_node(0, token_id, 1.0)
            else:
                # Add subsequent tokens as children of previous tokens
                tree.add_node(i, token_id, 1.0)
        
        return tree
    
    def verify_tree(self, prompt, tree, threshold=0.5):
        """
        Verify the tree using the base model.
        
        Args:
            prompt (str): Input prompt.
            tree (Tree): Tree structure to verify.
            threshold (float): Probability threshold for accepting a token.
            
        Returns:
            list: List of accepted token IDs.
        """
        # Get all paths from root to leaf nodes
        paths = []
        for leaf_idx in tree.leaf_nodes:
            path = tree.get_path_to_node(leaf_idx)
            tokens = tree.get_tokens_from_path(path)
            paths.append((path, tokens))
        
        # Verify each path
        accepted_tokens = []
        for path, tokens in paths:
            # Generate completion with the base model
            completion_prompt = prompt + self._detokenize(tokens[:-1])  # Exclude the last token for verification
            
            response = self._generate_completion(
                prompt=completion_prompt,
                model_name=self.model_name,
                max_tokens=1,
                temperature=0.0,
                stream=False
            )
            
            # Extract generated text
            generated_text = response.choices[0].message.content
            
            # Tokenize generated text
            generated_token_ids = self._tokenize(generated_text)
            
            # Verify if the generated token matches the draft token
            if generated_token_ids and generated_token_ids[0] == tokens[-1]:
                accepted_tokens.append(tokens[-1])
        
        return accepted_tokens
    
    def spgenerate(self, prompt, max_new_tokens=20, nodes=10, threshold=0.5, max_depth=3, temperature=0.7):
        """
        Generate text using speculative tree decoding.
        
        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            nodes (int): Maximum number of nodes in the tree.
            threshold (float): Probability threshold for accepting a token.
            max_depth (int): Maximum depth of the tree.
            temperature (float): Temperature for sampling.
            
        Returns:
            str: Generated text.
        """
        # Initialize tree
        tree = Tree(nodes=nodes, device=None, threshold=threshold, max_depth=max_depth)
        
        generated_text = prompt
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Draft tokens
            tree = self.draft(generated_text, tree, max_tokens=max_depth, temperature=temperature)
            
            # Verify tokens
            accepted_tokens = self.verify_tree(generated_text, tree, threshold=threshold)
            
            if not accepted_tokens:
                # No tokens accepted, generate one token autoregressively
                response = self._generate_completion(
                    prompt=generated_text,
                    model_name=self.model_name,
                    max_tokens=1,
                    temperature=temperature,
                    stream=False
                )
                
                next_token = response.choices[0].message.content
                generated_text += next_token
                tokens_generated += 1
            else:
                # Add accepted tokens to generated text
                accepted_text = self._detokenize(accepted_tokens)
                generated_text += accepted_text
                tokens_generated += len(accepted_tokens)
            
            # Reset tree for next iteration
            tree = Tree(nodes=nodes, device=None, threshold=threshold, max_depth=max_depth)
            
            # Check if we've generated enough tokens
            if tokens_generated >= max_new_tokens:
                break
        
        return generated_text
    
    def stream_generate(self, prompt, max_new_tokens=20, nodes=10, threshold=0.5, max_depth=3, temperature=0.7):
        """
        Generate text using speculative tree decoding with streaming output.
        
        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Maximum number of new tokens to generate.
            nodes (int): Maximum number of nodes in the tree.
            threshold (float): Probability threshold for accepting a token.
            max_depth (int): Maximum depth of the tree.
            temperature (float): Temperature for sampling.
            
        Yields:
            str: Generated text chunks.
        """
        # Initialize tree
        tree = Tree(nodes=nodes, device=None, threshold=threshold, max_depth=max_depth)
        
        generated_text = prompt
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Draft tokens
            tree = self.draft(generated_text, tree, max_tokens=max_depth, temperature=temperature)
            
            # Verify tokens
            accepted_tokens = self.verify_tree(generated_text, tree, threshold=threshold)
            
            if not accepted_tokens:
                # No tokens accepted, generate one token autoregressively
                response = self._generate_completion(
                    prompt=generated_text,
                    model_name=self.model_name,
                    max_tokens=1,
                    temperature=temperature,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        next_token = chunk.choices[0].delta.content
                        generated_text += next_token
                        tokens_generated += 1
                        yield next_token
            else:
                # Add accepted tokens to generated text
                accepted_text = self._detokenize(accepted_tokens)
                generated_text += accepted_text
                tokens_generated += len(accepted_tokens)
                yield accepted_text
            
            # Reset tree for next iteration
            tree = Tree(nodes=nodes, device=None, threshold=threshold, max_depth=max_depth)
            
            # Check if we've generated enough tokens
            if tokens_generated >= max_new_tokens:
                break
