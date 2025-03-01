import torch
import numpy as np
import time

def timer(func):
    """
    Timer decorator for measuring function execution time.
    
    Args:
        func: Function to time.
        
    Returns:
        Wrapped function that prints execution time.
    """
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        
        return result
    
    return wrapper

def prepare_logits_processor(
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0
):
    """
    Prepare logits processors for text generation.
    
    Args:
        temperature (float): Temperature for sampling.
        top_k (int): Number of top tokens to keep.
        top_p (float): Cumulative probability threshold for top-p sampling.
        repetition_penalty (float): Penalty for repeated tokens.
        
    Returns:
        function: Logits processor function.
    """
    processors = []
    
    # Add temperature scaling
    if temperature > 0:
        def temperature_scaling(logits):
            return logits / temperature
        processors.append(temperature_scaling)
    
    # Add top-k filtering
    if top_k > 0:
        def top_k_filtering(logits):
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
            indices_to_remove = logits < top_k_logits[-1]
            logits[indices_to_remove] = float('-inf')
            return logits
        processors.append(top_k_filtering)
    
    # Add top-p (nucleus) filtering
    if top_p < 1.0:
        def top_p_filtering(logits):
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            return logits
        processors.append(top_p_filtering)
    
    # Add repetition penalty
    if repetition_penalty > 1.0:
        def apply_repetition_penalty(logits, input_ids):
            for token_id in set(input_ids.tolist()):
                if logits[token_id] < 0:
                    logits[token_id] *= repetition_penalty
                else:
                    logits[token_id] /= repetition_penalty
            return logits
        
        def repetition_penalty_processor(logits, input_ids=None):
            if input_ids is not None:
                return apply_repetition_penalty(logits, input_ids)
            return logits
        
        processors.append(repetition_penalty_processor)
    
    # Combine all processors
    def process_logits(logits, input_ids=None):
        for processor in processors:
            if processor.__name__ == 'repetition_penalty_processor':
                logits = processor(logits, input_ids)
            else:
                logits = processor(logits)
        return logits
    
    return process_logits

def tree_decoding(tree, input_ids, model, tokenizer, max_new_tokens=20, temperature=0.7, top_k=0, top_p=0.9):
    """
    Perform tree-based speculative decoding.
    
    Args:
        tree: Tree structure for speculative decoding.
        input_ids (torch.Tensor): Input token IDs.
        model: Language model for generating tokens.
        tokenizer: Tokenizer for encoding/decoding tokens.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for sampling.
        top_k (int): Number of top tokens to keep.
        top_p (float): Cumulative probability threshold for top-p sampling.
        
    Returns:
        torch.Tensor: Generated token IDs.
    """
    device = input_ids.device
    input_len = input_ids.size(-1)
    generated_ids = input_ids.clone()
    
    # Prepare logits processor
    logits_processor = prepare_logits_processor(temperature=temperature, top_k=top_k, top_p=top_p)
    
    for _ in range(max_new_tokens // tree.max_depth + 1):
        # Initialize tree with first token
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[:, -1:, :]
            processed_logits = logits_processor(logits[0, -1, :], generated_ids[0])
            
            # Initialize tree
            tree_state = tree.initialize(processed_logits.unsqueeze(0).unsqueeze(0))
        
        # Expand tree
        for _ in range(tree.max_depth - 1):
            # Get current tree state
            tree_input_ids = torch.cat([generated_ids[0], tree_state["input_ids"]], dim=0).unsqueeze(0)
            
            # Generate next level of tree
            with torch.no_grad():
                outputs = model(input_ids=tree_input_ids)
                logits = outputs.logits[:, -tree.nodes:, :]
                
                # Process logits
                for i in range(logits.size(1)):
                    logits[0, i] = logits_processor(logits[0, i], tree_input_ids[0])
                
                # Update tree
                tree_state = tree.add(logits)
            
            # Check if tree is final
            if tree_state["is_final"]:
                break
        
        # Extract accepted tokens
        accepted_tokens = tree_state["input_ids"][:tree.nodes].tolist()
        
        if not accepted_tokens:
            # No tokens accepted, generate one token autoregressively
            next_token_logits = outputs.logits[0, -1, :]
            processed_logits = logits_processor(next_token_logits, generated_ids[0])
            next_token_id = torch.argmax(processed_logits).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
        else:
            # Add accepted tokens to generated sequence
            accepted_tokens_tensor = torch.tensor([accepted_tokens[:1]], device=device)  # Take only first token for simplicity
            generated_ids = torch.cat([generated_ids, accepted_tokens_tensor], dim=-1)
        
        # Check for end of sequence
        if generated_ids[0, -1].item() == tokenizer.eos_token_id:
            break
        
        # Reset tree for next iteration
        tree.reset()
    
    return generated_ids

def verify(draft_logits, target_logits, threshold=0.5):
    """
    Verify draft tokens against target model logits.
    
    Args:
        draft_logits (torch.Tensor): Logits from the draft model.
        target_logits (torch.Tensor): Logits from the target model.
        threshold (float): Probability threshold for accepting a token.
        
    Returns:
        list: List of verification results (True/False).
    """
    draft_probs = torch.softmax(draft_logits, dim=-1)
    target_probs = torch.softmax(target_logits, dim=-1)
    
    # Get top tokens from draft model
    top_draft_probs, top_draft_indices = torch.topk(draft_probs, k=1, dim=-1)
    
    # Check if target model agrees with draft model
    results = []
    for i in range(len(top_draft_indices)):
        token_id = top_draft_indices[i].item()
        target_prob = target_probs[i, token_id].item()
        
        # Accept if target probability is above threshold
        results.append(target_prob >= threshold)
    
    return results

def parallel_tree_decoding(tree, input_ids, model, tokenizer, max_new_tokens=20, temperature=0.7, top_k=0, top_p=0.9, num_gpus=8):
    """
    Perform tree-based speculative decoding in parallel across multiple GPUs.
    
    Args:
        tree: Tree structure for speculative decoding.
        input_ids (torch.Tensor): Input token IDs.
        model: Language model for generating tokens.
        tokenizer: Tokenizer for encoding/decoding tokens.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for sampling.
        top_k (int): Number of top tokens to keep.
        top_p (float): Cumulative probability threshold for top-p sampling.
        num_gpus (int): Number of GPUs to use.
        
    Returns:
        torch.Tensor: Generated token IDs.
    """
    # This is a simplified implementation that simulates parallel processing
    # In a real implementation, this would use torch.distributed or other parallel processing frameworks
    
    device = input_ids.device
    input_len = input_ids.size(-1)
    generated_ids = input_ids.clone()
    
    # Prepare logits processor
    logits_processor = prepare_logits_processor(temperature=temperature, top_k=top_k, top_p=top_p)
    
    # Increase tree nodes to leverage multiple GPUs
    tree.nodes = tree.nodes * num_gpus
    
    for _ in range(max_new_tokens // tree.max_depth + 1):
        # Initialize tree with first token
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits[:, -1:, :]
            processed_logits = logits_processor(logits[0, -1, :], generated_ids[0])
            
            # Initialize tree
            tree_state = tree.initialize(processed_logits.unsqueeze(0).unsqueeze(0))
        
        # Expand tree
        for _ in range(tree.max_depth - 1):
            # Get current tree state
            tree_input_ids = torch.cat([generated_ids[0], tree_state["input_ids"]], dim=0).unsqueeze(0)
            
            # Generate next level of tree
            with torch.no_grad():
                # In a real implementation, this would be distributed across GPUs
                outputs = model(input_ids=tree_input_ids)
                logits = outputs.logits[:, -tree.nodes:, :]
                
                # Process logits
                for i in range(logits.size(1)):
                    logits[0, i] = logits_processor(logits[0, i], tree_input_ids[0])
                
                # Update tree
                tree_state = tree.add(logits)
            
            # Check if tree is final
            if tree_state["is_final"]:
                break
        
        # Extract accepted tokens
        accepted_tokens = tree_state["input_ids"][:tree.nodes].tolist()
        
        if not accepted_tokens:
            # No tokens accepted, generate one token autoregressively
            next_token_logits = outputs.logits[0, -1, :]
            processed_logits = logits_processor(next_token_logits, generated_ids[0])
            next_token_id = torch.argmax(processed_logits).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
        else:
            # Add accepted tokens to generated sequence
            accepted_tokens_tensor = torch.tensor([accepted_tokens[:1]], device=device)  # Take only first token for simplicity
            generated_ids = torch.cat([generated_ids, accepted_tokens_tensor], dim=-1)
        
        # Check for end of sequence
        if generated_ids[0, -1].item() == tokenizer.eos_token_id:
            break
        
        # Reset tree for next iteration
        tree.reset()
    
    return generated_ids
