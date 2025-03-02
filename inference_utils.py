from openai import OpenAI
import anthropic
import time
import os

def create_custom_inference_server(verbose=False, time_generation=False):
    """
    Create a custom inference server function that uses the user's LLM endpoint
    """
    def query_custom_llm(prompt):
        client = OpenAI(
            base_url="https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
            api_key="EMPTY"
        )
        
        if verbose:
            print(f"Querying custom LLM endpoint with prompt: {prompt[:100]}...")
        
        start_time = None
        if time_generation:
            start_time = time.time()
        
        messages = [{"role": "user", "content": prompt}]
        
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            messages=messages,
            stream=True
        )
        
        # Collect the streaming output
        full_response = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                if verbose:
                    print(content, end="", flush=True)
        
        if time_generation and start_time:
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
        
        return full_response
    
    return query_custom_llm

def create_claude_inference_server(verbose=False, time_generation=False):
    """
    Create a Claude inference server function using Anthropic's API
    API key is read from ANTHROPIC_API_KEY environment variable
    """
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
        
    def query_claude(prompt):
        client = anthropic.Anthropic(api_key=api_key)
        
        if verbose:
            print(f"Querying Claude with prompt: {prompt[:100]}...")
        
        start_time = None
        if time_generation:
            start_time = time.time()
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response = message.content[0].text
        
        if verbose:
            print(response)
            
        if time_generation and start_time:
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
        
        return response
    
    return query_claude
