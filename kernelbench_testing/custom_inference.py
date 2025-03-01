from openai import OpenAI
import time

def create_custom_inference_server(verbose=False, time_generation=False):
    """
    Create a custom inference server function that uses the user's LLM endpoint
    """
    def query_custom_llm(prompt):
        client = OpenAI(
            base_url="https://api--openai-vllm--dbh9m5jlzc5l.code.run/v1",
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
