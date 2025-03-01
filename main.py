from openai import OpenAI
from speculative_decoding.model import SPModel
from speculative_decoding.parallel_model import ParallelSPModel
from speculative_decoding.tree import Tree

# Initialize client
client = OpenAI(
    base_url="https://api--openai-vllm--dbh9m5jlzc5l.code.run/v1",
    api_key="EMPTY"
)

# Initialize speculative decoding model
sp_model = SPModel(
    base_url="https://api--openai-vllm--dbh9m5jlzc5l.code.run/v1",
    api_key="EMPTY",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)

# Initialize parallel speculative decoding model
parallel_sp_model = ParallelSPModel(
    base_url="https://api--openai-vllm--dbh9m5jlzc5l.code.run/v1",
    api_key="EMPTY",
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    num_gpus=8
)

def generate_with_standard_model(prompt, max_tokens=20):
    """
    Generate text using the standard model.
    
    Args:
        prompt (str): Input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        
    Returns:
        str: Generated text.
    """
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        stream=True,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    
    generated_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            generated_text += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    return generated_text

def generate_with_speculative_decoding(prompt, max_tokens=20, use_parallel=True):
    """
    Generate text using speculative tree decoding.
    
    Args:
        prompt (str): Input prompt.
        max_tokens (int): Maximum number of new tokens to generate.
        use_parallel (bool): Whether to use parallel processing.
        
    Returns:
        str: Generated text.
    """
    if use_parallel:
        return parallel_sp_model.parallel_spgenerate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            nodes=20,
            threshold=0.5,
            max_depth=5,
            temperature=0.7
        )
    else:
        return sp_model.spgenerate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            nodes=10,
            threshold=0.5,
            max_depth=3,
            temperature=0.7
        )

def generate_with_streaming_speculative_decoding(prompt, max_tokens=20, use_parallel=True):
    """
    Generate text using speculative tree decoding with streaming output.
    
    Args:
        prompt (str): Input prompt.
        max_tokens (int): Maximum number of new tokens to generate.
        use_parallel (bool): Whether to use parallel processing.
    """
    if use_parallel:
        for chunk in parallel_sp_model.parallel_stream_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            nodes=20,
            threshold=0.5,
            max_depth=5,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)
    else:
        for chunk in sp_model.stream_generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            nodes=10,
            threshold=0.5,
            max_depth=3,
            temperature=0.7
        ):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    
    print("Generating with standard model:")
    standard_output = generate_with_standard_model(prompt)
    print("\n")
    
    print("Generating with speculative tree decoding:")
    speculative_output = generate_with_speculative_decoding(prompt)
    print(speculative_output)
    print("\n")
    
    print("Generating with streaming speculative tree decoding:")
    generate_with_streaming_speculative_decoding(prompt)
    print("\n")
