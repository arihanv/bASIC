import json
import os
from pathlib import Path
import requests
import argparse

def prepare_training_data():
    """Prepare training data in the format required by vLLM finetuning"""
    qa_pairs_path = Path(__file__).parent / "qa_pairs" / "kernelbench_qa_pairs.json"
    
    if not qa_pairs_path.exists():
        raise FileNotFoundError(f"Q&A pairs file not found at {qa_pairs_path}")
    
    with open(qa_pairs_path, 'r') as f:
        qa_pairs = json.load(f)
    
    # Format for vLLM finetuning
    training_data = []
    for pair in qa_pairs:
        # Format the prompt template
        prompt = f"""Given this CUDA kernel optimization problem:

{pair['problem_code']}

Create an optimized CUDA kernel implementation that:
1. Maximizes performance
2. Uses efficient memory access patterns
3. Properly handles all edge cases
4. Is compilable and functional

Only output the code, no explanations needed."""

        training_example = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": pair['solution_code']}
            ]
        }
        training_data.append(training_example)
    
    # Save the formatted training data
    output_path = Path(__file__).parent / "finetune_data.json"
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Prepared {len(training_data)} training examples")
    print(f"Saved to {output_path}")
    return output_path

def start_finetuning(data_path, api_base_url):
    """Start finetuning process using the vLLM API"""
    finetune_endpoint = f"{api_base_url}/v1/fine_tunes"
    
    # Read the training data
    with open(data_path, 'r') as f:
        training_data = json.load(f)
    
    # Prepare the finetuning request
    finetune_request = {
        "training_file": training_data,
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "hyperparameters": {
            "n_epochs": 3,
            "learning_rate": 1e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05
        }
    }
    
    # Start finetuning
    response = requests.post(
        finetune_endpoint,
        json=finetune_request,
        headers={"Authorization": "Bearer EMPTY"}
    )
    
    if response.status_code == 200:
        finetune_job = response.json()
        print("Finetuning started successfully!")
        print(f"Job ID: {finetune_job['id']}")
        return finetune_job
    else:
        print(f"Error starting finetuning: {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Prepare and start finetuning on KernelBench data')
    parser.add_argument('--api-base-url', type=str, 
                       default="https://api--openai-vllm--d8zwcx9rqzwl.code.run",
                       help='Base URL for the vLLM API')
    args = parser.parse_args()
    
    # Prepare training data
    print("Preparing training data...")
    data_path = prepare_training_data()
    
    # Start finetuning
    print("\nStarting finetuning process...")
    finetune_job = start_finetuning(data_path, args.api_base_url)
    
    if finetune_job:
        print("\nMonitor the finetuning progress using the job ID above.")
        print("Once complete, you can use the finetuned model by specifying its ID in your API calls.")

if __name__ == "__main__":
    main()
