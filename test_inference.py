from dotenv import load_dotenv
from inference_utils import create_custom_inference_server
import argparse
import os
import time

def read_prompt_file(file_path):
    """Read content from a prompt file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading prompt file: {str(e)}")
        return None

def save_response(response, prompt_file, output_dir="KernelBench/responses"):
    """Save the LLM response to a file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on prompt filename
    base_name = os.path.basename(prompt_file)
    output_name = f"response_{base_name}"
    output_path = os.path.join(output_dir, output_name)
    
    try:
        with open(output_path, 'w') as file:
            file.write(response)
        print(f"\nResponse saved to: {output_path}")
    except Exception as e:
        print(f"Error saving response: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test inference with custom endpoint')
    parser.add_argument('prompt_file', help='Path to the prompt file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--time-generation', action='store_true', help='Time the generation process')
    parser.add_argument('--save', action='store_true', help='Save response to file')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Read the prompt
    prompt = read_prompt_file(args.prompt_file)
    if prompt is None:
        return
    
    # Create custom inference function
    query_llm = create_custom_inference_server(
        verbose=args.verbose,
        time_generation=args.time_generation
    )
    
    try:
        # Get response from custom LLM
        start_time = time.time() if args.time_generation else None
        
        print(f"\nProcessing prompt from: {args.prompt_file}")
        response = query_llm(prompt)
        
        if args.time_generation:
            end_time = time.time()
            print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")
        
        # Save response if requested
        if args.save:
            save_response(response, args.prompt_file)
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")

if __name__ == "__main__":
    main()
