from dotenv import load_dotenv
from inference_utils import create_custom_inference_server
from anthropic import Anthropic
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

def query_claude(prompt, verbose=False, time_generation=False):
    """Query Claude API with the given prompt."""
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    try:
        start_time = time.time() if time_generation else None
        
        if verbose:
            print(f"\nSending prompt to Claude (first 100 chars):\n{prompt[:100]}...")
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text
        
        if time_generation:
            end_time = time.time()
            print(f"\nClaude response time: {end_time - start_time:.2f} seconds")
            
        return response_text
        
    except Exception as e:
        print(f"Error querying Claude: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test inference with custom endpoint')
    parser.add_argument('prompt_file', help='Path to the prompt file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--time-generation', action='store_true', help='Time the generation process')
    parser.add_argument('--save', action='store_true', help='Save response to file')
    parser.add_argument('--use-custom', action='store_true', help='Use custom inference endpoint instead of Claude')
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Read the prompt
    prompt = read_prompt_file(args.prompt_file)
    if prompt is None:
        return
    
    try:
        # Get response from either custom endpoint or Claude
        if args.use_custom:
            # Create custom inference function
            query_llm = create_custom_inference_server(
                verbose=args.verbose,
                time_generation=args.time_generation
            )
            response = query_llm(prompt)
        else:
            # Use Claude API
            response = query_claude(
                prompt,
                verbose=args.verbose,
                time_generation=args.time_generation
            )
            
        if response:
            print("\nResponse:", response)
            
            # Save response if requested
            if args.save:
                save_response(response, args.prompt_file)
        
    except Exception as e:
        print(f"\nError during inference: {str(e)}")

if __name__ == "__main__":
    main()
