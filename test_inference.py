from custom_inference import create_custom_inference_server

def main():
    # Create the custom inference server with verbose output
    server = create_custom_inference_server(verbose=True)
    
    # Test with a simple prompt
    prompt = "What is the capital of France?"
    print(f"Testing with prompt: {prompt}")
    
    # Get response from the server
    response = server(prompt)
    
    # Print the full response
    print("\nFull response:", response)

if __name__ == "__main__":
    main()
