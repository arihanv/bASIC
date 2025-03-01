import requests
import argparse
import time

def check_finetune_status(job_id, api_base_url):
    """Check the status of a finetuning job"""
    status_endpoint = f"{api_base_url}/v1/fine_tunes/{job_id}"
    
    response = requests.get(
        status_endpoint,
        headers={"Authorization": "Bearer EMPTY"}
    )
    
    if response.status_code == 200:
        status = response.json()
        return status
    else:
        print(f"Error checking status: {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Check status of finetuning job')
    parser.add_argument('job_id', type=str, help='Finetuning job ID')
    parser.add_argument('--api-base-url', type=str, 
                       default="https://api--openai-vllm--d8zwcx9rqzwl.code.run",
                       help='Base URL for the vLLM API')
    parser.add_argument('--watch', action='store_true',
                       help='Continuously watch the job status')
    args = parser.parse_args()
    
    while True:
        status = check_finetune_status(args.job_id, args.api_base_url)
        
        if status:
            print(f"\nStatus: {status['status']}")
            print(f"Progress: {status.get('progress', 'N/A')}%")
            print(f"Current epoch: {status.get('current_epoch', 'N/A')}")
            print(f"Training loss: {status.get('training_loss', 'N/A')}")
            
            if status['status'] in ['succeeded', 'failed']:
                if status['status'] == 'succeeded':
                    print(f"\nFinetuning complete! Model ID: {status['fine_tuned_model']}")
                else:
                    print(f"\nFinetuning failed: {status.get('error', 'Unknown error')}")
                break
        
        if not args.watch:
            break
            
        time.sleep(60)  # Check every minute when watching

if __name__ == "__main__":
    main()
