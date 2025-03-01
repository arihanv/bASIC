#!/usr/bin/env python3

import argparse
import requests
import json
import uuid
import time
import sys
from pathlib import Path


def submit_cuda_file(server_url, file_path):
    """
    Submit a CUDA file to the server for compilation and profiling.
    
    Args:
        server_url: The base URL of the server
        file_path: Path to the CUDA file to submit
    
    Returns:
        The request_id of the submitted job
    """
    request_id = str(uuid.uuid4())
    
    with open(file_path, 'rb') as f:
        files = {'cuda_file': (Path(file_path).name, f, 'text/x-cuda')}
        response = requests.post(
            f"{server_url}/submit-cuda-file?request_id={request_id}",
            files=files
        )
    
    if response.status_code != 200:
        print(f"Error submitting file: {response.text}")
        sys.exit(1)
    
    return request_id


def submit_cuda_code(server_url, cuda_code):
    """
    Submit CUDA code as a string for compilation and profiling.
    
    Args:
        server_url: The base URL of the server
        cuda_code: The CUDA code to submit
    
    Returns:
        The request_id of the submitted job
    """
    request_id = str(uuid.uuid4())
    
    payload = {
        "request_id": request_id,
        "cuda_code": cuda_code
    }
    
    response = requests.post(
        f"{server_url}/submit-cuda",
        json=payload
    )
    
    if response.status_code != 200:
        print(f"Error submitting code: {response.text}")
        sys.exit(1)
    
    return request_id


def stream_results(server_url, request_id):
    """
    Stream the results of a CUDA compilation and profiling job in real-time.
    
    Args:
        server_url: The base URL of the server
        request_id: The request_id of the job to stream
    """
    print(f"Streaming results for job {request_id}...")
    
    with requests.get(
        f"{server_url}/stream-results/{request_id}",
        stream=True
    ) as response:
        if response.status_code != 200:
            print(f"Error streaming results: {response.text}")
            sys.exit(1)
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    
                    if "error" in data:
                        print(f"Error: {data['error']}")
                    elif "log" in data:
                        log = data["log"]
                        timestamp = time.strftime(
                            "%Y-%m-%d %H:%M:%S", 
                            time.localtime(log["timestamp"])
                        )
                        
                        if log["type"] == "info":
                            print(f"[{timestamp}] INFO: {log['message']}")
                        elif log["type"] == "error":
                            print(f"[{timestamp}] ERROR: {log['message']}")
                        elif log["type"] == "output":
                            print(f"[{timestamp}] OUTPUT:\n{log['message']}")
                        elif log["type"] == "profiling":
                            print(f"[{timestamp}] PROFILING DATA:")
                            print("-" * 80)
                            print(log["message"])
                            print("-" * 80)
                    elif "status" in data:
                        print("\nJob completed with status:", data["status"])
                        if "metrics" in data and data["metrics"]:
                            print("\nPerformance Metrics:")
                            for key, value in data["metrics"].items():
                                print(f"  {key}: {value}")
                        
                        if "execution_time" in data and data["execution_time"]:
                            print(f"\nTotal execution time: {data['execution_time']:.2f} seconds")
                
                except json.JSONDecodeError:
                    print(f"Invalid JSON data: {line}")


def main():
    parser = argparse.ArgumentParser(description="CUDA Kernel Profiler Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Submit CUDA file
    file_parser = subparsers.add_parser("file", help="Submit a CUDA file")
    file_parser.add_argument("file_path", help="Path to the CUDA file")
    
    # Submit CUDA code from stdin or argument
    code_parser = subparsers.add_parser("code", help="Submit CUDA code")
    code_parser.add_argument("--code", help="CUDA code (if not provided, read from stdin)")
    
    # Stream existing job
    stream_parser = subparsers.add_parser("stream", help="Stream results for an existing job")
    stream_parser.add_argument("request_id", help="Request ID to stream")
    
    args = parser.parse_args()
    
    if args.command == "file":
        request_id = submit_cuda_file(args.server, args.file_path)
        stream_results(args.server, request_id)
    
    elif args.command == "code":
        if args.code:
            cuda_code = args.code
        else:
            print("Enter CUDA code (Ctrl+D to finish):")
            cuda_code = sys.stdin.read()
        
        request_id = submit_cuda_code(args.server, cuda_code)
        stream_results(args.server, request_id)
    
    elif args.command == "stream":
        stream_results(args.server, args.request_id)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 