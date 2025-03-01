from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import tempfile
import os
import uuid
import subprocess
import asyncio
import json
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time

app = FastAPI(title="CUDA Kernel Profiler")

# Store information about running and completed jobs
jobs: Dict[str, Dict[str, Any]] = {}

class CudaKernelRequest(BaseModel):
    request_id: str
    cuda_code: Optional[str] = None

async def compile_and_profile_cuda(cuda_code: str, request_id: str):
    """Compile, execute and profile a CUDA kernel, streaming results back."""
    
    jobs[request_id] = {
        "status": "processing",
        "logs": [],
        "metrics": {},
        "start_time": time.time(),
        "end_time": None
    }
    
    # Create a temporary directory for the job
    with tempfile.TemporaryDirectory() as temp_dir:
        cuda_file_path = os.path.join(temp_dir, f"kernel_{request_id}.cu")
        executable_path = os.path.join(temp_dir, f"kernel_{request_id}")
        
        # Write CUDA code to file
        with open(cuda_file_path, "w") as f:
            f.write(cuda_code)
        
        jobs[request_id]["logs"].append({"timestamp": time.time(), "message": "CUDA file created", "type": "info"})
        
        # Compile CUDA file
        try:
            compile_process = subprocess.run(
                ["nvcc", cuda_file_path, "-o", executable_path],
                capture_output=True,
                text=True,
                check=False
            )
            
            if compile_process.returncode != 0:
                error_message = compile_process.stderr
                jobs[request_id]["logs"].append({"timestamp": time.time(), "message": f"Compilation failed: {error_message}", "type": "error"})
                jobs[request_id]["status"] = "failed"
                return
            
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": "Compilation successful", "type": "info"})
            
            # Run basic execution test
            test_process = subprocess.run(
                [executable_path],
                capture_output=True,
                text=True,
                timeout=10,  # Limit execution time to prevent infinite loops
                check=False
            )
            
            if test_process.returncode != 0:
                error_message = test_process.stderr
                jobs[request_id]["logs"].append({"timestamp": time.time(), "message": f"Execution test failed: {error_message}", "type": "error"})
                jobs[request_id]["status"] = "failed"
                return
            
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": "Execution test successful", "type": "info"})
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": f"Output: {test_process.stdout}", "type": "output"})
            
            # Run profiling with nvprof
            profiling_process = subprocess.run(
                ["nvprof", "--metrics", "all", executable_path],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Parse profiling output
            profiling_data = profiling_process.stdout + profiling_process.stderr
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": "Profiling completed", "type": "info"})
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": profiling_data, "type": "profiling"})
            
            # Extract key metrics (this is a simplified example)
            # In a real implementation, you'd parse the nvprof output more carefully
            metrics = {}
            for line in profiling_data.split('\n'):
                if "GPU activities" in line:
                    metrics["execution_time"] = line.strip()
                if "Invocations" in line:
                    metrics["invocations"] = line.strip()
                if "SM Efficiency" in line:
                    metrics["sm_efficiency"] = line.strip()
            
            jobs[request_id]["metrics"] = metrics
            jobs[request_id]["status"] = "completed"
            
        except subprocess.TimeoutExpired:
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": "Execution timed out (>10s)", "type": "error"})
            jobs[request_id]["status"] = "failed"
        except Exception as e:
            jobs[request_id]["logs"].append({"timestamp": time.time(), "message": f"Error during processing: {str(e)}", "type": "error"})
            jobs[request_id]["status"] = "failed"
        
        jobs[request_id]["end_time"] = time.time()

async def stream_results(request_id: str):
    """Generator function to stream results back to the client."""
    last_log_index = 0
    
    while True:
        # Check if job exists
        if request_id not in jobs:
            yield json.dumps({"error": "Job not found"}) + "\n"
            break
        
        job = jobs[request_id]
        current_logs = job["logs"]
        
        # Stream any new logs
        while last_log_index < len(current_logs):
            yield json.dumps({"log": current_logs[last_log_index]}) + "\n"
            last_log_index += 1
        
        # If job is completed or failed, send final status and metrics
        if job["status"] in ["completed", "failed"]:
            yield json.dumps({
                "status": job["status"],
                "metrics": job["metrics"],
                "execution_time": job["end_time"] - job["start_time"] if job["end_time"] else None
            }) + "\n"
            break
        
        # Wait a bit before checking for new logs
        await asyncio.sleep(0.1)

@app.post("/submit-cuda")
async def submit_cuda(cuda_kernel: CudaKernelRequest, background_tasks: BackgroundTasks):
    """Submit a CUDA kernel for compilation and profiling."""
    request_id = cuda_kernel.request_id
    
    if not cuda_kernel.cuda_code:
        raise HTTPException(status_code=400, detail="CUDA code is required")
    
    # Start background task for compilation and profiling
    background_tasks.add_task(compile_and_profile_cuda, cuda_kernel.cuda_code, request_id)
    
    return {"request_id": request_id, "status": "processing"}

@app.post("/submit-cuda-file")
async def submit_cuda_file(request_id: str, cuda_file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    """Submit a CUDA file for compilation and profiling."""
    cuda_code = await cuda_file.read()
    cuda_code_str = cuda_code.decode("utf-8")
    
    # Start background task for compilation and profiling
    background_tasks.add_task(compile_and_profile_cuda, cuda_code_str, request_id)
    
    return {"request_id": request_id, "status": "processing"}

@app.get("/stream-results/{request_id}")
async def get_streaming_results(request_id: str):
    """Stream the results of a CUDA kernel compilation and profiling."""
    if request_id not in jobs:
        raise HTTPException(status_code=404, detail=f"No job found with request_id: {request_id}")
    
    return StreamingResponse(
        stream_results(request_id),
        media_type="application/x-ndjson"
    )

@app.get("/status/{request_id}")
async def get_job_status(request_id: str):
    """Get the current status of a CUDA kernel job."""
    if request_id not in jobs:
        raise HTTPException(status_code=404, detail=f"No job found with request_id: {request_id}")
    
    return {
        "request_id": request_id,
        "status": jobs[request_id]["status"],
        "logs_count": len(jobs[request_id]["logs"]),
        "metrics": jobs[request_id]["metrics"] if jobs[request_id]["status"] == "completed" else {}
    }

@app.get("/")
def read_root():
    return {
        "service": "CUDA Kernel Profiler",
        "endpoints": [
            {"path": "/submit-cuda", "method": "POST", "description": "Submit a CUDA kernel as JSON"},
            {"path": "/submit-cuda-file", "method": "POST", "description": "Submit a CUDA kernel as a file"},
            {"path": "/stream-results/{request_id}", "method": "GET", "description": "Stream results in real-time"},
            {"path": "/status/{request_id}", "method": "GET", "description": "Get the current status of a job"}
        ]
    }