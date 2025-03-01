# CUDA Kernel Profiler

A FastAPI-based service for compiling, executing, and profiling user-submitted CUDA kernels. This service allows users to test and evaluate the performance of CUDA code in real-time.

## Features

- Submit CUDA code via JSON or file upload
- Real-time streaming of compilation, execution, and profiling results
- Performance metrics including execution time, compute usage, and more
- Secure execution of arbitrary CUDA kernels in isolated environments

## Prerequisites

- NVIDIA GPU with CUDA drivers installed
- CUDA Toolkit (version 11.0+)
- Docker with NVIDIA container toolkit (recommended)

## Running with Docker

1. Build the Docker image:
```bash
docker build -t cuda-profiler .
```

2. Run the container:
```bash
docker run --gpus all -p 8000:8000 cuda-profiler
```

## Running Locally

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the FastAPI server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

### Submit CUDA Code via JSON

```
POST /submit-cuda
```

Request body:
```json
{
  "request_id": "unique-request-id",
  "cuda_code": "your CUDA code here"
}
```

### Submit CUDA Code via File Upload

```
POST /submit-cuda-file?request_id=unique-request-id
```

Form data:
- `cuda_file`: The CUDA (.cu) file to upload

### Stream Results

```
GET /stream-results/{request_id}
```

This endpoint streams real-time results as newline-delimited JSON objects.

### Check Job Status

```
GET /status/{request_id}
```

## Example CUDA Kernel

Here's a simple example of a CUDA kernel that performs vector addition:

```cuda
#include <stdio.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void) {
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    
    // Vectors and size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // Host vectors
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }
    
    // Device vectors
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    
    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // Validate (check first 5 elements)
    for (int i = 0; i < 5; ++i) {
        printf("%.2f + %.2f = %.2f\n", h_A[i], h_B[i], h_C[i]);
    }
    
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    printf("Completed successfully\n");
    return 0;
}
```

## Security Considerations

Running arbitrary CUDA code can be potentially harmful. This service implements the following security measures:

1. Code execution in isolated containers
2. Execution time limits to prevent infinite loops
3. Resource usage limits to prevent GPU resource exhaustion

For production use, consider implementing additional security measures.