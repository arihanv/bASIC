# KernelBench Q&A Pairs

This file contains Q&A pairs from the KernelBench leaderboard, where the question is the problem statement and the answer is the rank 1 solution.

## Level 1: 1_Square_matrix_multiplication_

**Speedup**: 0.17

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p1.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/fa3e10d50ee22fd656c8e4767b3f5157.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single square matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        return torch.matmul(A, B)

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 32

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < N && (tile * TILE_SIZE + tx) < N)
            shared_A[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            shared_A[ty][tx] = 0.0f;

        if (col < N && (tile * TILE_SIZE + ty) < N)
            shared_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            shared_B[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matrix_multiply_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matrix_multiply_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matrix_multiply_cuda(A, B)
```

## Level 1: 2_Standard_matrix_multiplication_

**Speedup**: 0.17

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p2.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d5e8b955452b19c73a2acc9f8372d306.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                            const int M, const int N, const int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within block
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    // Each thread computes one element of the block sub-matrix
    float sum = 0.0f;
    
    // Loop over all sub-matrices of A and B required to compute block sub-matrix
    for (int m = 0; m < K; m += 16) {
        // Load sub-matrices from global memory to shared memory
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];
        
        if ((blockRow * 16 + row < M) && (m + col < K))
            As[row][col] = A[(blockRow * 16 + row) * K + m + col];
        else
            As[row][col] = 0.0f;
            
        if ((m + row < K) && (blockCol * 16 + col < N))
            Bs[row][col] = B[(m + row) * N + blockCol * 16 + col];
        else
            Bs[row][col] = 0.0f;
            
        __syncthreads();
        
        // Compute partial sum
        #pragma unroll
        for (int k = 0; k < 16; k++)
            sum += As[row][k] * Bs[k][col];
            
        __syncthreads();
    }
    
    // Write result to global memory
    if ((blockRow * 16 + row < M) && (blockCol * 16 + col < N))
        C[(blockRow * 16 + row) * N + blockCol * 16 + col] = sum;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (M + 15) / 16);
    
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A.cuda(), B.cuda())
```

## Level 1: 3_Batched_matrix_multiplication

**Speedup**: 0.18

**Model**: gpt-o1

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p3.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d0c502aa627e20a3de8e80dbdc8d2ef0.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        return torch.bmm(A, B)

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the C++ function signature
batched_matmul_cpp_source = '''
torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B);
'''

# Define the CUDA kernel and the interface function
batched_matmul_cuda_source = '''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void batched_matrix_multiply_kernel(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int batch_size,
    int M, int K, int N) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N dimension

    if (batch < batch_size && row < M && col < N) {
        float value = 0.0f;
        for (int i = 0; i < K; ++i) {
            float a_element = A[batch * M * K + row * K + i];
            float b_element = B[batch * K * N + i * N + col];
            value += a_element * b_element;
        }
        C[batch * M * N + row * N + col] = value;
    }
}

torch::Tensor batched_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check input tensors are on the same device
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be CUDA tensors.");
    }

    // Ensure input tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    // Ensure that the batch sizes and inner dimensions match
    if (batch_size != B.size(0) || K != B.size(1)) {
        throw std::invalid_argument("Input tensor dimensions do not match for batched matrix multiplication.");
    }

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::zeros({batch_size, M, N}, options);

    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE,
                batch_size);

    batched_matrix_multiply_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N);

    return C;
}
'''

# Compile the inline CUDA code for batched matrix multiplication
batched_matmul = load_inline(
    name='batched_matmul',
    cpp_sources=batched_matmul_cpp_source,
    cuda_sources=batched_matmul_cuda_source,
    functions=['batched_matmul_cuda'],
    extra_cuda_cflags=['--restrict'],
    verbose=True
)

# Define the new model that uses the custom CUDA kernel
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.batched_matmul = batched_matmul

    def forward(self, A, B):
        return self.batched_matmul.batched_matmul_cuda(A, B)
```

## Level 1: 4_Matrix_vector_multiplication_

**Speedup**: 0.89

**Model**: gpt-o1

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p4.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/ace75607eb636186cc05385c61d181c8.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)

M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel and function for matrix-vector multiplication
matvec_mul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matvec_mul_kernel(const float* A, const float* B, float* C, int M, int K) {
    int row = blockIdx.x;  // Each block computes one row
    int tid = threadIdx.x;

    extern __shared__ float shared_sum[];  // Shared memory for partial sums

    // Each thread computes partial sum over parts of K
    float sum = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += A[row * K + i] * B[i];
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        C[row] = shared_sum[0];
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check device
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    // Check dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.size(0) == K && B.size(1) == 1, "B must have size K x 1");

    // Allocate output tensor
    auto C = torch::empty({M, 1}, A.options());

    // Launch kernel
    int threads = 256;
    int blocks = M;
    size_t shared_mem_size = threads * sizeof(float);

    matvec_mul_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K);

    return C;
}
"""

matvec_mul_cpp_source = """
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix-vector multiplication
matvec_mul = load_inline(
    name='matvec_mul',
    cpp_sources=matvec_mul_cpp_source,
    cuda_sources=matvec_mul_source,
    functions=['matvec_mul_cuda'],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matvec_mul = matvec_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matvec_mul.matvec_mul_cuda(A, B)
```

## Level 1: 5_Matrix_scalar_multiplication

**Speedup**: 0.69

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p5.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/5a8524b63a79d7dfb2085a8f82597b96.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        return A * s

M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matrix_scalar_mul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void matrix_scalar_mul_kernel(const float* A, float s, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * s;
    }
}

torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s) {
    auto size = A.numel();
    auto C = torch::zeros_like(A);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    matrix_scalar_mul_kernel<<<num_blocks, block_size>>>(
        A.data_ptr<float>(), 
        s,
        C.data_ptr<float>(), 
        size
    );

    return C;
}
"""

matrix_scalar_mul_cpp_source = "torch::Tensor matrix_scalar_mul_cuda(torch::Tensor A, float s);"

matrix_scalar_mul = load_inline(
    name='matrix_scalar_mul',
    cpp_sources=matrix_scalar_mul_cpp_source,
    cuda_sources=matrix_scalar_mul_source,
    functions=['matrix_scalar_mul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matrix_scalar_mul = matrix_scalar_mul

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        return self.matrix_scalar_mul.matrix_scalar_mul_cuda(A.cuda(), s)
```

## Level 1: 6_Matmul_with_large_K_dimension_

**Speedup**: 0.16

**Model**: gpt-o1

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p6.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/5a11f55d9e36f2b68b5774c540201de9.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        return torch.matmul(A, B)

M = 256
N = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication with shared memory optimization
matrix_mul_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Each thread computes one element of C
    float Cvalue = 0;

    // Thread row and column within the block
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Row and column indices of the element in C
    int rowC = blockRow * BLOCK_SIZE + row;
    int colC = blockCol * BLOCK_SIZE + col;

    // Loop over the tiles of K dimension
    for (int m = 0; m < (K + BLOCK_SIZE -1)/BLOCK_SIZE; ++m) {
        // Shared memory for A and B tiles
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Global indices for A and B
        int rowA = rowC;
        int colA = m * BLOCK_SIZE + col;
        int rowB = m * BLOCK_SIZE + row;
        int colB = colC;

        // Load A and B into shared memory
        As[row][col] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
        Bs[row][col] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;

        __syncthreads();

        // Multiply the tiles together
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }
        __syncthreads();
    }

    // Write the result to output if within bounds
    if (rowC < M && colC < N)
        C[rowC * N + colC] = Cvalue;
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const auto M = A.size(0);
    const auto K = A.size(1);
    const auto N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((N + BLOCK_SIZE -1)/BLOCK_SIZE, (M + BLOCK_SIZE -1)/BLOCK_SIZE);

    // Launch the kernel
    MatMulKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}
"""

matrix_mul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code for matrix multiplication
matrix_mul = load_inline(
    name='matmul_cuda',
    cpp_sources=matrix_mul_cpp_source,
    cuda_sources=matrix_mul_source,
    functions=['matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matrix_mul

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A.contiguous(), B.contiguous())
```

## Level 1: 7_Matmul_with_small_K_dimension_

**Speedup**: 0.39

**Model**: gpt-o1

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p7.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/1143f5a65506760015e6c3f06a283f87.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)

M = 16384
N = 16384
K = 32

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for matrix multiplication
matmul_kernel_code = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // Block size
    const int TILE_SIZE = 16;

    // Block row and column
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread row and column within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Row index of C and A
    int row = by * TILE_SIZE + ty;
    // Column index of C and B
    int col = bx * TILE_SIZE + tx;

    // Accumulate result
    float value = 0.0f;

    // Loop over tiles of K dimension
    for (int m = 0; m < (K + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Shared memory for A and B tiles
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load tile of A into shared memory
        if (row < M && (m * TILE_SIZE + tx) < K) {
            As[ty][tx] = A[row * K + m * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile of B into shared memory
        if (col < N && (m * TILE_SIZE + ty) < K) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().device(A.device()).dtype(A.dtype());
    auto C = torch::zeros({M, N}, options);

    const int TILE_SIZE = 16;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,
        N,
        K);

    return C;
}
"""

matmul_cpp_code = r"""
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# Compile the inline CUDA code for matrix multiplication
matmul_extension = load_inline(
    name='matmul_extension',
    cpp_sources=matmul_cpp_code,
    cuda_sources=matmul_kernel_code,
    functions=['matmul_cuda'],
    verbose=False,
    extra_cuda_cflags=['--use_fast_math']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_extension.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda(A.contiguous(), B.contiguous())
```

## Level 1: 8_Matmul_with_irregular_shapes_

**Speedup**: 0.22

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p8.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/175f44b0ad194769197ecf985280ba71.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        return torch.matmul(A, B)

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile sizes for shared memory
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
                             const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K-1)/TILE_SIZE + 1; ++t) {
        if (row < M && t*TILE_SIZE + tx < K)
            As[ty][tx] = A[row*K + t*TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (t*TILE_SIZE + ty < K && col < N)
            Bs[ty][tx] = B[(t*TILE_SIZE + ty)*N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row*N + col] = sum;
    }
}

torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    
    auto c = torch::zeros({M, N}, a.options());
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);
                   
    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        M, N, K);
        
    return c;
}
"""

matmul_cpp_source = """
torch::Tensor matmul_cuda(torch::Tensor a, torch::Tensor b);
"""

matmul_cuda = load_inline(
    name='matmul_cuda',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=['matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_cuda = matmul_cuda
        
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda.matmul_cuda(A.cuda(), B.cuda())
```

## Level 1: 9_Tall_skinny_matrix_multiplication_

**Speedup**: 0.61

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p9.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/bfe7cc072b506a9a376bb0b68acdf936.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) where one of the matrices is tall and skinny (M >> N or N >> M)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): Input matrix of shape (M, K) or (K, M) where M >> N or N >> M.
            B (torch.Tensor): Input matrix of shape (K, N) or (N, K) where M >> N or N >> M.

        Returns:
            torch.Tensor: Output matrix of shape (M, N) or (N, M)
        """
        return torch.matmul(A, B)

M = 16384
N = 16

def get_inputs():
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

matmul_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for shared memory
#define TILE_SIZE 16

__global__ void tall_skinny_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int M, const int N) {

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int i = 0; i < N; i += TILE_SIZE) {
        if (row < M && (i + tx) < N)
            As[ty][tx] = A[row * N + (i + tx)];
        else
            As[ty][tx] = 0.0f;
            
        if ((i + ty) < N && col < M)
            Bs[ty][tx] = B[(i + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    
    if (row < M && col < M) {
        C[row * M + col] = sum;
    }
}

torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int M = A.size(0);
    const int N = A.size(1);
    
    auto C = torch::zeros({M, M}, A.options());
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((M + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);
                   
    tall_skinny_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N);
        
    return C;
}
"""

matmul_cpp_source = """
torch::Tensor tall_skinny_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

tall_skinny_matmul = load_inline(
    name='tall_skinny_matmul',
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_source,
    functions=['tall_skinny_matmul_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul = tall_skinny_matmul
        
    def forward(self, A, B):
        return self.matmul.tall_skinny_matmul_cuda(A, B)
```

## Level 1: 10_3D_tensor_matrix_multiplication

**Speedup**: 0.20

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p10.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/8f8e6d114217c821c325b7d98d924fde.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs 3D tensor-matrix multiplication.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        """
        Performs 3D tensor-matrix multiplication.

        Args:
            A (torch.Tensor): Input 3D tensor of shape (N, M, K).
            B (torch.Tensor): Input matrix of shape (K, L).

        Returns:
            torch.Tensor: Output tensor of shape (N, M, L), resulting from the multiplication of A and B along the last dimension of A.
        """
        return torch.matmul(A, B)

N = 16
M = 1024
K = 2048
L = 768

def get_inputs():
    A = torch.randn(N, M, K)
    B = torch.randn(K, L)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

tensor_matmul_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for 3D tensor-matrix multiplication
__global__ void tensor_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B, 
    float* __restrict__ C,
    const int N, const int M, const int K, const int L) {

    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;

    // Thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Block size
    const int BLOCK_SIZE = 16;

    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Global indices
    const int row = by * BLOCK_SIZE + ty;
    const int col = bx * BLOCK_SIZE + tx;
    const int batch = bz;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile from A into shared memory
        if (row < M && (t * BLOCK_SIZE + tx) < K && batch < N) {
            As[ty][tx] = A[batch * M * K + row * K + t * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        if ((t * BLOCK_SIZE + ty) < K && col < L) {
            Bs[ty][tx] = B[(t * BLOCK_SIZE + ty) * L + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < L && batch < N) {
        C[batch * M * L + row * L + col] = sum;
    }
}

torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    auto C = torch::zeros({N, M, L}, A.options());

    const int BLOCK_SIZE = 16;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (L + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
        N
    );

    tensor_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M, K, L
    );

    return C;
}
"""

tensor_matmul_cpp_source = """
torch::Tensor tensor_matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

tensor_matmul = load_inline(
    name='tensor_matmul',
    cpp_sources=tensor_matmul_cpp_source,
    cuda_sources=tensor_matmul_cuda_source,
    functions=['tensor_matmul_cuda'],
    extra_cuda_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.tensor_matmul = tensor_matmul

    def forward(self, A, B):
        return self.tensor_matmul.tensor_matmul_cuda(A.cuda(), B.cuda())
```

## Level 2: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd

**Speedup**: 2.33

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p7.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d0d792daeb6fcee0a4d0e1b163978c0f.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies ReLU, LeakyReLU, GELU, Sigmoid activations, and bias in sequence.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        x = torch.sigmoid(x)
        x = x + self.bias
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]
```

### Solution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ float gelu_impl(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

__device__ float sigmoid_impl(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void fused_activations_bias_kernel(
    float* input,
    const float* bias,
    const int n_elements,
    const int channels,
    const int dhw) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        int c = (idx / dhw) % channels;
        float val = input[idx];
        
        // ReLU
        val = fmaxf(val, 0.0f);
        
        // LeakyReLU
        val = val > 0 ? val : 0.01f * val;
        
        // GELU
        val = gelu_impl(val);
        
        // Sigmoid
        val = sigmoid_impl(val);
        
        // Add bias
        val += bias[c];
        
        input[idx] = val;
    }
}

void fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias) {
    
    const int n_elements = input.numel();
    const int channels = input.size(1);
    const int dhw = input.size(2) * input.size(3) * input.size(4);
    
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;
    
    fused_activations_bias_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        n_elements,
        channels, 
        dhw);
}
"""

cpp_source = """
void fused_activations_bias_cuda(
    torch::Tensor input,
    torch::Tensor bias);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_activations_bias_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        self.fused_ops.fused_activations_bias_cuda(x, self.bias)
        return x
```

## Level 2: 14_Gemm_Divide_Sum_Scaling

**Speedup**: 3.17

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p14.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/3b400dd6bfedc298e2d95b1b95b73c6f.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        x = torch.matmul(x, self.weight.T)  # Gemm
        x = x / 2  # Divide
        x = torch.sum(x, dim=1, keepdim=True) # Sum
        x = x * self.scaling_factor  # Scaling
        return x


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel for matmul + divide + sum + scale
__global__ void fused_ops_kernel(
    const float* input,
    const float* weight,
    float* output,
    const float scaling_factor,
    const int batch_size,
    const int input_size,
    const int hidden_size
) {
    // Each thread handles one element in the batch
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        float sum = 0.0f;
        
        // Compute matmul and divide for this batch element
        for(int h = 0; h < hidden_size; h++) {
            float elem = 0.0f;
            for(int i = 0; i < input_size; i++) {
                elem += input[batch_idx * input_size + i] * 
                        weight[h * input_size + i];
            }
            // Divide by 2 as we go
            sum += (elem / 2.0f);
        }
        
        // Scale and store final result
        output[batch_idx] = sum * scaling_factor;
    }
}

torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor
) {
    const int batch_size = input.size(0);
    const int input_size = input.size(1);
    const int hidden_size = weight.size(0);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    fused_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        scaling_factor,
        batch_size,
        input_size,
        hidden_size
    );
    
    return output;
}
"""

cpp_source = """
torch::Tensor fused_ops_cuda(
    torch::Tensor input,
    torch::Tensor weight, 
    float scaling_factor
);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.fused_ops = fused_ops

    def forward(self, x):
        return self.fused_ops.fused_ops_cuda(x, self.weight, self.scaling_factor)
```

## Level 2: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd

**Speedup**: 1.75

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p20.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/ecce74df42226c991c533ae3ede3de06.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        original_x = x.clone().detach()
        x = x + self.bias
        x = x + original_x
        x = x * original_x
        x = x + original_x
        return x

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]
```

### Solution

```python
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for fused bias add + residual + multiply + residual
__global__ void fused_ops_kernel(float* x, const float* original_x, const float* bias,
                               int n, int c, int d, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = n * c * d * h * w;
    
    if (idx < total_size) {
        int c_idx = (idx / (d * h * w)) % c;
        float bias_val = bias[c_idx];
        float orig_val = original_x[idx];
        float x_val = x[idx];
        
        // Fused operations:
        // 1. Add bias
        // 2. Add residual
        // 3. Multiply with residual
        // 4. Add residual again
        x[idx] = ((x_val + bias_val + orig_val) * orig_val) + orig_val;
    }
}

std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias) {
    
    auto original_x = x.clone();
    auto sizes = x.sizes();
    int n = sizes[0];
    int c = sizes[1];
    int d = sizes[2];
    int h = sizes[3];
    int w = sizes[4];
    
    const int threads = 256;
    const int blocks = (n * c * d * h * w + threads - 1) / threads;
    
    fused_ops_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        original_x.data_ptr<float>(),
        bias.data_ptr<float>(),
        n, c, d, h, w
    );
    
    return {x};
}
"""

cpp_source = """
std::vector<torch::Tensor> fused_ops_cuda(
    torch::Tensor x,
    torch::Tensor bias);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, 
                                               stride=stride, padding=padding, 
                                               output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.fused_ops.fused_ops_cuda(x, self.bias)[0]
        return x
```

## Level 2: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU

**Speedup**: 2.02

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p90.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/447fac3129139e84230be00b8e0372d7.py

### Problem

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = x + self.sum_tensor
        x = torch.clamp(x, min=-1.0, max=1.0)
        x = torch.nn.functional.gelu(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]
```

### Solution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused LeakyReLU + Add + Clamp + GELU kernel
__global__ void fused_ops_kernel(
    float* input,
    const float* sum_tensor,
    const int batch_size,
    const int channels,
    const int depth,
    const int height, 
    const int width) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * channels * depth * height * width;
    
    if (idx < total_size) {
        const int c = (idx / (depth * height * width)) % channels;
        
        // LeakyReLU
        float val = input[idx];
        val = val > 0 ? val : 0.2f * val;
        
        // Add bias
        val += sum_tensor[c];
        
        // Clamp
        val = fminf(1.0f, fmaxf(-1.0f, val));
        
        // GELU approximation
        const float sqrt_2_over_pi = 0.7978845608028654f;
        const float coef = 0.044715f;
        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + coef * x3);
        val = 0.5f * val * (1.0f + tanhf(inner));
        
        input[idx] = val;
    }
}

void fused_ops_cuda(
    torch::Tensor input,
    const torch::Tensor sum_tensor) {
    
    const int batch_size = input.size(0);
    const int channels = input.size(1); 
    const int depth = input.size(2);
    const int height = input.size(3);
    const int width = input.size(4);
    
    const int total_elements = batch_size * channels * depth * height * width;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    fused_ops_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        sum_tensor.data_ptr<float>(),
        batch_size,
        channels, 
        depth,
        height,
        width);
}
"""

cpp_source = """
void fused_ops_cuda(
    torch::Tensor input,
    const torch::Tensor sum_tensor);
"""

fused_ops = load_inline(
    name='fused_ops',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_ops_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        self.fused_ops.fused_ops_cuda(x, self.sum_tensor)
        return x
```

## Level 3: 4_LeNet5

**Speedup**: 1.32

**Model**: claude-3.5-sonnet

**Problem URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l3_p4.py

**Solution URL**: https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/4f80ebb0657e05543d17fb5db74842ed.py

### Problem

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        """
        LeNet-5 architecture implementation in PyTorch.

        :param num_classes: The number of output classes.
        """
        super(Model, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
    
    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        :param x: The input tensor, shape (batch_size, 1, 32, 32)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # First convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Second convolutional layer with ReLU activation and max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 16*5*5)
        
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        
        # Final fully connected layer
        x = self.fc3(x)
        
        return x

# Test code for the LeNet-5 model
batch_size = 1
num_classes = 10

def get_inputs():
    return [torch.randn(batch_size, 1, 32, 32)]

def get_init_inputs():
    return [num_classes]
```

### Solution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv_relu_maxpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_relu_maxpool_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_channels, int out_channels,
    int in_height, int in_width, int kernel_size, int stride, int pool_size) {

    int b = blockIdx.x;
    int oc = blockIdx.y;
    int oh = blockIdx.z / ((in_width-kernel_size+1)/2);
    int ow = blockIdx.z % ((in_width-kernel_size+1)/2);

    float maxval = -1e9;
    
    for(int ph = 0; ph < pool_size; ph++) {
        for(int pw = 0; pw < pool_size; pw++) {
            float conv_val = bias[oc];
            
            for(int ic = 0; ic < in_channels; ic++) {
                for(int kh = 0; kh < kernel_size; kh++) {
                    for(int kw = 0; kw < kernel_size; kw++) {
                        int ih = oh*2 + ph + kh;
                        int iw = ow*2 + pw + kw;
                        
                        conv_val += input[b*in_channels*in_height*in_width + 
                                        ic*in_height*in_width +
                                        ih*in_width + iw] *
                                  weight[oc*in_channels*kernel_size*kernel_size +
                                        ic*kernel_size*kernel_size +
                                        kh*kernel_size + kw];
                    }
                }
            }
            
            float relu_val = conv_val > 0 ? conv_val : 0;
            maxval = max(maxval, relu_val);
        }
    }
    
    output[b*out_channels*((in_height-kernel_size+1)/2)*((in_width-kernel_size+1)/2) +
           oc*((in_height-kernel_size+1)/2)*((in_width-kernel_size+1)/2) +
           oh*((in_width-kernel_size+1)/2) + ow] = maxval;
}

torch::Tensor conv_relu_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    const int batch_size = input.size(0);
    const int in_channels = input.size(1); 
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    const int pool_size = 2;
    
    const int out_height = (in_height - kernel_size + 1) / 2;
    const int out_width = (in_width - kernel_size + 1) / 2;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                             input.options());
    
    dim3 grid(batch_size, out_channels, out_height * out_width);
    dim3 block(1);
    
    conv_relu_maxpool_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_channels, out_channels,
        in_height, in_width, kernel_size, 1, pool_size);
    
    return output;
}
"""

conv_relu_maxpool_cpp_source = """
torch::Tensor conv_relu_maxpool_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

linear_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void linear_relu_kernel(
    const float* input, const float* weight, const float* bias,
    float* output, int batch_size, int in_features, int out_features) {
    
    int b = blockIdx.x;
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    
    if(o < out_features) {
        float val = bias[o];
        for(int i = 0; i < in_features; i++) {
            val += input[b*in_features + i] * weight[o*in_features + i];
        }
        output[b*out_features + o] = val > 0 ? val : 0;
    }
}

torch::Tensor linear_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    
    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, input.options());
    
    const int threads = 256;
    const int blocks_y = (out_features + threads - 1) / threads;
    
    dim3 grid(batch_size, blocks_y);
    dim3 block(threads);
    
    linear_relu_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), batch_size, in_features, out_features);
    
    return output;
}
"""

linear_relu_cpp_source = """
torch::Tensor linear_relu_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias);
"""

conv_relu_maxpool = load_inline(
    name='conv_relu_maxpool',
    cpp_sources=conv_relu_maxpool_cpp_source,
    cuda_sources=conv_relu_maxpool_source,
    functions=['conv_relu_maxpool_cuda'],
    verbose=True
)

linear_relu = load_inline(
    name='linear_relu',
    cpp_sources=linear_relu_cpp_source,
    cuda_sources=linear_relu_source,
    functions=['linear_relu_cuda'],
    verbose=True
)

class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super(ModelNew, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Fused conv+relu+maxpool operations
        x = conv_relu_maxpool.conv_relu_maxpool_cuda(
            x, self.conv1.weight, self.conv1.bias)
        x = conv_relu_maxpool.conv_relu_maxpool_cuda(
            x, self.conv2.weight, self.conv2.bias)
        
        x = x.view(-1, 16*5*5)
        
        # Fused linear+relu operations
        x = linear_relu.linear_relu_cuda(
            x, self.fc1.weight, self.fc1.bias)
        x = linear_relu.linear_relu_cuda(
            x, self.fc2.weight, self.fc2.bias)
        
        # Final linear layer (no ReLU)
        x = F.linear(x, self.fc3.weight, self.fc3.bias)
        
        return x
```

