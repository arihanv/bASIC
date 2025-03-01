import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# CUDA kernel code
cuda_code = '''
#define TILE_SIZE 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < N && t * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (t * TILE_SIZE + ty < N && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
'''

matmul_kernel = load(name='matmul_kernel', sources=[cuda_code], verbose=False)

class ModelNew(nn.Module):
    """Model that performs a single square matrix multiplication (C = A * B) using a custom CUDA kernel"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs the matrix multiplication using a custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        N = A.size(0)
        C = torch.zeros_like(A)

        block_size = 32
        grid_size = (N + block_size - 1) // block_size

        matmul_kernel.matmul_kernel(
            A.contiguous().cuda(),
            B.contiguous().cuda(),
            C.cuda(),
            N,
            block=(block_size, block_size, 1),
            grid=(grid_size, grid_size)
        )

        return C

N = 2048

def get_inputs():
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed