import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os

class ModelNew(nn.Module):
    """Simple model that performs a single square matrix multiplication using a custom CUDA kernel"""
    def __init__(self):
        super(ModelNew, self).__init__()
        # Define the CUDA kernel
        self.matmul = self.cuda_matmul()

    def cuda_matmul(self):
        kernel_code = """
        #include <torch/extension.h>
        #include <CUDA.h>

        #define BLOCK_SIZE 16

        __global__ void matmul_kernel(float* A, float* B, float* C, int N) {
            __shared__ float ds_A[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float ds_B[BLOCK_SIZE][BLOCK_SIZE];

            int block_row = blockIdx.y * BLOCK_SIZE;
            int block_col = blockIdx.x * BLOCK_SIZE;

            int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
            int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

            float value = 0.;

            for (int k = 0; k < (N / BLOCK_SIZE); k++) {
                int a_row = block_row + threadIdx.y;
                int a_col = k * BLOCK_SIZE + threadIdx.x;
                int b_row = k * BLOCK_SIZE + threadIdx.y;
                int b_col = block_col + threadIdx.x;

                ds_A[threadIdx.y][threadIdx.x] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.;
                ds_B[threadIdx.y][threadIdx.x] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.;

                __syncthreads();

                for (int m = 0; m < BLOCK_SIZE; m++) {
                    value += ds_A[threadIdx.y][m] * ds_B[m][threadIdx.x];
                }

                __syncthreads();
            }

            if (row < N && col < N) {
                C[row * N + col] = value;
            }
        }

        std::vector<torch::Tensor> matmul(torch::Tensor A, torch::Tensor B) {
            int N = A.size(0);
            torch::Tensor C = torch::empty({N, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

            matmul_kernel<<<grid, block, 0, torch::cuda::getCurrentCUDAStream()>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

            return {C};
        }

        TORCH_LIBRARY_IMPL(custom_matmul, CUDA, m) {
            m.def("matmul(Tensor A, Tensor B) -> Tensor", matmul, torch::dtype(torch::kFloat32).device(torch::kCUDA));
        }
        """
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        compile_params = ['--cuda', '-gencode=arch=compute_70,code=sm_70']
        module = load(name='custom_matmul', sources=[kernel_code], rebuild=True, extra_cuda_cflags=compile_params)
        return module.matmul
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Performs the matrix multiplication using the custom CUDA kernel.

        Args:
            A (torch.Tensor): Input matrix A of shape (N, N).
            B (torch.Tensor): Input matrix B of shape (N, N).

        Returns:
            torch.Tensor: Output matrix C of shape (N, N).
        """
        if A.device.type != 'cuda':
            A = A.cuda()
        if B.device.type != 'cuda':
            B = B.cuda()
            
        N = A.size(0)
        C = torch.empty((N, N), device=A.device, dtype=A.dtype)
        
        block_size = 16
        grid_size = (N + block_size - 1) // block_size
        
        with torch.cuda.stream(torch.cuda.default_stream()):
            self.matmul(A, B, C)  # type: ignore
            
        return C
        
def get_inputs():
    A = torch.randn(N, N).cuda()
    B = torch.randn(N, N).cuda()
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed