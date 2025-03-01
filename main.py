from openai import OpenAI

client = OpenAI(
    base_url="https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
    api_key="EMPTY"
)

completion = client.chat.completions.create(
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  stream=True,
  messages=[
    {"role": "user", "content": "Write a CUDA Kernel. Here are several examples:\n\n"
     "__global__ void vectorAdd(float *A, float *B, float *C, int N) {\n"
     "    int index = threadIdx.x + blockIdx.x * blockDim.x;\n"
     "    if (index < N) {\n"
     "        C[index] = A[index] + B[index];\n"
     "    }\n"
     "}\n\n"
     "__global__ void matrixMul(float *A, float *B, float *C, int N) {\n"
     "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
     "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
     "    if (row < N && col < N) {\n"
     "        float value = 0;\n"
     "        for (int k = 0; k < N; ++k) {\n"
     "            value += A[row * N + k] * B[k * N + col];\n"
     "        }\n"
     "        C[row * N + col] = value;\n"
     "    }\n"
     "}\n\n"
     "__global__ void squareArray(float *input, float *output, int N) {\n"
     "    int index = threadIdx.x + blockIdx.x * blockDim.x;\n"
     "    if (index < N) {\n"
     "        output[index] = input[index] * input[index];\n"
     "    }\n"
     "}\n\n"
     "__global__ void reduceSum(float *input, float *output, int N) {\n"
     "    extern __shared__ float sharedData[];\n"
     "    int index = threadIdx.x + blockIdx.x * blockDim.x;\n"
     "    int tid = threadIdx.x;\n"
     "    if (index < N) {\n"
     "        sharedData[tid] = input[index];\n"
     "    } else {\n"
     "        sharedData[tid] = 0.0f;\n"
     "    }\n"
     "    __syncthreads();\n"
     "    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {\n"
     "        if (tid < stride) {\n"
     "            sharedData[tid] += sharedData[tid + stride];\n"
     "        }\n"
     "        __syncthreads();\n"
     "    }\n"
     "    if (tid == 0) {\n"
     "        output[blockIdx.x] = sharedData[0];\n"
     "    }\n"
     "}\n"
    }
  ]
)

for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)