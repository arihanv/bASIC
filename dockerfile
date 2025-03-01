# Use NVIDIA CUDA image as base
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install KernelBench in development mode
WORKDIR /app/KernelBench
RUN pip install -e .

# Return to app directory
WORKDIR /app

# Expose the port
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]