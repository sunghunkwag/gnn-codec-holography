# GNN-Codec Holography Engine Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric
RUN pip install torch-geometric

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY setup.py .
COPY pyproject.toml .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for Jupyter if needed
EXPOSE 8888

# Default command
CMD ["python", "examples/compress_model.py"]

# Health check
HEALTHCHeck --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import torch; import torch_geometric; print('Container healthy')" || exit 1