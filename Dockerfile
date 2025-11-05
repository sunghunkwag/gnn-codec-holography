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

# Install PyTorch Geometric with proper CUDA support
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

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

# Create a simple health check script
RUN echo 'import torch; import torch_geometric; print("Container healthy"); print(f"CUDA available: {torch.cuda.is_available()}")' > /app/healthcheck.py

# Default command - fallback to healthcheck if compress_model.py doesn't exist
CMD python examples/compress_model.py 2>/dev/null || python healthcheck.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python /app/healthcheck.py || exit 1