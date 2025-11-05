# GNN-Codec Holography Engine Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ALL necessary files first (including README.md which setup.py needs)
COPY README.md .
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Install Python dependencies from requirements.txt first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric with proper CUDA support
RUN pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Copy source code after installing dependencies
COPY src/ ./src/
COPY examples/ ./examples/

# Install the package in development mode
# Add error handling to see what exactly fails
RUN pip install -e . || (echo "Installation failed. Listing current directory:"; ls -la; echo "Checking src directory:"; ls -la src/; echo "Checking requirements:"; cat requirements.txt; exit 1)

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for Jupyter if needed
EXPOSE 8888

# Create a simple health check script
RUN echo 'import torch; import torch_geometric; import gnn_codec; print("Container healthy"); print(f"CUDA available: {torch.cuda.is_available()}"); print(f"GNN Codec version: {gnn_codec.__version__}")' > /app/healthcheck.py

# Default command - fallback to healthcheck if compress_model.py doesn't exist
CMD python examples/compress_model.py 2>/dev/null || python healthcheck.py

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python /app/healthcheck.py || exit 1