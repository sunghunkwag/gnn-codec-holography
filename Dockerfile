# GNN-Codec Holography Engine Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy ALL necessary files first (including README.md which setup.py needs)
COPY README.md .
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies from requirements.txt first
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric with proper CUDA support
# Use conda to avoid potential conflicts
RUN conda install -c pyg torch-geometric -y || \
    pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Copy source code after installing dependencies
COPY src/ ./src/
COPY examples/ ./examples/

# Copy tests directory if it exists (safe shell-based approach)
RUN if [ -d tests ]; then \
      mkdir -p ./tests && cp -r tests/* ./tests/; \
    else \
      echo "No tests directory found, skipping..."; \
    fi

# Install the package in development mode
# Add comprehensive error handling and debugging
RUN echo "=== Installation Debug Info ===" && \
    echo "Current directory contents:" && ls -la && \
    echo "Source directory contents:" && ls -la src/ && \
    echo "Requirements:" && cat requirements.txt && \
    echo "=== Installing package ===" && \
    pip install -e . && \
    echo "=== Installation successful ==="

# Run installation test to verify everything works
RUN python examples/test_installation.py

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Expose port for Jupyter if needed
EXPOSE 8888

# Create entrypoint script using printf (safer than echo with complex strings)
RUN printf '#!/bin/bash\n\
echo "GNN Codec Holography Engine - Docker Container"\n\
echo "==========================================="\n\
echo "Available examples:"\n\
ls -la examples/\n\
echo ""\n\
echo "Running compress_model.py if available, otherwise test_installation.py"\n\
if [ -f "examples/compress_model.py" ]; then\n\
    python examples/compress_model.py\n\
else\n\
    echo "compress_model.py not found, running installation test..."\n\
    python examples/test_installation.py\n\
fi\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Default command
CMD ["/app/entrypoint.sh"]

# Health check using our test script
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD python examples/test_installation.py || exit 1