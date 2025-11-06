# GNN-Codec Holography Engine Docker Image (Robust Multi-Stage)
# Slimmer base to reduce layer size and disk pressure
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

WORKDIR /app

# System deps (minimal) and cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    curl \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Copy manifest files early for better layer caching
COPY README.md .
COPY requirements.txt .
COPY pyproject.toml .
COPY setup.py .

# Python tooling and deps (pip-only; no conda to avoid bloat)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge || true \
 && rm -rf /root/.cache || true

# Torch Geometric (CUDA 11.8 compatible)
RUN pip install --no-cache-dir torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cu118.html \
 && pip cache purge || true \
 && rm -rf /root/.cache || true

# ---------- TEST STAGE ----------
FROM base AS test
WORKDIR /app
COPY src/ ./src/
COPY examples/ ./examples/
# tests are optional; copy if present
COPY tests/ ./tests/ 2>/dev/null || true

# Editable install and validate inside build to fail early
RUN pip install --no-cache-dir -e . \
 && python examples/test_installation.py \
 && pip cache purge || true \
 && rm -rf /root/.cache || true

# ---------- RUNTIME/PROD STAGE ----------
FROM base AS prod
WORKDIR /app
# Only what we need at runtime (keep image lean)
COPY --from=test /app/src /app/src
COPY --from=test /app/examples /app/examples
COPY --from=test /app/README.md /app/README.md
COPY --from=test /app/pyproject.toml /app/pyproject.toml
COPY --from=test /app/setup.py /app/setup.py

# Install package (non-editable) for production
RUN pip install --no-cache-dir . \
 && pip cache purge || true \
 && rm -rf /root/.cache || true

ENV PYTHONPATH=/app/src:/app
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

EXPOSE 8888

# Entrypoint kept simple and robust
RUN printf '#!/bin/bash\n\
echo "GNN Codec Holography Engine - Docker Container"\n\
echo "==========================================="\n\
echo "Available examples:"\n\
ls -la examples/ || true\n\
echo ""\n\
echo "Running compress_model.py if available, otherwise installation test"\n\
if [ -f "examples/compress_model.py" ]; then\n\
    python examples/compress_model.py\n\
else\n\
    echo "compress_model.py not found, running installation test..."\n\
    python examples/test_installation.py\n\
fi\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD python examples/test_installation.py || exit 1