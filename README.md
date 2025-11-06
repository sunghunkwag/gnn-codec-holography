# GNN-Based Codec-Holography Engine

[![Container](https://img.shields.io/badge/GHCR-gnn--codec--holography-2ea44f)](#docker-deployment)

A neural network weight compression system using Graph Neural Networks (GNNs) and holographic representations.

## Overview

This project implements an experimental approach to neural network compression by:
- Converting weights to graph structures 
- Applying holographic transformations to complex-valued representations
- Using Graph Neural Networks for predictive compression
- Exploring compression techniques with acceptable quality trade-offs

## Key Features

- **Graph-based Architecture**: Leverages the natural graph structure of neural networks
- **Holographic Representation**: Complex-valued encoding for higher information density  
- **Predictive Compression**: GNN-based weight prediction reduces residual information
- **Memory Efficient**: O(E) complexity vs O(N²) for transformer-based approaches
- **Hardware Agnostic**: Supports CPU/GPU acceleration via PyTorch Geometric
- **Research Implementation**: Docker containers for experimentation and testing

## Performance

- Base GNN compression: 28.9x average (preliminary results)
- Target system: Higher compression ratios under investigation
- Memory efficiency improvements demonstrated
- Processing time: Varies based on model size and hardware

## Quick Start

### Docker Deployment

```bash
# Pull pre-built image
docker pull ghcr.io/sunghunkwag/gnn-codec-holography:latest

# Run container
docker run --gpus all -it ghcr.io/sunghunkwag/gnn-codec-holography:latest

# Or build locally
docker build -t gnn-codec-holography .
docker run --gpus all -it gnn-codec-holography
```

> Latest container: `ghcr.io/sunghunkwag/gnn-codec-holography:latest`

### Local Installation

```bash
# Install dependencies
pip install torch torch-geometric numpy

# Clone repository
git clone https://github.com/sunghunkwag/gnn-codec-holography.git
cd gnn-codec-holography

# Install package
pip install -e .

# Run example
python examples/compress_model.py
```

## Architecture

```
Weight Tensor → Graph Builder → Holographic Transform → GNN Predictor → Quantizer → Compressed
     ↓                                                                                  ↓
Reconstructed ← Inverse Transform ← Residual + Prediction ← Dequantizer ← Storage
```

## Repository Structure

- `src/`: Core implementation modules
- `examples/`: Usage examples and demonstrations  
- `tests/`: Unit tests and benchmarks
- `docs/`: Technical documentation
- `benchmarks/`: Performance evaluation scripts
- `Dockerfile`: Container configuration for testing
- `.github/workflows/`: Automated CI/CD pipelines

## Development

### Building from Source

```bash
git clone https://github.com/sunghunkwag/gnn-codec-holography.git
cd gnn-codec-holography
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

### Docker Development

```bash
# Build development image
docker build -t gnn-codec-dev .

# Run with volume mount for development
docker run --gpus all -v $(pwd):/app -it gnn-codec-dev bash
```

## Deployment

### Cloud Infrastructure

For experimentation and testing:

```bash
# AWS ECS/EKS
docker pull ghcr.io/sunghunkwag/gnn-codec-holography:latest

# Google Cloud Run
gcloud run deploy gnn-codec --image ghcr.io/sunghunkwag/gnn-codec-holography:latest

# Azure Container Instances
az container create --resource-group myResourceGroup \
  --name gnn-codec --image ghcr.io/sunghunkwag/gnn-codec-holography:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gnn-codec-deployment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gnn-codec
  template:
    metadata:
      labels:
        app: gnn-codec
    spec:
      containers:
      - name: gnn-codec
        image: ghcr.io/sunghunkwag/gnn-codec-holography:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

## Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests.

## License

MIT License - see LICENSE file for details

## Citation

If you use this work, please cite:
```
@software{gnn_codec_holography_2025,
  title={GNN-Based Codec-Holography Engine},
  author={Sung hun kwag},
  year={2025},
  url={https://github.com/sunghunkwag/gnn-codec-holography}
}
```