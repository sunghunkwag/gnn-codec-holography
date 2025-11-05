# Usage Guide

This guide covers basic usage patterns and advanced configurations for the GNN-based Codec-Holography Engine.

## Installation

### From Source

```bash
git clone https://github.com/sunghunkwag/gnn-codec-holography.git
cd gnn-codec-holography
pip install -e .
```

### Dependencies

- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.4.0
- NumPy >= 1.21.0

## Basic Usage

### Compressing a Pre-trained Model

```python
import torch
import torchvision.models as models
from gnn_codec import GNNCodecHolographyEngine

# Load model
model = models.resnet18(pretrained=True)

# Initialize codec engine
engine = GNNCodecHolographyEngine(
    phase_bits=8,
    amp_bits=4,
    use_graph_prediction=True
)

# Compress model weights
compressed_model = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        encoded = engine(param.data)
        compressed_model[name] = encoded

# Calculate compression ratio
total_original = sum(p.numel() * 32 for p in model.parameters())
total_compressed = sum(engine.get_compression_info(data)['compressed_size_bits'] 
                      for data in compressed_model.values())
compression_ratio = total_original / total_compressed
print(f"Compression ratio: {compression_ratio:.1f}x")
```

### Decompressing Weights

```python
# Decompress weights
decompressed_model = {}
for name, encoded_data in compressed_model.items():
    decompressed_model[name] = engine.decode(encoded_data)

# Load back into model
for name, param in model.named_parameters():
    if name in decompressed_model:
        param.data.copy_(decompressed_model[name])
```

## Training the Codec Engine

### Basic Training

```python
from gnn_codec import GNNTrainingSystem

# Create engine
engine = GNNCodecHolographyEngine(
    phase_bits=8,
    amp_bits=4,
    gnn_hidden_dim=64,
    gnn_layers=4
)

# Setup training
trainer = GNNTrainingSystem(engine, device='cuda')
trainer.setup_scheduler('cosine', T_max=100)

# Collect training weights
training_weights = []
for model in [models.resnet18(), models.vgg16(), models.efficientnet_b0()]:
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            training_weights.append(param.data.clone())

# Train
history = trainer.train_on_weights(
    training_weights,
    epochs=100,
    batch_size=8,
    validation_split=0.1,
    checkpoint_dir='./checkpoints'
)

# Evaluate
results = trainer.evaluate_compression_performance(training_weights[:10])
print(f"Final compression: {results['summary']['mean_compression']:.1f}x")
```

## Configuration Options

### Engine Parameters

#### Quantization Settings
- `phase_bits`: Bits for phase quantization (default: 8)
- `amp_bits`: Bits for amplitude quantization (default: 4)
- Higher values = better quality, lower compression

#### GNN Architecture
- `gnn_hidden_dim`: Hidden layer size (default: 64)
- `gnn_layers`: Number of GNN layers (default: 4)
- `num_heads`: Attention heads (fixed: 4)

### Performance Optimization

```python
# GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
engine = engine.to(device)

# Memory management for large models
for name, param in large_model.named_parameters():
    if 'weight' in name:
        param_gpu = param.to('cuda')
        encoded = engine(param_gpu)
        del param_gpu
        torch.cuda.empty_cache()
```

## Quality Assessment

### Reconstruction Quality

```python
def assess_quality(original, compressed, engine):
    decoded = engine.decode(compressed)
    
    # Mean squared error
    mse = torch.mean((original - decoded) ** 2)
    
    # Signal-to-noise ratio
    signal_power = torch.mean(original ** 2)
    snr_db = 10 * torch.log10(signal_power / (mse + 1e-8))
    
    return {
        'mse': mse.item(),
        'snr_db': snr_db.item()
    }
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `gnn_hidden_dim` or process smaller batches
2. **Poor Compression**: Increase training epochs or use more training data
3. **High Reconstruction Error**: Lower quantization bits or improve GNN architecture
4. **Slow Training**: Use GPU acceleration and mixed precision

### Performance Tuning

1. **For Better Compression**: Train longer, use more GNN layers
2. **For Better Speed**: Reduce GNN complexity, use smaller graphs  
3. **For Better Quality**: Use higher quantization bits, add regularization