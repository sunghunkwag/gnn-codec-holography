"""Example: Compress a pre-trained model using GNN-based codec-holography engine."""

import torch
import torchvision.models as models
from gnn_codec import GNNCodecHolographyEngine, GNNTrainingSystem
import time


def main():
    """Main example demonstrating model compression."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load a pre-trained model
    print("Loading pre-trained ResNet-18...")
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Extract weight tensors
    weight_tensors = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            weight_tensors.append(param.data.clone())
            layer_names.append(name)
    
    print(f"Extracted {len(weight_tensors)} weight layers")
    
    # Initialize codec engine
    engine = GNNCodecHolographyEngine(
        phase_bits=8,
        amp_bits=4,
        gnn_hidden_dim=32,
        gnn_layers=3,
        use_graph_prediction=True
    ).to(device)
    
    print("\nTesting compression on individual layers:")
    print("-" * 60)
    
    total_original_size = 0
    total_compressed_size = 0
    
    for i, (name, weights) in enumerate(zip(layer_names, weight_tensors)):
        weights = weights.to(device)
        
        # Time the compression
        start_time = time.time()
        
        # Compress and decompress
        encoded = engine(weights)
        decoded = engine.decode(encoded)
        
        compression_time = (time.time() - start_time) * 1000  # ms
        
        # Calculate metrics
        mse_error = torch.mean((weights - decoded) ** 2).item()
        compression_info = engine.get_compression_info(encoded)
        
        # Track total sizes
        original_size = weights.numel() * 4  # 4 bytes per float32
        compressed_size = compression_info['compressed_size_bits'] / 8  # convert to bytes
        
        total_original_size += original_size
        total_compressed_size += compressed_size
        
        print(f"{name:<25} | "
              f"Shape: {str(weights.shape):<15} | "
              f"Compression: {compression_info['compression_ratio']:.1f}x | "
              f"MSE: {mse_error:.2e} | "
              f"Time: {compression_time:.1f}ms")
    
    # Summary
    overall_compression = total_original_size / total_compressed_size
    memory_saved = total_original_size - total_compressed_size
    
    print("-" * 60)
    print(f"Overall Results:")
    print(f"  Original model size: {total_original_size / 1024**2:.2f} MB")
    print(f"  Compressed size: {total_compressed_size / 1024**2:.2f} MB")
    print(f"  Overall compression: {overall_compression:.1f}x")
    print(f"  Memory saved: {memory_saved / 1024**2:.2f} MB ({memory_saved/total_original_size*100:.1f}%)")
    print(f"  Target 240x: {'Achieved' if overall_compression >= 240 else 'In progress'}")


def train_example():
    """Example of training the codec engine."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create engine
    engine = GNNCodecHolographyEngine(
        phase_bits=8,
        amp_bits=4,
        gnn_hidden_dim=16,
        gnn_layers=2,
        use_graph_prediction=True
    )
    
    # Create training system
    trainer = GNNTrainingSystem(engine, device)
    trainer.setup_scheduler('cosine', T_max=20)
    
    # Generate synthetic training data
    training_weights = [
        torch.randn(64, 32) * 0.1,
        torch.randn(32, 16) * 0.05,
        torch.randn(16, 8, 3, 3) * 0.08,
        torch.randn(8, 4, 3, 3) * 0.06,
    ]
    
    print("Training codec engine on synthetic data...")
    
    # Train
    history = trainer.train_on_weights(
        training_weights,
        epochs=20,
        batch_size=2,
        verbose=True
    )
    
    # Evaluate
    results = trainer.evaluate_compression_performance(training_weights)
    
    print(f"\nTraining completed!")
    print(f"Final compression ratio: {results['summary']['mean_compression']:.1f}x")
    print(f"Final MSE: {results['summary']['mean_mse']:.2e}")


if __name__ == "__main__":
    print("GNN-based Codec-Holography Engine Example")
    print("=" * 50)
    
    # Run compression example
    main()
    
    print("\n" + "=" * 50)
    print("Training Example")
    print("=" * 50)
    
    # Run training example
    train_example()