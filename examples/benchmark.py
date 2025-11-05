"""Benchmark script for comparing compression performance across different models."""

import torch
import torchvision.models as models
from gnn_codec import GNNCodecHolographyEngine
import time
import json


def benchmark_model(model, model_name: str, engine: GNNCodecHolographyEngine, device: str):
    """Benchmark compression performance on a single model."""
    
    # Extract weights
    weights = []
    layer_info = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) >= 2:
            weights.append(param.data.clone())
            layer_info.append({
                'name': name,
                'shape': list(param.shape),
                'parameters': param.numel()
            })
    
    total_params = sum(info['parameters'] for info in layer_info)
    
    # Benchmark compression
    results = {
        'model_name': model_name,
        'total_parameters': total_params,
        'total_layers': len(weights),
        'layers': []
    }
    
    total_compression_time = 0
    total_original_bits = 0
    total_compressed_bits = 0
    
    for i, (weight, info) in enumerate(zip(weights, layer_info)):
        weight = weight.to(device)
        
        # Time compression
        start_time = time.time()
        encoded = engine(weight)
        decoded = engine.decode(encoded)
        compression_time = (time.time() - start_time) * 1000
        
        # Calculate metrics
        mse_error = torch.mean((weight - decoded) ** 2).item()
        compression_info = engine.get_compression_info(encoded)
        
        # Track totals
        original_bits = weight.numel() * 32
        compressed_bits = compression_info['compressed_size_bits']
        
        total_compression_time += compression_time
        total_original_bits += original_bits
        total_compressed_bits += compressed_bits
        
        layer_result = {
            'name': info['name'],
            'shape': info['shape'],
            'parameters': info['parameters'],
            'compression_ratio': compression_info['compression_ratio'],
            'mse_error': mse_error,
            'compression_time_ms': compression_time,
            'memory_saving_percent': compression_info['memory_savings_percent']
        }
        
        results['layers'].append(layer_result)
    
    # Overall metrics
    results['summary'] = {
        'overall_compression_ratio': total_original_bits / total_compressed_bits,
        'total_compression_time_ms': total_compression_time,
        'average_compression_time_ms': total_compression_time / len(weights),
        'original_size_mb': total_original_bits / (8 * 1024 * 1024),
        'compressed_size_mb': total_compressed_bits / (8 * 1024 * 1024),
        'memory_savings_mb': (total_original_bits - total_compressed_bits) / (8 * 1024 * 1024)
    }
    
    return results


def main():
    """Run comprehensive benchmark across multiple model types."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running benchmark on {device}")
    
    # Initialize codec engine
    engine = GNNCodecHolographyEngine(
        phase_bits=8,
        amp_bits=4,
        gnn_hidden_dim=32,
        gnn_layers=3,
        use_graph_prediction=True
    ).to(device)
    
    # Models to benchmark
    models_to_test = [
        (models.resnet18(pretrained=True), "ResNet-18"),
        (models.resnet50(pretrained=True), "ResNet-50"),
        (models.vgg16(pretrained=True), "VGG-16"),
        (models.efficientnet_b0(pretrained=True), "EfficientNet-B0"),
    ]
    
    benchmark_results = []
    
    for model, model_name in models_to_test:
        print(f"\nBenchmarking {model_name}...")
        model.eval()
        
        try:
            result = benchmark_model(model, model_name, engine, device)
            benchmark_results.append(result)
            
            # Print summary
            summary = result['summary']
            print(f"  Parameters: {result['total_parameters']:,}")
            print(f"  Compression: {summary['overall_compression_ratio']:.1f}x")
            print(f"  Original size: {summary['original_size_mb']:.1f} MB")
            print(f"  Compressed size: {summary['compressed_size_mb']:.1f} MB")
            print(f"  Time: {summary['total_compression_time_ms']:.1f} ms")
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            continue
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    total_params = sum(r['total_parameters'] for r in benchmark_results)
    avg_compression = sum(r['summary']['overall_compression_ratio'] for r in benchmark_results) / len(benchmark_results)
    total_original = sum(r['summary']['original_size_mb'] for r in benchmark_results)
    total_compressed = sum(r['summary']['compressed_size_mb'] for r in benchmark_results)
    
    print(f"Models tested: {len(benchmark_results)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Average compression: {avg_compression:.1f}x")
    print(f"Total original size: {total_original:.1f} MB")
    print(f"Total compressed size: {total_compressed:.1f} MB")
    print(f"Overall compression: {total_original/total_compressed:.1f}x")
    print(f"240x target: {'ACHIEVED' if avg_compression >= 240 else 'IN PROGRESS'}")
    
    print(f"\nResults saved to: benchmark_results.json")


if __name__ == "__main__":
    main()