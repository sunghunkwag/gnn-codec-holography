#!/usr/bin/env python3
"""
Simple test script to verify that the GNN Codec Holography package 
is properly installed and all dependencies work correctly.

This script is used as a fallback in the Docker container to ensure
everything is working properly.
"""

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch Geometric: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import gnn_codec
        print(f"✓ GNN Codec {gnn_codec.__version__} imported successfully")
        
        # Test key components
        from gnn_codec import GNNCodecHolographyEngine, ComplexTensor
        print("✓ Core components imported successfully")
        
    except ImportError as e:
        print(f"✗ Failed to import GNN Codec: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the package."""
    print("\nTesting basic functionality...")
    
    try:
        import torch
        import numpy as np
        from gnn_codec import ComplexTensor, HolographicTransform
        
        # Test ComplexTensor creation (requires both real and imaginary parts)
        real_data = torch.randn(10, 5)
        imag_data = torch.randn(10, 5) 
        complex_tensor = ComplexTensor(real_data, imag_data)
        print(f"✓ Created ComplexTensor with shape {complex_tensor.real.shape}")
        
        # Test basic operations
        magnitude = complex_tensor.abs()
        print(f"✓ Computed magnitude: {magnitude.mean().item():.4f}")
        
        # Test angle computation
        angle = complex_tensor.angle()
        print(f"✓ Computed angle: {angle.mean().item():.4f}")
        
        # Test HolographicTransform
        transform = HolographicTransform()
        test_input = torch.randn(5, 3)
        transformed = transform(test_input)
        print(f"✓ HolographicTransform test: output shape {transformed.real.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("GNN Codec Holography - Installation Test")
    print("=" * 50)
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    # Test basic functionality
    if not test_basic_functionality():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! Installation is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())