"""Tests for core holographic transformation components."""

import torch
import pytest
from gnn_codec.core import ComplexTensor, HolographicTransform


class TestComplexTensor:
    """Test cases for ComplexTensor class."""
    
    def test_initialization(self):
        """Test ComplexTensor initialization."""
        real = torch.randn(10, 20)
        imag = torch.randn(10, 20)
        
        complex_tensor = ComplexTensor(real, imag)
        
        assert complex_tensor.shape == (10, 20)
        assert torch.allclose(complex_tensor.real, real)
        assert torch.allclose(complex_tensor.imag, imag)
    
    def test_magnitude_and_phase(self):
        """Test magnitude and phase calculations."""
        real = torch.tensor([3.0, 0.0, -1.0])
        imag = torch.tensor([4.0, 1.0, 0.0])
        
        complex_tensor = ComplexTensor(real, imag)
        
        expected_magnitude = torch.tensor([5.0, 1.0, 1.0])
        expected_phase = torch.tensor([0.9273, 1.5708, 3.1416], dtype=torch.float32)
        
        assert torch.allclose(complex_tensor.abs(), expected_magnitude)
        assert torch.allclose(complex_tensor.angle(), expected_phase, atol=1e-4)
    
    def test_arithmetic_operations(self):
        """Test arithmetic operations."""
        real1, imag1 = torch.tensor([1.0, 2.0]), torch.tensor([2.0, 1.0])
        real2, imag2 = torch.tensor([3.0, 1.0]), torch.tensor([1.0, 3.0])
        
        c1 = ComplexTensor(real1, imag1)
        c2 = ComplexTensor(real2, imag2)
        
        # Addition
        c_add = c1 + c2
        assert torch.allclose(c_add.real, torch.tensor([4.0, 3.0]))
        assert torch.allclose(c_add.imag, torch.tensor([3.0, 4.0]))
        
        # Subtraction
        c_sub = c1 - c2
        assert torch.allclose(c_sub.real, torch.tensor([-2.0, 1.0]))
        assert torch.allclose(c_sub.imag, torch.tensor([1.0, -2.0]))
        
        # Multiplication
        c_mul = c1 * c2
        expected_real = real1 * real2 - imag1 * imag2  # [1*3 - 2*1, 2*1 - 1*3] = [1, -1]
        expected_imag = real1 * imag2 + imag1 * real2  # [1*1 + 2*3, 2*3 + 1*1] = [7, 7]
        
        assert torch.allclose(c_mul.real, expected_real)
        assert torch.allclose(c_mul.imag, expected_imag)


class TestHolographicTransform:
    """Test cases for HolographicTransform."""
    
    def test_initialization(self):
        """Test HolographicTransform initialization."""
        transform = HolographicTransform(alpha=0.5, beta=1.5)
        
        assert torch.allclose(transform.alpha, torch.tensor(0.5))
        assert torch.allclose(transform.beta, torch.tensor(1.5))
    
    def test_forward_backward(self):
        """Test forward and inverse transforms."""
        transform = HolographicTransform()
        
        # Test with simple input
        input_tensor = torch.randn(100) * 0.1
        
        # Forward transform
        complex_repr = transform.forward(input_tensor)
        
        assert isinstance(complex_repr, ComplexTensor)
        assert complex_repr.shape == input_tensor.shape
        
        # Inverse transform
        reconstructed = transform.inverse(complex_repr)
        
        assert reconstructed.shape == input_tensor.shape
        
        # Check reconstruction quality (should be reasonable)
        mse_error = torch.mean((input_tensor - reconstructed) ** 2)
        assert mse_error < 1.0  # Reasonable reconstruction
    
    def test_gradient_flow(self):
        """Test that gradients flow through the transform."""
        transform = HolographicTransform()
        input_tensor = torch.randn(50, requires_grad=True) * 0.1
        
        # Forward pass
        complex_repr = transform.forward(input_tensor)
        
        # Create a simple loss
        loss = torch.mean(complex_repr.abs())
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        transform = HolographicTransform()
        
        # Test with very small values
        small_input = torch.ones(10) * 1e-8
        complex_repr = transform.forward(small_input)
        reconstructed = transform.inverse(complex_repr)
        
        assert torch.isfinite(reconstructed).all()
        
        # Test with larger values  
        large_input = torch.ones(10) * 10.0
        complex_repr = transform.forward(large_input)
        reconstructed = transform.inverse(complex_repr)
        
        assert torch.isfinite(reconstructed).all()


if __name__ == "__main__":
    pytest.main([__file__])