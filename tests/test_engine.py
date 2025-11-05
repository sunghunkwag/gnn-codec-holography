"""Tests for the main codec engine."""

import torch
import pytest
from gnn_codec import GNNCodecHolographyEngine


class TestGNNCodecHolographyEngine:
    """Test cases for the main codec engine."""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        return GNNCodecHolographyEngine(
            phase_bits=8,
            amp_bits=4,
            gnn_hidden_dim=16,
            gnn_layers=2,
            use_graph_prediction=True
        )
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.phase_bits == 8
        assert engine.amp_bits == 4
        assert engine.use_graph_prediction == True
        assert hasattr(engine, 'gnn_predictor')
    
    def test_conv_weight_compression(self, engine):
        """Test compression of convolutional weights."""
        # Create a small conv weight tensor
        conv_weight = torch.randn(8, 4, 3, 3) * 0.1
        
        # Encode
        encoded = engine(conv_weight)
        
        assert encoded['success'] == True
        assert 'H_quantized' in encoded
        assert 'compression_stats' in encoded
        assert encoded['original_shape'] == conv_weight.shape
        
        # Decode
        decoded = engine.decode(encoded)
        
        assert decoded.shape == conv_weight.shape
        
        # Check compression ratio
        compression_info = engine.get_compression_info(encoded)
        assert compression_info['compression_ratio'] > 1.0
    
    def test_fc_weight_compression(self, engine):
        """Test compression of fully connected weights."""
        # Create a small FC weight tensor
        fc_weight = torch.randn(32, 64) * 0.05
        
        # Encode
        encoded = engine(fc_weight)
        
        assert encoded['success'] == True
        
        # Decode
        decoded = engine.decode(encoded)
        
        assert decoded.shape == fc_weight.shape
        
        # Check reconstruction quality
        mse_error = torch.mean((fc_weight - decoded) ** 2)
        assert mse_error < 1.0  # Reasonable reconstruction
    
    def test_compression_consistency(self, engine):
        """Test that compression is consistent across multiple runs."""
        weight = torch.randn(16, 32) * 0.1
        
        # Run compression multiple times
        results = []
        for _ in range(3):
            encoded = engine(weight)
            decoded = engine.decode(encoded)
            mse = torch.mean((weight - decoded) ** 2)
            results.append(mse.item())
        
        # Results should be identical (deterministic)
        assert len(set(results)) == 1
    
    def test_loss_computation(self, engine):
        """Test loss computation."""
        weight = torch.randn(16, 32) * 0.1
        
        encoded = engine(weight)
        decoded = engine.decode(encoded)
        
        # Compute loss
        loss = engine.compute_loss(weight, decoded, encoded)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Non-negative loss
    
    def test_gradient_flow(self, engine):
        """Test that gradients flow through the engine."""
        weight = torch.randn(8, 16, requires_grad=True) * 0.1
        
        # Forward pass
        encoded = engine(weight)
        decoded = engine.decode(encoded)
        
        # Compute loss
        loss = engine.compute_loss(weight, decoded, encoded)
        
        # Backward pass
        loss.backward()
        
        # Check that engine parameters have gradients
        has_grad = False
        for param in engine.parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "No gradients found in engine parameters"
    
    def test_device_compatibility(self):
        """Test that engine works on different devices."""
        # Test on CPU
        engine_cpu = GNNCodecHolographyEngine(gnn_hidden_dim=8, gnn_layers=1)
        weight_cpu = torch.randn(8, 16) * 0.1
        
        encoded = engine_cpu(weight_cpu)
        decoded = engine_cpu.decode(encoded)
        
        assert decoded.device == weight_cpu.device
        
        # Test on CUDA if available
        if torch.cuda.is_available():
            engine_cuda = engine_cpu.cuda()
            weight_cuda = weight_cpu.cuda()
            
            encoded = engine_cuda(weight_cuda)
            decoded = engine_cuda.decode(encoded)
            
            assert decoded.device == weight_cuda.device
    
    def test_compression_info(self, engine):
        """Test compression information extraction."""
        weight = torch.randn(16, 32) * 0.1
        encoded = engine(weight)
        
        info = engine.get_compression_info(encoded)
        
        required_keys = [
            'compression_ratio',
            'original_size_bits',
            'compressed_size_bits',
            'memory_savings_percent'
        ]
        
        for key in required_keys:
            assert key in info
            assert isinstance(info[key], (int, float))
        
        assert info['compression_ratio'] > 1.0
        assert 0 <= info['memory_savings_percent'] <= 100


if __name__ == "__main__":
    pytest.main([__file__])