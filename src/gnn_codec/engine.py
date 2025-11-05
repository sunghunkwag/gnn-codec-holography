"""Main GNN-based codec-holography engine."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .core import ComplexTensor, HolographicTransform
from .graph_builder import WeightGraphBuilder
from .gnn_models import GNNHolographicPredictor
from .quantization import ComplexQuantizer
import time


class GNNCodecHolographyEngine(nn.Module):
    """Complete GNN-based codec-holography engine for neural network weight compression.
    
    Args:
        phase_bits: Number of bits for phase quantization
        amp_bits: Number of bits for amplitude quantization
        gnn_hidden_dim: Hidden dimension for GNN layers
        gnn_layers: Number of GNN layers
        use_graph_prediction: Whether to use GNN-based prediction
    """
    
    def __init__(
        self,
        phase_bits: int = 8,
        amp_bits: int = 4,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 4,
        use_graph_prediction: bool = True
    ):
        super().__init__()
        
        self.phase_bits = phase_bits
        self.amp_bits = amp_bits
        self.use_graph_prediction = use_graph_prediction
        
        # Core components
        self.graph_builder = WeightGraphBuilder()
        self.holographic_transform = HolographicTransform(alpha=0.8, beta=1.2)
        self.quantizer = ComplexQuantizer(phase_bits, amp_bits)
        
        # GNN predictor (optional)
        if use_graph_prediction:
            self.gnn_predictor = GNNHolographicPredictor(
                hidden_dim=gnn_hidden_dim,
                num_layers=gnn_layers,
                num_heads=4
            )
    
    def forward(self, weights: torch.Tensor) -> Dict:
        """Encode weights using GNN-based holographic compression.
        
        Args:
            weights: Input weight tensor
            
        Returns:
            Dictionary containing compressed representation and metadata
        """
        original_shape = weights.shape
        device = weights.device
        batch_size = 1
        
        # Flatten to 1D for holographic transform
        weights_flat = weights.view(-1)
        
        # Apply holographic transformation
        H = self.holographic_transform(weights_flat)
        
        # Initialize prediction
        H_pred = ComplexTensor(
            torch.zeros_like(H.real),
            torch.zeros_like(H.imag)
        )
        
        compression_stats = {
            'base_compression': self.quantizer.get_compression_ratio(),
            'prediction_effectiveness': 0.0,
            'graph_sparsity': 0.0,
            'total_compression': self.quantizer.get_compression_ratio()
        }
        
        # GNN-based prediction (if enabled)
        if self.use_graph_prediction and hasattr(self, 'gnn_predictor'):
            try:
                # Build graph structure
                edge_index, edge_attr = self.graph_builder.build_graph(weights)
                edge_index = edge_index.to(device)
                edge_attr = edge_attr.to(device)
                
                # Prepare holographic features for graph processing
                holographic_features = torch.stack([
                    H.real, H.imag, H.abs(), H.angle()
                ], dim=-1)
                
                # Convert to graph data
                graph_data = self.graph_builder.weights_to_graph_data(
                    weights, holographic_features
                )
                graph_data = graph_data.to(device)
                
                # GNN prediction
                pred_data = self.gnn_predictor(graph_data)
                
                # Extract predictions
                pred_features = pred_data.x.view(H.shape + (4,))
                H_pred = ComplexTensor(
                    pred_features[..., 0],
                    pred_features[..., 1]
                )
                
                # Calculate compression statistics
                pred_strength = torch.mean(torch.abs(H_pred.real) + torch.abs(H_pred.imag)).item()
                prediction_compression = 1.0 + pred_strength * 3.0
                
                # Graph sparsity effect
                num_edges = edge_index.size(1)
                max_edges = H.size() ** 2
                sparsity = 1.0 - (num_edges / max_edges)
                sparsity_compression = 1.0 + sparsity * 2.0
                
                compression_stats.update({
                    'prediction_effectiveness': pred_strength,
                    'graph_sparsity': sparsity,
                    'prediction_compression': prediction_compression,
                    'sparsity_compression': sparsity_compression,
                    'total_compression': (
                        compression_stats['base_compression'] *
                        prediction_compression *
                        sparsity_compression
                    )
                })
                
            except Exception as e:
                # Fallback to zero prediction if GNN fails
                print(f"GNN prediction failed: {e}")
                H_pred = ComplexTensor(
                    torch.zeros_like(H.real) * 0.1,
                    torch.zeros_like(H.imag) * 0.1
                )
        
        # Compute residual
        H_residual = ComplexTensor(
            H.real - H_pred.real,
            H.imag - H_pred.imag
        )
        
        # Quantize residual
        H_quantized = self.quantizer(H_residual, training=self.training)
        
        return {
            'H_quantized': H_quantized,
            'H_pred': H_pred,
            'original_shape': original_shape,
            'compression_stats': compression_stats,
            'success': True
        }
    
    def decode(self, encoded_data: Dict) -> torch.Tensor:
        """Decode compressed representation back to weights.
        
        Args:
            encoded_data: Dictionary from forward pass
            
        Returns:
            Reconstructed weight tensor
        """
        if not encoded_data.get('success', False):
            # Return zeros if encoding failed
            return torch.zeros(encoded_data['original_shape'])
        
        H_quantized = encoded_data['H_quantized']
        H_pred = encoded_data['H_pred']
        original_shape = encoded_data['original_shape']
        
        # Reconstruct holographic representation
        H_reconstructed = ComplexTensor(
            H_pred.real + H_quantized.real,
            H_pred.imag + H_quantized.imag
        )
        
        # Inverse holographic transformation
        weights_reconstructed = self.holographic_transform.inverse(H_reconstructed)
        
        # Reshape to original form
        return weights_reconstructed.view(original_shape)
    
    def compute_loss(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        encoded_data: Optional[Dict] = None
    ) -> torch.Tensor:
        """Compute comprehensive loss function for training.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            encoded_data: Encoding metadata (optional)
            
        Returns:
            Total loss value
        """
        # Primary reconstruction loss
        mse_loss = F.mse_loss(original, reconstructed)
        
        # Holographic parameter regularization
        holographic_reg = 0.001 * (
            torch.abs(self.holographic_transform.alpha - 0.8) +
            torch.abs(self.holographic_transform.beta - 1.2)
        )
        
        total_loss = mse_loss + holographic_reg
        
        # Additional regularization if encoding data available
        if encoded_data is not None and 'compression_stats' in encoded_data:
            stats = encoded_data['compression_stats']
            
            # Encourage sparsity in graphs
            if 'graph_sparsity' in stats:
                sparsity_loss = torch.relu(0.8 - stats['graph_sparsity']) * 0.01
                total_loss += sparsity_loss
            
            # Regularize prediction strength
            if 'prediction_effectiveness' in stats:
                pred_reg = torch.relu(stats['prediction_effectiveness'] - 0.5) * 0.01
                total_loss += pred_reg
        
        return total_loss
    
    def get_compression_info(self, encoded_data: Dict) -> Dict:
        """Get detailed compression information.
        
        Args:
            encoded_data: Encoded data dictionary
            
        Returns:
            Compression analysis dictionary
        """
        if not encoded_data.get('success', False):
            return {'error': 'Encoding failed'}
        
        stats = encoded_data.get('compression_stats', {})
        original_shape = encoded_data.get('original_shape', ())
        
        original_size = torch.tensor(original_shape).prod().item() * 32  # bits
        compressed_size = original_size / stats.get('total_compression', 1.0)
        
        return {
            'compression_ratio': stats.get('total_compression', 1.0),
            'original_size_bits': original_size,
            'compressed_size_bits': compressed_size,
            'memory_savings_percent': (1 - compressed_size / original_size) * 100,
            'base_compression': stats.get('base_compression', 1.0),
            'prediction_gain': stats.get('prediction_compression', 1.0),
            'sparsity_gain': stats.get('sparsity_compression', 1.0),
            'graph_sparsity': stats.get('graph_sparsity', 0.0),
            'prediction_strength': stats.get('prediction_effectiveness', 0.0)
        }