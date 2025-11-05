"""Graph Neural Network models for holographic weight prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from typing import Optional
import math


class ComplexGraphConv(MessagePassing):
    """Complex-valued graph convolution layer for holographic representations.
    
    Args:
        in_channels: Number of input node features
        out_channels: Number of output node features  
        edge_dim: Number of edge features
    """
    
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 2):
        super().__init__(aggr='mean', flow='source_to_target')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Separate networks for real and imaginary components
        self.real_transform = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        self.imag_transform = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Self-connection transformation
        self.self_loop = nn.Linear(in_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        for module in [self.real_transform, self.imag_transform, self.self_loop]:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
            else:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass through complex graph convolution.
        
        Args:
            x: Node features [num_nodes, 4] with [real, imag, magnitude, phase]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, 4]
        """
        # Add self-loops for message passing
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value=0.0, num_nodes=x.size(0)
        )
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Add self-connection
        out += self.self_loop(x)
        
        return out
    
    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Generate messages considering complex-valued structure.
        
        Args:
            x_j: Source node features
            edge_attr: Edge attributes
            
        Returns:
            Messages to be aggregated
        """
        # Combine node features with edge attributes
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        
        # Separate real and imaginary message computation
        real_msg = self.real_transform(msg_input)
        imag_msg = self.imag_transform(msg_input)
        
        # Reconstruct complex representation [real, imag, magnitude, phase]
        magnitude = torch.sqrt(real_msg**2 + imag_msg**2)
        phase = torch.atan2(imag_msg, real_msg + 1e-8)
        
        return torch.cat([real_msg, imag_msg, magnitude, phase], dim=-1)


class ComplexGraphAttention(nn.Module):
    """Multi-head graph attention mechanism for complex-valued features.
    
    Args:
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(4, hidden_dim)  # 4: [real, imag, mag, phase]
        self.k_proj = nn.Linear(4, hidden_dim)
        self.v_proj = nn.Linear(4, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, 4)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to graph nodes.
        
        Args:
            x: Node features [num_nodes, 4]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Attention-weighted features [num_nodes, 4]
        """
        num_nodes = x.size(0)
        
        # Project to query, key, value
        Q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Extract edge connections
        row, col = edge_index
        
        # Compute attention for connected nodes only
        attention_outputs = []
        
        for head in range(self.num_heads):
            # Get head-specific Q, K, V
            q_head = Q[:, head, :]  # [num_nodes, head_dim]
            k_head = K[:, head, :]
            v_head = V[:, head, :]
            
            # Compute attention scores for edges
            scores = torch.sum(q_head[row] * k_head[col], dim=1) / math.sqrt(self.head_dim)
            attention_weights = F.softmax(scores, dim=0)
            
            # Apply attention to aggregate messages
            attended = torch.zeros_like(v_head)
            for i in range(len(row)):
                attended[col[i]] += attention_weights[i] * v_head[row[i]]
            
            attention_outputs.append(attended)
        
        # Concatenate multi-head outputs
        attended_out = torch.cat(attention_outputs, dim=-1)  # [num_nodes, hidden_dim]
        
        # Final projection
        output = self.out_proj(attended_out)
        output = self.dropout(output)
        
        # Residual connection
        return output + x


class GNNHolographicPredictor(nn.Module):
    """Complete GNN-based predictor for holographic weight compression.
    
    Args:
        hidden_dim: Hidden dimension for internal layers
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
    """
    
    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(4, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ComplexGraphConv(hidden_dim, hidden_dim, edge_dim=2)
            for _ in range(num_layers)
        ])
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            ComplexGraphAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(4) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4),  # [real, imag, mag, phase]
            nn.Tanh()  # Bound output range
        )
        
        # Residual connection weights
        self.residual_weights = nn.Parameter(torch.ones(num_layers) * 0.1)
        
    def forward(self, data: Data) -> Data:
        """Forward pass through GNN predictor.
        
        Args:
            data: PyTorch Geometric data with node features and edge information
            
        Returns:
            Data with predicted holographic features
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # Input projection to hidden dimension
        h = self.input_proj(x)
        
        # Process through GNN and attention layers
        for i in range(self.num_layers):
            residual = x
            
            # GNN convolution
            h = self.gnn_layers[i](h, edge_index, edge_attr)
            
            # Project back to 4D for attention
            x_4d = h[:, :4]  # Use first 4 dimensions
            
            # Apply attention
            x_4d = self.attention_layers[i](x_4d, edge_index)
            
            # Layer normalization
            x_4d = self.layer_norms[i](x_4d)
            
            # Residual connection with learnable weight
            x = x_4d + residual * self.residual_weights[i]
            
            # Re-project to hidden dimension for next layer
            h = self.input_proj(x)
        
        # Final prediction
        prediction = self.output_proj(h)
        
        # Scale prediction strength to prevent over-prediction
        prediction = prediction * 0.3
        
        return Data(x=prediction, edge_index=edge_index, edge_attr=edge_attr)