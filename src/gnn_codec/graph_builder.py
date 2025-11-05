"""Graph construction utilities for converting neural network weights to graph structures."""

import torch
import numpy as np
from typing import Tuple, Union
from torch_geometric.data import Data


class WeightGraphBuilder:
    """Converts neural network weights into graph structures for GNN processing.
    
    Args:
        spatial_radius: Radius for spatial neighborhood connections
        channel_connections: Number of channel-wise connections per node
    """
    
    def __init__(self, spatial_radius: int = 1, channel_connections: int = 8):
        self.spatial_radius = spatial_radius
        self.channel_connections = channel_connections
    
    def build_conv_graph(self, weight_shape: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph structure for convolutional layer weights.
        
        Args:
            weight_shape: Shape of convolutional weights (out_ch, in_ch, h, w)
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        out_ch, in_ch, h, w = weight_shape
        num_nodes = out_ch * in_ch * h * w
        
        edges = []
        edge_features = []
        
        # Spatial connections within kernel
        for o in range(out_ch):
            for i in range(in_ch):
                for y in range(h):
                    for x in range(w):
                        node_id = o * in_ch * h * w + i * h * w + y * w + x
                        
                        # Connect to spatial neighbors
                        for dy in range(-self.spatial_radius, self.spatial_radius + 1):
                            for dx in range(-self.spatial_radius, self.spatial_radius + 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < h and 0 <= nx < w and (dy != 0 or dx != 0):
                                    neighbor_id = o * in_ch * h * w + i * h * w + ny * w + nx
                                    
                                    edges.append([node_id, neighbor_id])
                                    distance = np.sqrt(dy**2 + dx**2)
                                    edge_features.append([distance, 1.0])  # [distance, spatial_type]
        
        # Channel connections
        for o in range(out_ch):
            for y in range(h):
                for x in range(w):
                    # Connect across input channels
                    channel_connections = min(self.channel_connections, in_ch)
                    for i1 in range(in_ch):
                        node_id1 = o * in_ch * h * w + i1 * h * w + y * w + x
                        
                        # Connect to nearby channels
                        start_ch = max(0, i1 - channel_connections // 2)
                        end_ch = min(in_ch, i1 + channel_connections // 2)
                        
                        for i2 in range(start_ch, end_ch):
                            if i1 != i2:
                                node_id2 = o * in_ch * h * w + i2 * h * w + y * w + x
                                
                                edges.append([node_id1, node_id2])
                                channel_distance = abs(i1 - i2)
                                edge_features.append([channel_distance, 2.0])  # [distance, channel_type]
        
        # Output channel connections (limited for computational efficiency)
        max_out_connections = min(out_ch, 16)
        for i in range(in_ch):
            for y in range(h):
                for x in range(w):
                    for o1 in range(max_out_connections):
                        for o2 in range(o1 + 1, max_out_connections):
                            node_id1 = o1 * in_ch * h * w + i * h * w + y * w + x
                            node_id2 = o2 * in_ch * h * w + i * h * w + y * w + x
                            
                            edges.append([node_id1, node_id2])
                            output_distance = abs(o1 - o2)
                            edge_features.append([output_distance, 3.0])  # [distance, output_type]
        
        if not edges:
            # Fallback: create simple linear connections
            edges = [[i, i + 1] for i in range(num_nodes - 1)]
            edge_features = [[1.0, 0.0] for _ in range(num_nodes - 1)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def build_fc_graph(self, weight_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph structure for fully connected layer weights.
        
        Args:
            weight_shape: Shape of FC weights (out_features, in_features)
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        out_features, in_features = weight_shape
        num_nodes = out_features * in_features
        
        edges = []
        edge_features = []
        
        # Input feature connections (within same output neuron)
        for out_idx in range(out_features):
            for in_idx in range(in_features):
                node_id = out_idx * in_features + in_idx
                
                # Connect to nearby input features
                neighbor_range = min(8, in_features // 4)  # Adaptive neighborhood size
                start_in = max(0, in_idx - neighbor_range)
                end_in = min(in_features, in_idx + neighbor_range)
                
                for neighbor_in in range(start_in, end_in):
                    if neighbor_in != in_idx:
                        neighbor_id = out_idx * in_features + neighbor_in
                        
                        edges.append([node_id, neighbor_id])
                        distance = abs(in_idx - neighbor_in)
                        edge_features.append([distance, 1.0])  # [distance, input_type]
        
        # Output neuron connections (limited for efficiency)
        max_out_connections = min(out_features, 32)
        for in_idx in range(in_features):
            for out_idx in range(max_out_connections):
                node_id1 = out_idx * in_features + in_idx
                
                # Connect to nearby output neurons
                neighbor_range = min(4, max_out_connections // 8)
                start_out = max(0, out_idx - neighbor_range)
                end_out = min(max_out_connections, out_idx + neighbor_range)
                
                for neighbor_out in range(start_out, end_out):
                    if neighbor_out != out_idx:
                        node_id2 = neighbor_out * in_features + in_idx
                        
                        edges.append([node_id1, node_id2])
                        distance = abs(out_idx - neighbor_out)
                        edge_features.append([distance, 2.0])  # [distance, output_type]
        
        if not edges:
            # Fallback: create simple linear connections
            edges = [[i, i + 1] for i in range(num_nodes - 1)]
            edge_features = [[1.0, 0.0] for _ in range(num_nodes - 1)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return edge_index, edge_attr
    
    def build_graph(self, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build appropriate graph structure based on weight tensor shape.
        
        Args:
            weights: Weight tensor to convert to graph
            
        Returns:
            Tuple of (edge_index, edge_attr)
        """
        if len(weights.shape) == 4:  # Convolutional layer
            return self.build_conv_graph(weights.shape)
        elif len(weights.shape) == 2:  # Fully connected layer
            return self.build_fc_graph(weights.shape)
        else:
            # Generic 1D or other shapes: simple linear graph
            num_nodes = weights.numel()
            edges = [[i, i + 1] for i in range(num_nodes - 1)]
            edge_features = [[1.0, 0.0] for _ in range(num_nodes - 1)]
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
            
            return edge_index, edge_attr
    
    def weights_to_graph_data(self, weights: torch.Tensor, holographic_features: torch.Tensor) -> Data:
        """Convert weights and holographic features to PyTorch Geometric Data object.
        
        Args:
            weights: Original weight tensor
            holographic_features: Complex holographic features [real, imag, magnitude, phase]
            
        Returns:
            PyTorch Geometric Data object
        """
        edge_index, edge_attr = self.build_graph(weights)
        
        # Flatten holographic features to node features
        node_features = holographic_features.view(-1, holographic_features.size(-1))
        
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)