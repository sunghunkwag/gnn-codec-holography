"""Complex-valued quantization for holographic representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import ComplexTensor
from typing import Optional
import numpy as np


class ComplexQuantizer(nn.Module):
    """Differentiable quantizer for complex-valued holographic representations.
    
    Args:
        phase_bits: Number of bits for phase quantization
        amp_bits: Number of bits for amplitude quantization
        temperature: Temperature parameter for soft quantization during training
    """
    
    def __init__(self, phase_bits: int = 8, amp_bits: int = 4, temperature: float = 1.0):
        super().__init__()
        self.phase_bits = phase_bits
        self.amp_bits = amp_bits
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        self.phase_levels = 2 ** phase_bits
        self.amp_levels = 2 ** amp_bits
    
    def forward(self, H: ComplexTensor, training: bool = True) -> ComplexTensor:
        """Apply quantization to complex tensor.
        
        Args:
            H: Complex tensor to quantize
            training: Whether in training mode (soft) or inference mode (hard)
            
        Returns:
            Quantized complex tensor
        """
        amplitude = H.abs()
        phase = H.angle()
        
        if training:
            # Soft quantization with straight-through estimator
            phase_quantized = self._soft_quantize(phase, self.phase_bits, phase_range=True)
            amp_quantized = self._soft_quantize(amplitude, self.amp_bits, phase_range=False)
        else:
            # Hard quantization for inference
            phase_quantized = self._hard_quantize(phase, self.phase_bits, phase_range=True)
            amp_quantized = self._hard_quantize(amplitude, self.amp_bits, phase_range=False)
        
        # Reconstruct complex representation
        H_q_real = amp_quantized * torch.cos(phase_quantized)
        H_q_imag = amp_quantized * torch.sin(phase_quantized)
        
        return ComplexTensor(H_q_real, H_q_imag)
    
    def _soft_quantize(self, x: torch.Tensor, bits: int, phase_range: bool = False) -> torch.Tensor:
        """Soft quantization using straight-through estimator.
        
        Args:
            x: Input tensor
            bits: Number of quantization bits
            phase_range: Whether input is phase ([-pi, pi]) or amplitude ([0, inf])
            
        Returns:
            Soft-quantized tensor
        """
        levels = 2 ** bits
        
        if phase_range:
            # Normalize phase from [-pi, pi] to [0, 1]
            x_norm = (x + torch.pi) / (2 * torch.pi)
        else:
            # Normalize amplitude to [0, 1] range
            x_min, x_max = torch.min(x), torch.max(x)
            if x_max > x_min:
                x_norm = (x - x_min) / (x_max - x_min)
            else:
                x_norm = torch.zeros_like(x)
        
        # Apply dithering for training
        if self.training:
            noise = (torch.rand_like(x_norm) - 0.5) / levels
            x_norm = x_norm + noise * 0.1
        
        # Quantize to discrete levels
        x_norm = torch.clamp(x_norm, 0, 1)
        x_discrete = x_norm * (levels - 1)
        
        # Straight-through estimator: round in forward, keep gradients
        x_quantized = torch.round(x_discrete) / (levels - 1)
        x_quantized = x_quantized + (x_discrete - x_discrete.detach())
        
        if phase_range:
            # Convert back to [-pi, pi]
            return x_quantized * 2 * torch.pi - torch.pi
        else:
            # Convert back to original range
            if x_max > x_min:
                return x_quantized * (x_max - x_min) + x_min
            else:
                return x
    
    def _hard_quantize(self, x: torch.Tensor, bits: int, phase_range: bool = False) -> torch.Tensor:
        """Hard quantization for inference.
        
        Args:
            x: Input tensor
            bits: Number of quantization bits
            phase_range: Whether input is phase ([-pi, pi]) or amplitude ([0, inf])
            
        Returns:
            Hard-quantized tensor
        """
        levels = 2 ** bits
        
        if phase_range:
            # Normalize phase
            x_norm = (x + torch.pi) / (2 * torch.pi)
        else:
            # Normalize amplitude
            x_min, x_max = torch.min(x), torch.max(x)
            if x_max > x_min:
                x_norm = (x - x_min) / (x_max - x_min)
            else:
                return x
        
        # Quantize
        x_norm = torch.clamp(x_norm, 0, 1)
        x_quantized = torch.round(x_norm * (levels - 1)) / (levels - 1)
        
        if phase_range:
            return x_quantized * 2 * torch.pi - torch.pi
        else:
            return x_quantized * (x_max - x_min) + x_min
    
    def get_compression_ratio(self) -> float:
        """Calculate theoretical compression ratio.
        
        Returns:
            Compression ratio compared to 32-bit float
        """
        total_bits = self.phase_bits + self.amp_bits
        return 32.0 / total_bits
    
    def estimate_bitrate(self, data_size: int) -> float:
        """Estimate compressed data size in bits.
        
        Args:
            data_size: Number of complex values to compress
            
        Returns:
            Estimated compressed size in bits
        """
        bits_per_sample = self.phase_bits + self.amp_bits
        return data_size * bits_per_sample