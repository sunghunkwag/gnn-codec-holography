"""Core components for holographic transformations and complex tensor operations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional
import numpy as np


class ComplexTensor:
    """Complex-valued tensor representation for holographic encoding.
    
    Args:
        real: Real part of complex tensor
        imag: Imaginary part of complex tensor
    """
    
    def __init__(self, real: torch.Tensor, imag: torch.Tensor):
        assert real.shape == imag.shape, "Real and imaginary parts must have same shape"
        self.real = real.float()
        self.imag = imag.float()
        self.device = real.device
        self.shape = real.shape
    
    def to(self, device: torch.device) -> 'ComplexTensor':
        """Move tensor to specified device."""
        return ComplexTensor(self.real.to(device), self.imag.to(device))
    
    def abs(self) -> torch.Tensor:
        """Compute magnitude of complex tensor."""
        return torch.sqrt(self.real**2 + self.imag**2)
    
    def angle(self) -> torch.Tensor:
        """Compute phase angle of complex tensor."""
        return torch.atan2(self.imag, self.real)
    
    def conj(self) -> 'ComplexTensor':
        """Compute complex conjugate."""
        return ComplexTensor(self.real, -self.imag)
    
    def __add__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self.real + other.real, self.imag + other.imag)
        else:
            return ComplexTensor(self.real + other, self.imag)
    
    def __sub__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            return ComplexTensor(self.real - other.real, self.imag - other.imag)
        else:
            return ComplexTensor(self.real - other, self.imag)
    
    def __mul__(self, other) -> 'ComplexTensor':
        if isinstance(other, ComplexTensor):
            # Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            real_part = self.real * other.real - self.imag * other.imag
            imag_part = self.real * other.imag + self.imag * other.real
            return ComplexTensor(real_part, imag_part)
        else:
            return ComplexTensor(self.real * other, self.imag * other)
    
    def size(self) -> int:
        """Return total number of elements."""
        return self.real.numel()


class HolographicSTE(Function):
    """Holographic Straight-Through Estimator for differentiable holographic encoding."""
    
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: float = 0.8, beta: float = 1.2) -> Tuple[torch.Tensor, torch.Tensor]:
        epsilon = 1e-8
        
        # Adaptive normalization
        input_std = torch.std(input_tensor) + epsilon
        input_norm = input_tensor / input_std
        
        magnitude = torch.abs(input_norm)
        
        # Stabilized amplitude encoding
        log_magnitude = torch.log(magnitude + epsilon)
        amplitude = alpha * torch.tanh(log_magnitude)
        
        # Multi-scale phase encoding
        phase_component1 = torch.pi * torch.tanh(input_norm * 2.0)
        phase_component2 = 0.2 * torch.pi * torch.sin(input_norm * 4.0)
        phase = beta * (phase_component1 + phase_component2)
        
        # Complex holographic representation
        H_real = amplitude * torch.cos(phase)
        H_imag = amplitude * torch.sin(phase)
        
        # Save context for backward pass
        ctx.save_for_backward(input_tensor, H_real, H_imag)
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.input_std = input_std
        
        return H_real, H_imag
    
    @staticmethod
    def backward(ctx, grad_real: torch.Tensor, grad_imag: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        # Straight-through estimator: pass gradients through directly
        grad_input = grad_real + grad_imag
        return grad_input, None, None


class HolographicTransform(nn.Module):
    """Holographic transformation layer for weight encoding.
    
    Args:
        alpha: Amplitude scaling parameter
        beta: Phase scaling parameter
    """
    
    def __init__(self, alpha: float = 0.8, beta: float = 1.2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.epsilon = 1e-8
    
    def forward(self, x: torch.Tensor) -> ComplexTensor:
        """Apply holographic transformation.
        
        Args:
            x: Input weight tensor
            
        Returns:
            Complex-valued holographic representation
        """
        H_real, H_imag = HolographicSTE.apply(x, self.alpha, self.beta)
        return ComplexTensor(H_real, H_imag)
    
    def inverse(self, H: ComplexTensor) -> torch.Tensor:
        """Inverse holographic transformation.
        
        Args:
            H: Complex holographic representation
            
        Returns:
            Reconstructed weight tensor
        """
        amplitude = H.abs()
        phase = H.angle()
        
        # Inverse transformation with numerical stability
        log_magnitude = torch.atanh(torch.clamp(amplitude / self.alpha, -0.99, 0.99))
        magnitude_restored = torch.exp(log_magnitude) - self.epsilon
        
        # Phase inversion (primary component only for stability)
        phase_normalized = torch.atanh(torch.clamp(phase / (self.beta * torch.pi), -0.99, 0.99))
        phase_restored = phase_normalized / 2.0
        
        # Combine magnitude and phase information
        magnitude_factor = magnitude_restored / (torch.abs(phase_restored) + self.epsilon)
        magnitude_factor = torch.clamp(magnitude_factor, 0.1, 10.0)
        
        reconstructed = phase_restored * magnitude_factor
        
        return reconstructed