"""Holographic complex-valued transforms for weight encoding."""

import torch
import torch.nn.functional as F
from typing import Tuple

class HolographicTransform:
    """Encodes real-valued weights into complex-valued holographic features.
    Outputs stacked features: [real, imag, magnitude, phase].
    """
    def __init__(self, freq_scale: float = 1.0):
        self.freq_scale = freq_scale

    def encode(self, weights: torch.Tensor) -> torch.Tensor:
        w = weights.float()
        # Simple Fourier-like projection (demo placeholder)
        real = w
        imag = torch.tanh(w * self.freq_scale)
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real)
        return torch.stack([real, imag, magnitude, phase], dim=-1)

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        real = encoded[..., 0]
        imag = encoded[..., 1]
        # Inverse of the simple projection (demo placeholder)
        return real
