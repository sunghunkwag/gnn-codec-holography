"""GNN-based Codec-Holography Engine.

A neural network weight compression system using Graph Neural Networks
and holographic representations.
"""

__version__ = "0.1.0"
__author__ = "Sung hun kwag"

from .core import ComplexTensor, HolographicTransform
from .graph_builder import WeightGraphBuilder
from .gnn_models import ComplexGraphConv, ComplexGraphAttention, GNNHolographicPredictor
from .quantization import ComplexQuantizer
from .engine import GNNCodecHolographyEngine
from .training import GNNTrainingSystem

__all__ = [
    "ComplexTensor",
    "HolographicTransform", 
    "WeightGraphBuilder",
    "ComplexGraphConv",
    "ComplexGraphAttention",
    "GNNHolographicPredictor",
    "ComplexQuantizer",
    "GNNCodecHolographyEngine",
    "GNNTrainingSystem",
]