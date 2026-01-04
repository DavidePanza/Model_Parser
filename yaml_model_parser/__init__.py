"""
YAML Model Parser - Build PyTorch models from YAML configurations
"""

from .model_parser import ModelParser, Model, build_model_from_yaml
from .layers import (
    get_layer,
    LAYER_REGISTRY,
    Conv,
    C3,
    SPPF,
    Concat,
    Detect,
    Bottleneck,
    ResBlock,
    SEBlock
)

__version__ = "0.1.0"
__all__ = [
    "ModelParser",
    "Model",
    "build_model_from_yaml",
    "get_layer",
    "LAYER_REGISTRY",
    "Conv",
    "C3",
    "SPPF",
    "Concat",
    "Detect",
    "Bottleneck",
    "ResBlock",
    "SEBlock",
]
