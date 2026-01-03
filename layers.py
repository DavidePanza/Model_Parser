"""
Neural network layer definitions for model building.
Contains common building blocks like Conv, C3, SPPF, etc.
"""

import torch
import torch.nn as nn
from typing import List


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
    def __init__(self, c_in: int, c_out: int, kernel: int = 1, stride: int = 1, 
                 padding: int = None, groups: int = 1, activation: bool = True):
        """
        Args:
            c_in: Input channels
            c_out: Output channels
            kernel: Kernel size
            stride: Stride
            padding: Padding (auto if None)
            groups: Groups for grouped convolution
            activation: Whether to use activation
        """
        super().__init__()
        padding = kernel // 2 if padding is None else padding
        
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride, padding, 
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    
    def __init__(self, c_in: int, c_out: int, shortcut: bool = True, 
                 groups: int = 1, expansion: float = 0.5):
        """
        Args:
            c_in: Input channels
            c_out: Output channels
            shortcut: Whether to use residual connection
            groups: Groups for convolution
            expansion: Channel expansion ratio
        """
        super().__init__()
        c_hidden = int(c_out * expansion)
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_hidden, c_out, 3, 1, groups=groups)
        self.add = shortcut and c_in == c_out
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    
    def __init__(self, c_in: int, c_out: int, num_bottlenecks: int = 1, 
                 shortcut: bool = True, groups: int = 1, expansion: float = 0.5):
        """
        Args:
            c_in: Input channels
            c_out: Output channels
            num_bottlenecks: Number of bottleneck layers
            shortcut: Whether to use shortcuts in bottlenecks
            groups: Groups for convolution
            expansion: Channel expansion ratio
        """
        super().__init__()
        c_hidden = int(c_out * expansion)
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_in, c_hidden, 1, 1)
        self.cv3 = Conv(2 * c_hidden, c_out, 1, 1)
        
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(c_hidden, c_hidden, shortcut, groups, expansion=1.0) 
              for _ in range(num_bottlenecks)]
        )
    
    def forward(self, x):
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, c_in: int, c_out: int, kernel_size: int = 5):
        """
        Args:
            c_in: Input channels
            c_out: Output channels
            kernel_size: Pooling kernel size
        """
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = Conv(c_in, c_hidden, 1, 1)
        self.cv2 = Conv(c_hidden * 4, c_out, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Concat(nn.Module):
    """Concatenate tensors along a dimension."""
    
    def __init__(self, dimension: int = 1):
        """
        Args:
            dimension: Dimension to concatenate along (default: 1 for channels)
        """
        super().__init__()
        self.dim = dimension
    
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: List of tensors to concatenate
        """
        return torch.cat(x, dim=self.dim)


class Detect(nn.Module):
    """YOLOv5 detection head."""
    
    def __init__(self, nc: int, anchors: List[List[int]], ch: List[int]):
        """
        Args:
            nc: Number of classes
            anchors: Anchor boxes for each scale
            ch: List of input channels from different scales [P3, P4, P5]
        """
        super().__init__()
        self.nc = nc  # Number of classes
        self.no = nc + 5  # Number of outputs per anchor (classes + x,y,w,h,obj)
        self.nl = len(anchors)  # Number of detection layers
        self.na = len(anchors[0]) // 2  # Number of anchors per layer
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # Grid cache
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # Anchor grid cache
        
        # Register anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        
        # Detection head convolutions
        self.m = nn.ModuleList(
            nn.Conv2d(ch[i], self.no * self.na, 1) for i in range(self.nl)
        )
    
    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: List of feature maps from different scales [P3, P4, P5]
        
        Returns:
            List of predictions for each scale
        """
        z = []  # Inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # Apply detection conv
            bs, _, ny, nx = x[i].shape  # Batch, channels, height, width
            
            # Reshape: (bs, na*no, ny, nx) -> (bs, na, ny, nx, no)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            z.append(x[i].view(bs, -1, self.no))
        
        return torch.cat(z, dim=1) if self.training else (torch.cat(z, dim=1), x)


class Upsample(nn.Module):
    """Wrapper for nn.Upsample to handle different argument formats."""
    
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        """
        Args:
            size: Target spatial size (int or tuple)
            scale_factor: Multiplier for spatial size (float or tuple)
            mode: Upsampling mode ('nearest', 'bilinear', etc.)
        """
        super().__init__()
        
        # Only pass non-None parameter to avoid conflict
        if size is not None:
            self.upsample = nn.Upsample(size=size, mode=mode)
        elif scale_factor is not None:
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        else:
            raise ValueError("Either 'size' or 'scale_factor' must be specified")
    
    def forward(self, x):
        return self.upsample(x)


# Layer registry for easy lookup
LAYER_REGISTRY = {
    # Standard PyTorch layers
    'Conv': self._make_conv,
    'Linear': self._make_linear,
    'BatchNorm': self._make_batchnorm,
    'ReLU': lambda args: nn.ReLU(inplace=True),
    'Sigmoid': lambda args: nn.Sigmoid(),
    'MaxPool': self._make_maxpool,
    'AvgPool': lambda args: nn.AvgPool2d(*args),
    'AdaptiveAvgPool': self._make_adaptive_pool,
    'Dropout': lambda args: nn.Dropout(args[0] if args else 0.5),
    'Flatten': self._make_flatten,
    
    # Custom modules
    'Concat': lambda args: Concat(args[0] if args else 1),
    'ResBlock': self._make_resblock,
    'Conv': Conv,
    'Bottleneck': Bottleneck,
    'C3': C3,
    'SPPF': SPPF,
    'Concat': Concat,
    'Detect': Detect,
    'Upsample': Upsample,
    'nn.Upsample': Upsample,  # Alias
}


def get_layer(name: str):
    """
    Get layer class by name, handling 'nn.' prefix.
    
    Args:
        name: Layer name (e.g., 'Conv', 'C3', 'SPPF', 'nn.Upsample')
    
    Returns:
        Layer class
    
    Raises:
        ValueError: If layer name not found
    """
    # Strip 'nn.' prefix if present
    if name.startswith('nn.'):
        name = name[3:]
    
    if name not in LAYER_REGISTRY:
        raise ValueError(f"Layer '{name}' not found. Available layers: {list(LAYER_REGISTRY.keys())}")
    return LAYER_REGISTRY[name]