# YAML Model Parser

Build PyTorch neural networks from YAML configuration files. Inspired by YOLOv5.

## Installation

```bash
# From GitHub
pip install git+https://github.com/YOUR_USERNAME/yaml-model-parser.git

# From source
pip install -e .
```

## Quick Start

```python
from yaml_model_parser import build_model_from_yaml

model = build_model_from_yaml('resnet18.yaml', input_channels=3)

# Use it
import torch
x = torch.randn(1, 3, 224, 224)
output = model(x)
```

## YAML Format

Each layer: `[from, repeat, module, args]`

- **from**: -1 (previous layer), layer index, or list for concatenation
- **repeat**: Number of times to repeat the module
- **module**: Layer type (Conv, ResBlock, Linear, etc.)
- **args**: Layer arguments (channels, kernel size, etc.)

### Example: ResNet18

```yaml
# resnet18.yaml
nc: 1000  # number of classes

backbone:
  - [-1, 1, Conv, [64, 7, 2, 3]]
  - [-1, 1, BatchNorm, []]
  - [-1, 1, ReLU, []]
  - [-1, 2, ResBlock, [64, 1]]       # repeat 2x
  - [-1, 1, ResBlock, [128, 2]]      # downsample

head:
  - [-1, 1, AdaptiveAvgPool, [1]]
  - [-1, 1, Flatten, []]
  - [-1, 1, Linear, [nc]]            # nc → 1000
```

### Example: YOLO with Skip Connections

```yaml
nc: 80
anchors: [[10,13], [16,30], [33,23]]

backbone:
  - [-1, 1, Conv, [64, 6, 2, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C3, [128]]

head:
  - [-1, 1, Conv, [256, 1, 1]]
  - [[-1, 2], 1, Concat, [1]]        # Concat layers -1 and 2
  - [-1, 3, C3, [256]]
  - [[4, 6], 1, Detect, [nc, anchors]]
```

## Available Layers

**YOLO-style**
- `Conv` - Conv2d + BatchNorm + SiLU
- `C3` - CSP Bottleneck
- `SPPF` - Spatial Pyramid Pooling
- `Concat` - Concatenate tensors
- `Detect` - YOLO detection head

**ResNet-style**
- `ResBlock` - Residual block
- `SEBlock` - Squeeze-and-Excitation block
- `Bottleneck` - Bottleneck block

**Standard PyTorch**
- `Linear`, `BatchNorm`, `ReLU`, `Sigmoid`
- `MaxPool`, `AvgPool`, `AdaptiveAvgPool`
- `Dropout`, `Flatten`, `Upsample`

## Advanced Usage

### Inspect Model

```python
from yaml_model_parser import ModelParser

parser = ModelParser('resnet18.yaml')
model, save_indices = parser.parse(input_channels=3)

print(f"Skip connection indices: {save_indices}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Add Custom Layer

1. Define layer in `yaml_model_parser/layers.py`:

```python
class MyLayer(nn.Module):
    def __init__(self, c_in, c_out, kernel):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel)

    def forward(self, x):
        return self.conv(x)

# Register it
LAYER_REGISTRY['MyLayer'] = MyLayer
```

2. Use in YAML:

```yaml
backbone:
  - [-1, 1, MyLayer, [64, 3]]
```

## Requirements

- Python >= 3.7
- PyTorch >= 1.9.0
- PyYAML >= 5.4.0
