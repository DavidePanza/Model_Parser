# YAML Model Parser

A minimalistic and flexible PyTorch model parser that builds neural networks from YAML configuration files. Inspired by YOLOv5's configuration system, this library allows you to define complex architectures declaratively.

## Features

- **Simple YAML syntax** for defining model architectures
- **Automatic channel inference** - no need to manually track tensor dimensions
- **Skip connections** - automatic handling of concatenation and residual connections
- **Extensible layer registry** - easy to add custom layers
- **Support for standard architectures** - ResNet, YOLO-style models, and more
- **Built-in layers**: Conv, C3, SPPF, ResBlock, SEBlock, Detect, and standard PyTorch layers

## Installation

### From source (development mode)

```bash
cd "/path/to/Yaml Model Creations"
pip install -e .
```

### For users

```bash
pip install yaml-model-parser
```

## 📁 File Structure

```
.
├── yaml_model_parser/
│   ├── __init__.py
│   ├── layers.py           # Layer definitions (Conv, C3, SPPF, Detect, etc.)
│   └── model_parser.py     # Main parser that builds models from YAML
├── setup.py                # Package installation script
└── README.md               # This file
```

## 🚀 Quick Start

### Basic Usage

```python
from yaml_model_parser import build_model_from_yaml
import torch

# Build model from YAML
model = build_model_from_yaml('resnet18.yaml', input_channels=3)

# Use the model
input_image = torch.randn(1, 3, 224, 224)
output = model(input_image)
print(f"Output shape: {output.shape}")  # [1, 1000]
```

### YAML Format

```yaml
# Parameters
nc: 80  # number of classes
anchors:
  - [10, 13, 16, 30, 33, 23]  # P3/8
  - [30, 61, 62, 45, 59, 119]  # P4/16
  - [116, 90, 156, 198, 373, 326]  # P5/32

# Backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]],  # 0
    [-1, 1, Conv, [128, 3, 2]],     # 1
    [-1, 3, C3, [128]],              # 2
    # ... more layers
  ]

# Head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [[-1, 6], 1, Concat, [1]],  # Skip connection
    # ... more layers
    [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detection head
  ]
```

## 📚 Layer Definitions

### Available Layers

All layer definitions are in `layers.py`:

| Layer | Description | Arguments |
|-------|-------------|-----------|
| `Conv` | Conv2d + BatchNorm + SiLU | `[c_out, kernel, stride, padding]` |
| `C3` | CSP Bottleneck | `[c_out, shortcut]` |
| `SPPF` | Spatial Pyramid Pooling | `[c_out, kernel_size]` |
| `Concat` | Concatenate tensors | `[dimension]` |
| `Upsample` | Upsample layer | `[size, scale_factor, mode]` |
| `Detect` | YOLO detection head | `[nc, anchors]` |

### Adding Custom Layers

1. **Define your layer in `layers.py`:**

```python
class MyCustomLayer(nn.Module):
    def __init__(self, c_in, c_out, my_param):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.my_param = my_param
    
    def forward(self, x):
        return self.conv(x) * self.my_param
```

2. **Register it in the layer registry:**

```python
LAYER_REGISTRY['MyCustomLayer'] = MyCustomLayer
```

3. **Use it in your YAML:**

```yaml
backbone:
  [
    [-1, 1, MyCustomLayer, [64, 2.0]],  # [c_out, my_param]
  ]
```

## 🔧 How It Works

### 1. Parser Flow

```
YAML file → ModelParser → YOLOModel → Forward Pass
```

### 2. Key Components

**`ModelParser` class:**
- Loads YAML configuration
- Iterates through layer definitions
- Builds each layer using `layers.py`
- Tracks channel dimensions
- Identifies skip connections

**`YOLOModel` class:**
- Wraps all layers
- Handles forward pass with skip connections
- Manages layer outputs for concatenation

### 3. Skip Connections

The parser automatically handles skip connections:

```yaml
# Layer 12 concatenates output from layer -1 (previous) and layer 6
[[-1, 6], 1, Concat, [1]]
```

Internally:
1. Parser identifies layers 6 and -1 need to be saved
2. During forward pass, outputs are stored
3. Concat layer receives both outputs and concatenates them

## 📖 Examples

### Example 1: Basic Model Building

```python
from model_parser import build_model_from_yaml

model = build_model_from_yaml('yolov5n.yaml')
```

### Example 2: Inspect Architecture

```python
from yaml_model_parser import ModelParser

parser = ModelParser('resnet18.yaml')
model, save_indices = parser.parse(input_channels=3)

print(f"Layers with skip connections: {save_indices}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Example 3: Training Loop

```python
import torch
import torch.nn as nn
from yaml_model_parser import build_model_from_yaml

# Build model
model = build_model_from_yaml('resnet18.yaml')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(10):
    for images, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
```

### Example 4: Different Input Sizes

```python
model = build_model_from_yaml('yolov5n.yaml')
model.eval()

# Works with any input size (divisible by 32)
for size in [320, 480, 640]:
    x = torch.randn(1, 3, size, size)
    out = model(x)
    print(f"{size}x{size} → {out.shape}")
```

## 🎯 Key Features

✅ **Simple and Clean** - Easy to understand codebase  
✅ **No Scaling Factors** - Direct channel/layer counts from YAML  
✅ **Extensible** - Easy to add new layer types  
✅ **Skip Connections** - Automatic handling of concatenations  
✅ **Flexible** - Works with any YAML structure  
✅ **Well Documented** - Clear comments and examples  

## 🔍 Differences from Original YOLOv5

This implementation **removes**:
- ❌ `depth_multiple` and `width_multiple` scaling
- ❌ Complex auto-anchor calculations
- ❌ Model export features
- ❌ Training-specific utilities

This implementation **keeps**:
- ✅ Core layer building logic
- ✅ Skip connection handling
- ✅ Detection head structure
- ✅ Clean, readable code

## 🛠️ Extending the Parser

### Add Support for New Module Types

In `model_parser.py`, add a new case in `_build_layer()`:

```python
def _build_layer(self, module_name, args, c_in, num_repeats):
    # ... existing code ...
    
    elif module_name == 'MyNewLayer':
        c_out = args[0]
        custom_param = args[1]
        module = LayerClass(c_in, c_out, custom_param)
    
    # ... rest of code ...
```

### Modify Forward Pass Logic

If you need custom forward pass logic, modify `YOLOModel.forward()`:

```python
def forward(self, x):
    outputs = []
    
    for i, layer in enumerate(self.layers):
        # Add your custom logic here
        if layer.layer_idx == 10:
            x = self.custom_processing(x)
        
        # ... rest of forward logic ...
```

## 📝 Requirements

```
torch >= 1.8.0
pyyaml >= 5.4.0
```

## 🤝 Contributing

To add a new layer type:
1. Add layer class to `layers.py`
2. Register in `LAYER_REGISTRY`
3. Add parsing logic in `model_parser.py` if needed
4. Add example to `example_usage.py`
5. Update this README

## 📄 License

This is a simplified educational implementation. For production use, refer to the official YOLOv5 repository.

## 💡 Tips

- **Channel Tracking**: The parser automatically tracks output channels of each layer
- **Error Messages**: If a layer fails, check the YAML format and arguments
- **Debugging**: Use `print()` statements in `_build_layer()` to see what's happening
- **Testing**: Always test with a small input first: `torch.randn(1, 3, 32, 32)`

## 🎓 Learning Resources

To understand this code better, study these concepts:
1. PyTorch `nn.Module` basics
2. YAML file format
3. Skip connections in neural networks
4. Object detection architectures (YOLO)

---

**Happy model building! 🚀**