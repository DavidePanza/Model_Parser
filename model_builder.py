import yaml
import torch
import torch.nn as nn
from pathlib import Path

class GenericModelParser:
    """Parse any architecture from YAML"""
    
    def __init__(self, yaml_path, num_classes=1000):
        with open(yaml_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.num_classes = num_classes
        
        # Track current tensor shape through the network
        self.current_shape = self.cfg['input_shape']  # [C, H, W]
        
        # Module registry
        self.modules = {
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
            # 'AttentionBlock': self._make_attention,
        }
        
        self.layers = nn.ModuleList()
        self.save = []
    
    def _update_shape_conv(self, out_channels, kernel, stride, padding):
        """Update shape after convolution"""
        c, h, w = self.current_shape
        h_out = (h + 2 * padding - kernel) // stride + 1
        w_out = (w + 2 * padding - kernel) // stride + 1
        self.current_shape = [out_channels, h_out, w_out]
    
    def _update_shape_pool(self, kernel, stride, padding):
        """Update shape after pooling"""
        c, h, w = self.current_shape
        h_out = (h + 2 * padding - kernel) // stride + 1
        w_out = (w + 2 * padding - kernel) // stride + 1
        self.current_shape = [c, h_out, w_out]
    
    def _make_conv(self, args):
        """Create Conv2d with automatic in_channels"""
        out_ch, k, s = args[:3]
        p = args[3] if len(args) > 3 else k // 2
        in_ch = self.current_shape[0]
        
        self._update_shape_conv(out_ch, k, s, p)
        return nn.Conv2d(in_ch, out_ch, k, s, p)
    
    def _make_maxpool(self, args):
        """Create MaxPool2d"""
        k, s = args[:2]
        p = args[2] if len(args) > 2 else 0
        
        self._update_shape_pool(k, s, p)
        return nn.MaxPool2d(k, s, p)
    
    def _make_adaptive_pool(self, args):
        """Create AdaptiveAvgPool2d"""
        output_size = tuple(args)
        c = self.current_shape[0]
        self.current_shape = [c, output_size[0], output_size[1]]
        return nn.AdaptiveAvgPool2d(output_size)
    
    def _make_flatten(self, args):
        """Create Flatten and update shape"""
        c, h, w = self.current_shape
        flattened_size = c * h * w
        self.current_shape = [flattened_size]  # Now 1D
        return nn.Flatten()
    
    def _make_linear(self, args):
        """Create Linear layer"""
        out_features = args[0]
        
        # Get input features (should be flattened by now)
        if len(self.current_shape) == 1:
            in_features = self.current_shape[0]
        else:
            # If not flattened, calculate total size
            in_features = self.current_shape[0] * self.current_shape[1] * self.current_shape[2]
        
        self.current_shape = [out_features]
        return nn.Linear(in_features, out_features)
    
    def _make_batchnorm(self, args):
        """Create BatchNorm2d"""
        num_features = args[0] if args else self.current_shape[0]
        return nn.BatchNorm2d(num_features)
    
    def _make_resblock(self, args):
        """Create residual block"""
        in_ch, out_ch = args[:2]
        stride = args[2] if len(args) > 2 else 1
        
        # Update shape
        c, h, w = self.current_shape
        if stride > 1:
            h = h // stride
            w = w // stride
        self.current_shape = [out_ch, h, w]
        
        return ResidualBlock(in_ch, out_ch, stride)
    
    # def _make_attention(self, args):
    #     """Create attention mechanism"""
    #     channels = args[0]
    #     return AttentionBlock(channels)
    
    def parse_layer(self, from_idx, repeat, module_name, args):
        """Parse a single layer specification"""
        # Replace 'num_classes' placeholder
        args = [self.num_classes if a == 'num_classes' else a for a in args]
        
        # Get module builder
        module_builder = self.modules.get(module_name)
        if not module_builder:
            raise ValueError(f"Unknown module: {module_name}")
        
        # Single input
        if repeat > 1:
            layers = []
            for _ in range(repeat):
                layers.append(module_builder(args))
            layer = nn.Sequential(*layers)
        else:
            layer = module_builder(args)
        
        return layer, from_idx
    
    def build(self):
        """Build complete model from config"""
        print(f"Building model from {self.cfg['model_name']}")
        print(f"Input shape: {self.current_shape}")
        
        # Build backbone
        for i, layer_cfg in enumerate(self.cfg['backbone']):
            from_idx, repeat, module_name, args = layer_cfg
            print(f"\nLayer {i}: {module_name} (repeat={repeat})")
            print(f"  Before: {self.current_shape}")
            
            layer, save_idx = self.parse_layer(from_idx, repeat, module_name, args)
            self.layers.append(layer)
            
            print(f"  After: {self.current_shape}")
            
            if save_idx not in [-1]:
                self.save.append(len(self.layers) - 1)
        
        # Build head if present
        if 'head' in self.cfg:
            for i, layer_cfg in enumerate(self.cfg['head']):
                from_idx, repeat, module_name, args = layer_cfg
                print(f"\nHead {i}: {module_name}")
                print(f"  Before: {self.current_shape}")
                
                layer, _ = self.parse_layer(from_idx, repeat, module_name, args)
                self.layers.append(layer)
                
                print(f"  After: {self.current_shape}")
        
        print(f"\nFinal output shape: {self.current_shape}")
        return Model(self.layers, self.save)


class Model(nn.Module):
    """Generic model that can execute any architecture"""
    
    def __init__(self, layers, save_indices):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.save = save_indices
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    
    def forward(self, x):
        return torch.cat(x, self.d)


class ResidualBlock(nn.Module):
    """Simple residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.out_channels = out_channels
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# class AttentionBlock(nn.Module):
#     """Simple attention block"""
#     def __init__(self, channels):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channels, channels // 16, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channels // 16, channels, bias=False),
#             nn.Sigmoid()
#         )
#         self.out_channels = channels
    
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


