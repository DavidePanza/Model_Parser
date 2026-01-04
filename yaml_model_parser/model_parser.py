"""
Minimalistic YAML model parser for building neural networks.
Imports layer definitions from layers.py module.
"""

import torch
import torch.nn as nn
import yaml
from typing import List, Tuple, Any, Union
from layers import get_layer, LAYER_REGISTRY


class ModelParser:
    """Build PyTorch models from YAML configuration."""
    
    def __init__(self, yaml_path: str):
        """
        Load YAML configuration.
        
        Args:
            yaml_path: Path to YAML config file
        """
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.nc = self.config['nc']
        self.anchors = self.config.get('anchors', None)
        
    def parse(self, input_channels: int = 3) -> Tuple[nn.Module, List[int]]:
        """
        Build model from configuration.
        
        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            
        Returns:
            model: YOLOModel containing all layers
            save_layers: List of layer indices to save for skip connections
        """
        layers = []
        save_layers = []
        channel_list = [input_channels]  # Track output channels of each layer
        
        # Combine backbone and head
        all_layers = self.config['backbone'] + self.config['head']
        print(len(all_layers))
        
        print(f"\n{'Layer':<8}{'From':<12}{'Module':<20}{'Out Channels':<15}{'Params':<12}")
        print("-" * 80)
        
        for i, layer_def in enumerate(all_layers):
            from_idx, num_repeats, module_name, args = layer_def
            
            # Get input channels  
            if isinstance(from_idx, int):
                # For single input, just use channel_list with the index
                # Python's negative indexing handles -1 correctly as "previous"
                c_in = channel_list[from_idx]
            else:  # Multiple inputs (for Concat)
                c_in = []
                for idx in from_idx:
                    # For concat, convert negative indices relative to current position
                    if idx < 0:
                        # -1 at layer i means channel_list[i] (previous layer's output)
                        actual_idx = (i + 1) + idx  # i+1 because channel_list[0] is input
                    else:
                        # Positive index is absolute layer number
                        actual_idx = idx + 1  # +1 because channel_list[0] is input
                    c_in.append(channel_list[actual_idx])
            
            # Build the layer
            module, c_out = self._build_layer(
                module_name=module_name,
                args=args,
                c_in=c_in,
                num_repeats=num_repeats
            )
            
            # Store metadata
            module.layer_idx = i
            module.from_idx = from_idx
            
            # Count parameters
            num_params = sum(p.numel() for p in module.parameters())
            
            # Add to model
            layers.append(module)
            channel_list.append(c_out)
            
            # Track layers needed for skip connections
            if isinstance(from_idx, list):
                save_layers.extend([idx for idx in from_idx if idx != -1])
            elif from_idx != -1:
                save_layers.append(from_idx)
            
            # Print layer info
            from_str = str(from_idx) if isinstance(from_idx, int) else str(from_idx)
            print(f"{i:<8}{from_str:<12}{module_name:<20}{c_out:<15}{num_params:<12,}")
        
        print("-" * 80)
        total_params = sum(p.numel() for p in nn.ModuleList(layers).parameters())
        print(f"Total parameters: {total_params:,}\n")
        
        # Create model with forward logic for skip connections
        model = Model(layers, save_layers)
        return model, sorted(set(save_layers))
    
    def _build_layer(
        self, 
        module_name: str, 
        args: List, 
        c_in: Union[int, List[int]], 
        num_repeats: int
    ) -> Tuple[nn.Module, int]:
        """
        Build a single layer or module.
        
        Args:
            module_name: Name of the layer (e.g., 'Conv', 'C3')
            args: Arguments for the layer
            c_in: Input channels (int or list for Concat)
            num_repeats: Number of times to repeat this layer
        
        Returns:
            module: PyTorch module
            c_out: Output channels
        """

        # Evaluate string references to config variables
        config_mapping = {
            'nc': self.nc,
            'anchors': self.anchors
        }
        
        evaluated_args = []
        for arg in args:
            if isinstance(arg, str):
                # Check if it's a config variable
                if arg in config_mapping:
                    evaluated_args.append(config_mapping[arg])
                # Convert string "None" to actual None
                elif arg == "None" or arg == 'None':
                    evaluated_args.append(None)
                else:
                    # It's a regular string (like "nearest"), keep it as-is
                    evaluated_args.append(arg)
            else:
                evaluated_args.append(arg)

        args = evaluated_args

        # Get layer class
        LayerClass = get_layer(module_name)
        
        # Build layer based on type
        if module_name == 'Conv':
            c_out, kernel, stride, *padding = args
            padding = padding[0] if padding else None
            module = LayerClass(c_in, c_out, kernel, stride, padding)
            
        elif module_name == 'C3':
            c_out = args[0]
            shortcut = args[1] if len(args) > 1 else True
            module = LayerClass(c_in, c_out, num_repeats, shortcut)
            return module, c_out  # ← Return immediately, skip repetition code
            
        elif module_name == 'SPPF':
            c_out, kernel_size = args
            module = LayerClass(c_in, c_out, kernel_size)
            
        elif module_name in ['Upsample', 'nn.Upsample']:
            size, scale_factor, mode = args
            module = LayerClass(size, scale_factor, mode)
            c_out = c_in  # Upsample doesn't change channels
            
        elif module_name == 'Concat':
            dimension = args[0]
            module = LayerClass(dimension)
            c_out = sum(c_in) if isinstance(c_in, list) else c_in # This assumes concat across channels. Modify if you want a different implementation

        elif module_name == 'Detect':
            nc = args[0]
            anchors = args[1] if len(args) > 1 else self.anchors
            module = LayerClass(nc, anchors, c_in)
            na = len(anchors[0]) // 2
            c_out = na * (nc + 5)
        
        # ========== Standard PyTorch Layers ==========
        
        elif module_name == 'Linear':
            c_out = args[0]
            module = LayerClass(c_in, c_out)
        
        elif module_name == 'BatchNorm':
            c_out = c_in
            module = LayerClass(c_in)
        
        elif module_name in ['ReLU', 'Sigmoid']:
            c_out = c_in
            module = LayerClass()
        
        elif module_name in ['MaxPool', 'AvgPool']:
            kernel_size = args[0]
            stride = args[1] if len(args) > 1 else None
            padding = args[2] if len(args) > 2 else 0
            module = LayerClass(kernel_size, stride, padding)
            c_out = c_in
        
        elif module_name == 'AdaptiveAvgPool':
            output_size = args[0]
            module = LayerClass(output_size)
            c_out = c_in
        
        elif module_name == 'Dropout':
            p = args[0] if args else 0.5
            module = LayerClass(p)
            c_out = c_in
        
        elif module_name == 'Flatten':
            module = LayerClass()
            # For flatten, calculate flattened size
            # This is a placeholder - actual size depends on spatial dims
            c_out = c_in  # Will be updated if needed
        
        elif module_name == 'ResBlock':
            c_out = args[0]
            stride = args[1] if len(args) > 1 else 1
            module = LayerClass(c_in, c_out, stride)
        
        elif module_name == 'SEBlock':
            reduction = args[0] if args else 16
            module = LayerClass(c_in, reduction)
            c_out = c_in
            
        else:
            raise NotImplementedError(f"Module '{module_name}' not implemented")
        
        # Repeat module if needed
        if num_repeats > 1:
            module = nn.Sequential(*[module for _ in range(num_repeats)])
        
        return module, c_out


class Model(nn.Module):
    """
    Model wrapper that handles forward pass with skip connections.
    """
    
    def __init__(self, layers: List[nn.Module], save_indices: List[int]):
        """
        Args:
            layers: List of all model layers
            save_indices: Indices of layers to save for skip connections
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.save_indices = set(save_indices)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_outputs = {}  
        
        for i, layer in enumerate(self.layers):
            from_idx = layer.from_idx
            
            # Helper function to resolve index
            def get_output(idx):
                if idx == -1:
                    return x  # Use current x for -1
                else:
                    return layer_outputs[idx]  
            
            # Get input(s)
            if isinstance(from_idx, int):
                input_tensor = get_output(from_idx)
                x = layer(input_tensor)
            else:
                # Multiple inputs
                input_tensors = [get_output(idx) for idx in from_idx]
                x = layer(input_tensors)
            
            # Only save if needed for future layers
            if i in self.save_indices:
                layer_outputs[i] = x
        
        return x


def build_model_from_yaml(yaml_path: str, input_channels: int = 3) -> nn.Module:
    """
    Convenience function to build a model from YAML.
    
    Args:
        yaml_path: Path to YAML config file
        input_channels: Number of input channels (default: 3)
    
    Returns:
        PyTorch model
    
    Example:
        >>> model = build_model_from_yaml('yolov5n.yaml')
        >>> output = model(torch.randn(1, 3, 640, 640))
    """
    parser = ModelParser(yaml_path)
    model, save_indices = parser.parse(input_channels)
    return model


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        yaml_path = sys.argv[1]
    else:
        yaml_path = 'model1.yaml'
    
    print(f"Building model from {yaml_path}...")
    model = build_model_from_yaml(yaml_path)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model built successfully!")