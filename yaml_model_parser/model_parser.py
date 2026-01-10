"""
Minimalistic YAML model parser for building neural networks.
Imports layer definitions from layers.py module.
"""

import torch
import torch.nn as nn
import yaml
from typing import List, Tuple, Any, Union, Dict, Set
from .layers import get_layer, LAYER_REGISTRY


# Architecture type definitions
ARCHITECTURE_TYPES = {
    'classification': {
        'required_sections': ['backbone', 'head'],
        'optional_sections': [],
        'section_order': ['backbone', 'head'],
        'description': 'Standard classification/detection (ResNet, YOLO)'
    },
    'autoencoder': {
        'required_sections': ['encoder', 'decoder'],
        'optional_sections': ['latent'],
        'section_order': ['encoder', 'latent', 'decoder'],
        'description': 'Encoder-decoder architecture (Autoencoder, U-Net)'
    },
    'gan': {
        'required_sections': ['generator', 'discriminator'],
        'optional_sections': [],
        'section_order': ['generator', 'discriminator'],
        'description': 'Generative Adversarial Network'
    },
    'multitask': {
        'required_sections': ['backbone'],
        'optional_sections': [],
        'section_order': ['backbone', 'head_*'],  # Wildcard pattern
        'description': 'Multi-task with shared backbone'
    },
    'transformer': {
        'required_sections': ['encoder', 'decoder'],
        'optional_sections': ['attention'],
        'section_order': ['encoder', 'attention', 'decoder'],
        'description': 'Transformer architecture (future)'
    }
}

# Meta fields that are not model sections
META_FIELDS = {'model_name', 'nc', 'input_shape', 'anchors', 'architecture', 'layers_order'}


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

        self.nc = self.config.get('nc', None) # number of classes
        self.anchors = self.config.get('anchors', None)

        # Detect and validate architecture type
        self.arch_type = self._detect_architecture()
        self._validate_architecture()
        
    def parse(self, input_channels: int = 3) -> Tuple[nn.Module, List[int]]:
        """
        Build model from configuration.

        Args:
            input_channels: Number of input channels (default: 3 for RGB)

        Returns:
            model: Model containing all layers
            save_layers: List of layer indices to save for skip connections
        """
        layers = []
        save_layers = []
        channel_list = [input_channels]  # Track output channels of each layer

        # Initialize section manager
        section_mgr = SectionManager(self.config, self.arch_type)
        all_layers_with_meta = section_mgr.get_all_layers()

        # Print architecture info
        print(f"\nArchitecture: {self.arch_type}")
        print(f"Sections: {' → '.join(section_mgr.get_section_order())}")
        print(f"Total layers: {len(all_layers_with_meta)}\n")

        print(f"{'Layer':<8}{'Section':<15}{'From':<20}{'Module':<20}{'Out Channels':<15}{'Params':<12}")
        print("-" * 100)

        for i, (layer_def, section_name, local_idx) in enumerate(all_layers_with_meta):
            from_idx, num_repeats, module_name, args = layer_def

            # Resolve cross-section references
            from_idx = self._resolve_references(from_idx, i, section_name, section_mgr)

            # Get input channels
            c_in = self._get_input_channels(from_idx, channel_list, i)

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
            module.section_name = section_name
            module.section_local_idx = local_idx

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
            print(f"{i:<8}{section_name:<15}{from_str:<20}{module_name:<20}{c_out:<15}{num_params:<12,}")

        print("-" * 100)
        total_params = sum(p.numel() for p in nn.ModuleList(layers).parameters())
        print(f"Total parameters: {total_params:,}\n")

        # Create model with forward logic for skip connections
        model = Model(layers, save_layers, section_mgr.section_boundaries)
        return model, sorted(set(save_layers))

    def _detect_architecture(self) -> str:
        """
        Detect architecture type from YAML.
        Uses explicit 'architecture' field or infers from layers_order.

        Returns:
            Architecture type string

        Raises:
            ValueError: If layers_order is missing
        """
        # Check for explicit architecture declaration
        if 'architecture' in self.config:
            arch = self.config['architecture']
            if arch not in ARCHITECTURE_TYPES:
                raise ValueError(
                    f"Unknown architecture type '{arch}'. "
                    f"Valid types: {list(ARCHITECTURE_TYPES.keys())}"
                )
            return arch

        # Require layers_order
        if 'layers_order' not in self.config:
            raise ValueError(
                "Missing required field 'layers_order' in YAML config.\n"
                "Add 'layers_order: [section1, section2, ...]' to specify the order of layer sections.\n"
                "Example: layers_order: [encoder, decoder]"
            )

        # Infer from layers_order
        layers_order = self.config['layers_order']
        sections = set(layers_order)

        # Check architecture patterns (order matters - most specific first)
        if 'generator' in sections and 'discriminator' in sections:
            return 'gan'
        if 'encoder' in sections and 'decoder' in sections:
            return 'autoencoder'
        if 'backbone' in sections and any(s.startswith('head_') for s in sections):
            return 'multitask'
        if 'backbone' in sections and 'head' in sections:
            return 'classification'

        # Default fallback
        return 'classification'

    def _validate_architecture(self):
        """
        Validate that config has valid layer sections.

        Raises:
            ValueError: If layers_order is missing or references undefined sections
        """
        # Require layers_order
        if 'layers_order' not in self.config:
            raise ValueError(
                "Missing required field 'layers_order' in YAML config.\n"
                "Add 'layers_order: [section1, section2, ...]' to specify the order of layer sections.\n"
                "Example: layers_order: [encoder, decoder]"
            )

        # Get available sections (everything that's a list)
        available_sections = {k for k, v in self.config.items()
                            if isinstance(v, list) and k not in META_FIELDS}

        layers_order = self.config['layers_order']

        if not isinstance(layers_order, list):
            raise ValueError("'layers_order' must be a list of section names")

        # Check all sections in layers_order exist
        for section_name in layers_order:
            if section_name not in self.config:
                raise ValueError(
                    f"Section '{section_name}' in layers_order not found in config.\n"
                    f"Available sections: {sorted(available_sections)}"
                )
            if not isinstance(self.config[section_name], list):
                raise ValueError(
                    f"Section '{section_name}' must be a list of layers, "
                    f"got {type(self.config[section_name])}"
                )

    def _resolve_references(
        self,
        from_idx: Union[int, List],
        current_idx: int,
        current_section: str,
        section_mgr: 'SectionManager'
    ) -> Union[int, List[int]]:
        """
        Resolve layer references including cross-section references.

        Supports:
        - Negative indices: -1 (previous layer)
        - Positive indices: 3 (absolute layer 3)
        - String refs: 'encoder:2' (layer 2 from encoder section)

        Args:
            from_idx: Layer reference (int, list, or string)
            current_idx: Current global layer index
            current_section: Current section name
            section_mgr: SectionManager instance

        Returns:
            Resolved global layer index/indices
        """
        if isinstance(from_idx, int):
            # Standard integer reference - no change needed
            return from_idx

        elif isinstance(from_idx, list):
            # List of references - resolve each
            resolved = []
            for ref in from_idx:
                if isinstance(ref, str) and ':' in ref:
                    # Parse 'section:index' format
                    section_name, idx_str = ref.split(':', 1)
                    local_idx = int(idx_str)

                    # Get section boundaries
                    if section_name not in section_mgr.section_boundaries:
                        raise ValueError(
                            f"Unknown section '{section_name}' in reference '{ref}'. "
                            f"Available sections: {list(section_mgr.section_boundaries.keys())}"
                        )

                    start, end = section_mgr.section_boundaries[section_name]
                    global_idx = start + local_idx

                    if global_idx > end:
                        raise ValueError(
                            f"Index {local_idx} out of range for section '{section_name}' "
                            f"(section has {end - start + 1} layers, indices 0-{end - start})"
                        )

                    resolved.append(global_idx)

                elif isinstance(ref, int):
                    # Standard integer reference
                    resolved.append(ref)
                else:
                    raise ValueError(f"Unsupported reference format: {ref}")

            return resolved

        else:
            raise ValueError(f"Unsupported from_idx type: {type(from_idx)}")

    def _get_input_channels(
        self,
        from_idx: Union[int, List[int]],
        channel_list: List[int],
        current_idx: int
    ) -> Union[int, List[int]]:
        """
        Get input channel count(s) for a layer.

        Args:
            from_idx: Layer reference (int or list for Concat)
            channel_list: List of output channels for all previous layers
            current_idx: Current global layer index

        Returns:
            Input channel count (int) or list of counts (for Concat)
        """
        if isinstance(from_idx, int):
            # Convert negative indices relative to current position
            if from_idx < 0:
                # -1 at layer i means channel_list[i] (previous layer's output)
                actual_idx = (current_idx + 1) + from_idx  # +1 because channel_list[0] is input
            else:
                # Positive index is absolute layer number
                actual_idx = from_idx + 1  # +1 because channel_list[0] is input
            c_in = channel_list[actual_idx]
        else:  # Multiple inputs (for Concat)
            c_in = []
            for idx in from_idx:
                # Convert negative indices relative to current position
                if idx < 0:
                    # -1 at layer i means channel_list[i] (previous layer's output)
                    actual_idx = (current_idx + 1) + idx  # +1 because channel_list[0] is input
                else:
                    # Positive index is absolute layer number
                    actual_idx = idx + 1  # +1 because channel_list[0] is input
                c_in.append(channel_list[actual_idx])

        return c_in

    def _evaluate_config_expression(self, val):
        """
        Evaluate a value that may contain config variable references or expressions.

        Args:
            val: Value to evaluate (can be string with variable/expression, or any other type)

        Returns:
            Evaluated value

        Examples:
            "base_channel_size" -> 32
            "base_channel_size*2" -> 64
            "latent_dim" -> 128
            "None" -> None
            "nearest" -> "nearest" (unchanged)
        """
        if not isinstance(val, str):
            return val

        # Define metadata fields to exclude
        META_FIELDS = {'model_name', 'input_shape', 'architecture', 'layers_order'}
        config_mapping = {
            k: v for k, v in self.config.items()
            if k not in META_FIELDS and not isinstance(v, (list, dict))
        }

        # Check for None
        if val in ["None", 'None']:
            return None

        # Check for direct variable reference
        if val in config_mapping:
            return config_mapping[val]

        # Check for arithmetic expressions (e.g., "base_channel_size*2")
        # Replace variables with their values in the expression
        expr = val
        has_variables = False
        for var_name, var_value in config_mapping.items():
            if var_name in expr:
                expr = expr.replace(var_name, str(var_value))
                has_variables = True

        # Only try to evaluate if we found variables
        if has_variables:
            try:
                # Only allow safe arithmetic operations
                import ast
                import operator

                # Safe operators
                safe_ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                }

                def eval_expr(node):
                    if isinstance(node, ast.Constant):  # number (Python 3.8+)
                        return node.value
                    elif isinstance(node, ast.Num):  # number (deprecated, for older Python)
                        return node.n
                    elif isinstance(node, ast.BinOp):  # binary operation
                        op = safe_ops.get(type(node.op))
                        if op is None:
                            raise ValueError(f"Unsafe operation: {type(node.op)}")
                        return op(eval_expr(node.left), eval_expr(node.right))
                    elif isinstance(node, ast.UnaryOp):  # unary operation (e.g., -5)
                        if isinstance(node.op, ast.USub):
                            return -eval_expr(node.operand)
                        elif isinstance(node.op, ast.UAdd):
                            return eval_expr(node.operand)
                    else:
                        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

                tree = ast.parse(expr, mode='eval')
                result = eval_expr(tree.body)

                # Return as int if it's a whole number
                if isinstance(result, float) and result.is_integer():
                    return int(result)
                return result

            except (SyntaxError, ValueError, KeyError):
                # If evaluation fails, return original value
                pass

        return val

    def _parse_args(self, args, param_names: List[str], defaults: dict = None):
        """
        Parse arguments that can be either positional (list) or named (dict).

        Args:
            args: Arguments from YAML (list or dict)
            param_names: Expected parameter names in order
            defaults: Default values for optional parameters

        Returns:
            dict: Parsed arguments as {param_name: value}

        Examples:
            # Positional: [64, 3, 2, 1]
            # Named: {c_out: 64, kernel: 3, stride: 2, padding: 1}
            # Mixed: [64, 3, {stride: 2, padding: 1}]
        """
        defaults = defaults or {}
        parsed = {}

        if isinstance(args, dict):
            # Named arguments
            parsed = {k: self._evaluate_config_expression(v) for k, v in args.items()}
        elif isinstance(args, list):
            # Positional or mixed
            pos_idx = 0
            for arg in args:
                if isinstance(arg, dict):
                    # Rest are named arguments
                    parsed.update({k: self._evaluate_config_expression(v) for k, v in arg.items()})
                else:
                    # Positional argument
                    if pos_idx < len(param_names):
                        parsed[param_names[pos_idx]] = self._evaluate_config_expression(arg)
                        pos_idx += 1

        # Apply defaults for missing parameters
        for param, default_val in defaults.items():
            if param not in parsed:
                parsed[param] = default_val

        return parsed

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

        # Evaluate string references to config variables and expressions
        # Skip this for dict args (they're handled in _parse_args)
        if not isinstance(args, dict):
            args = [self._evaluate_config_expression(arg) for arg in args]

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
            
        elif module_name == 'Upsample':
            size, scale_factor, mode = args
            module = LayerClass(size, scale_factor, mode)
            c_out = c_in  # Upsample doesn't change channels

        elif module_name == 'ConvTranspose2d':
            # Support both positional and named arguments
            parsed = self._parse_args(
                args,
                param_names=['c_out', 'kernel_size', 'stride', 'padding', 'output_padding'],
                defaults={'padding': 0, 'output_padding': 0}
            )
            c_out = parsed['c_out']
            module = LayerClass(
                c_in,
                parsed['c_out'],
                parsed['kernel_size'],
                parsed['stride'],
                parsed['padding'],
                parsed['output_padding']
            )

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
        
        elif module_name in ['ReLU', 'GELU', 'Sigmoid', 'Tanh']:
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
            # For flatten, optionally specify the flattened output size
            # Format: [] (auto, keeps c_in) or [output_features]
            if args and len(args) > 0:
                c_out = args[0]  # Explicit flattened feature count
            else:
                c_out = c_in  # Fallback (may need manual correction)

        elif module_name == 'Reshape':
            # Args: shape dimensions (e.g., [-1, 4, 4] or [128, 4, 4])
            shape = args  # All args are shape dimensions
            module = LayerClass(*shape)
            # Calculate output channels from first dimension
            if len(shape) > 0:
                c_out = shape[0] if shape[0] != -1 else c_in
            else:
                c_out = c_in

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


class SectionManager:
    """Manages section processing for different architecture types."""

    def __init__(self, config: Dict, arch_type: str):
        """
        Initialize section manager.

        Args:
            config: YAML configuration dictionary
            arch_type: Architecture type (classification, autoencoder, etc.)
        """
        self.config = config
        self.arch_type = arch_type
        self.arch_spec = ARCHITECTURE_TYPES[arch_type]
        self.section_boundaries = {}  # Track section start/end indices: {section: (start, end)}
        self.section_outputs = {}     # Track section final output indices: {section: end_idx}

    def get_section_order(self) -> List[str]:
        """
        Get ordered list of sections to process from layers_order.

        Returns:
            List of section names in processing order

        Raises:
            ValueError: If layers_order is missing (should be caught in validation)
        """
        # layers_order is required (validated in _validate_architecture)
        if 'layers_order' not in self.config:
            raise ValueError("layers_order is required but missing from config")

        return self.config['layers_order']

    def get_all_layers(self) -> List[Tuple[List, str, int]]:
        """
        Get all layers in order with section metadata.

        Returns:
            List of (layer_def, section_name, section_local_idx) tuples
        """
        all_layers = []
        global_idx = 0

        for section_name in self.get_section_order():
            section_layers = self.config[section_name]
            if not section_layers:
                continue

            section_start = global_idx

            for local_idx, layer_def in enumerate(section_layers):
                all_layers.append((layer_def, section_name, local_idx))
                global_idx += 1

            # Track section boundaries
            section_end = global_idx - 1
            self.section_boundaries[section_name] = (section_start, section_end)
            self.section_outputs[section_name] = section_end

        return all_layers


class Model(nn.Module):
    """
    Model wrapper that handles forward pass with skip connections.
    Supports multi-output architectures (GANs, multi-task models).
    """

    def __init__(self, layers: List[nn.Module], save_indices: List[int], section_boundaries: Dict[str, Tuple[int, int]] = None):
        """
        Args:
            layers: List of all model layers
            save_indices: Indices of layers to save for skip connections
            section_boundaries: Dict mapping section names to (start_idx, end_idx) tuples
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.save_indices = set(save_indices)
        self.section_boundaries = section_boundaries or {}
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass with support for multi-output architectures.

        Returns:
            - Single tensor for single-output architectures (classification, autoencoder)
            - Dict of {section_name: tensor} for multi-output architectures (GAN, multi-task)
        """
        layer_outputs = {}
        section_outputs = {}  # Track final output of each section

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

            # Track section outputs (last layer of each section)
            if hasattr(layer, 'section_name'):
                for section, (start, end) in self.section_boundaries.items():
                    if i == end:
                        section_outputs[section] = x

        # Return dict for multi-output architectures, single tensor otherwise
        if self._is_multi_output():
            return section_outputs
        return x

    def _is_multi_output(self) -> bool:
        """
        Determine if this is a multi-output architecture.
        Multi-output architectures have multiple independent sections (GAN, multi-task).
        """
        if not self.section_boundaries:
            return False

        # Check for multiple heads (multi-task) or separate networks (GAN)
        sections = list(self.section_boundaries.keys())

        # GAN: generator + discriminator
        if 'generator' in sections and 'discriminator' in sections:
            return True

        # Multi-task: multiple heads
        head_sections = [s for s in sections if s.startswith('head_')]
        if len(head_sections) > 1:
            return True

        # Single output for classification and autoencoder
        return False


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
        yaml_path = 'autoencoder_basic.yaml'
    
    print(f"Building model from {yaml_path}...")
    model = build_model_from_yaml(yaml_path)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("\n✓ Model built successfully!")