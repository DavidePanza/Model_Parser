"""
Minimalistic YAML model parser for building neural networks.
Imports layer definitions from layers.py module.
"""

import torch
import torch.nn as nn
import yaml
import warnings
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
META_FIELDS = {'model_name', 'nc', 'input_shape', 'anchors', 'architecture'}


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
        Auto-detect architecture type from YAML sections.

        Returns:
            Architecture type string
        """
        # Check for explicit declaration
        if 'architecture' in self.config:
            arch = self.config['architecture']
            if arch not in ARCHITECTURE_TYPES:
                warnings.warn(f"Unknown architecture type '{arch}', falling back to auto-detection")
            else:
                return arch

        # Auto-detect from sections
        sections = set(self.config.keys()) - META_FIELDS

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
        Validate that config matches architecture requirements.

        Raises:
            ValueError: If required sections are missing
        """
        arch_spec = ARCHITECTURE_TYPES[self.arch_type]
        required = set(arch_spec['required_sections'])
        available_sections = set(self.config.keys()) - META_FIELDS

        # Handle wildcard patterns
        validated_required = set()
        for req in required:
            if '*' in req:
                # Wildcard pattern - check if at least one matching section exists
                prefix = req.replace('*', '')
                matches = [s for s in available_sections if s.startswith(prefix)]
                if not matches:
                    raise ValueError(
                        f"No sections found matching pattern '{req}' for {self.arch_type} architecture"
                    )
            else:
                validated_required.add(req)

        # Check non-wildcard required sections
        missing = validated_required - available_sections
        if missing:
            raise ValueError(
                f"Missing required sections for {self.arch_type} architecture: {missing}\n"
                f"Available sections: {available_sections}"
            )

        # Warn about unknown sections
        all_valid = (set(arch_spec['required_sections']) |
                    set(arch_spec['optional_sections']))
        # Remove wildcards for comparison
        all_valid = {s.replace('*', '') for s in all_valid}
        unknown = [s for s in available_sections
                  if not any(s.startswith(v) for v in all_valid)]
        if unknown:
            warnings.warn(f"Unknown sections will be ignored: {unknown}")

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
            
        elif module_name == 'Upsample':
            size, scale_factor, mode = args
            module = LayerClass(size, scale_factor, mode)
            c_out = c_in  # Upsample doesn't change channels

        elif module_name == 'ConvTranspose2d':
            c_out, kernel_size, stride, *padding = args
            padding = padding[0] if padding else 0
            module = LayerClass(c_in, c_out, kernel_size, stride, padding)

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
        Get ordered list of sections to process.

        Returns:
            List of section names in processing order
        """
        order = self.arch_spec['section_order']
        expanded = []

        for section in order:
            if '*' in section:
                # Handle wildcard patterns (e.g., 'head_*')
                prefix = section.replace('*', '')
                matches = [k for k in self.config.keys()
                          if k.startswith(prefix) and k not in META_FIELDS]
                expanded.extend(sorted(matches))
            elif section in self.config:
                # Only add if section exists in config (do not add optional sections)
                expanded.append(section)

        return expanded

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