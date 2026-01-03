"""
Example usage of the minimalistic model parser.
Shows how to build and use models from YAML configs.
"""

import torch
from model_parser import build_model_from_yaml, ModelParser


def example_1_basic_usage():
    """Basic usage: Build model from YAML."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Build model
    model = build_model_from_yaml('yolov5n.yaml', input_channels=3)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


def example_2_custom_layers():
    """Add custom layers to the registry."""
    print("\n" + "=" * 80)
    print("Example 2: Adding Custom Layers")
    print("=" * 80)
    
    from layers import LAYER_REGISTRY
    import torch.nn as nn
    
    # Define a custom layer
    class MyCustomLayer(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            return self.relu(self.conv(x))
    
    # Register it
    LAYER_REGISTRY['MyCustomLayer'] = MyCustomLayer
    
    print("Custom layer 'MyCustomLayer' registered!")
    print(f"Available layers: {list(LAYER_REGISTRY.keys())}")


def example_3_inspect_model():
    """Inspect model architecture."""
    print("\n" + "=" * 80)
    print("Example 3: Model Inspection")
    print("=" * 80)
    
    parser = ModelParser('yolov5n.yaml')
    model, save_indices = parser.parse(input_channels=3)
    
    print(f"\nLayers with skip connections: {save_indices}")
    print(f"\nModel structure:")
    print(model)
    
    # Count parameters per layer
    print("\n" + "-" * 80)
    print("Parameters per layer:")
    print("-" * 80)
    for i, layer in enumerate(model.layers):
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"Layer {i:2d}: {num_params:>10,} parameters")


def example_4_save_load_model():
    """Save and load model."""
    print("\n" + "=" * 80)
    print("Example 4: Save and Load Model")
    print("=" * 80)
    
    # Build and save
    model = build_model_from_yaml('yolov5n.yaml')
    torch.save(model.state_dict(), 'model.pt')
    print("✓ Model saved to 'model.pt'")
    
    # Load
    model_new = build_model_from_yaml('yolov5n.yaml')
    model_new.load_state_dict(torch.load('model.pt'))
    print("✓ Model loaded from 'model.pt'")
    
    # Verify
    dummy_input = torch.randn(1, 3, 640, 640)
    out1 = model(dummy_input)
    out2 = model_new(dummy_input)
    
    if torch.allclose(out1, out2):
        print("✓ Models produce identical outputs")
    else:
        print("✗ Models produce different outputs")


def example_5_different_input_size():
    """Test with different input sizes."""
    print("\n" + "=" * 80)
    print("Example 5: Different Input Sizes")
    print("=" * 80)
    
    model = build_model_from_yaml('yolov5n.yaml')
    model.eval()
    
    sizes = [320, 480, 640, 800]
    for size in sizes:
        input_tensor = torch.randn(1, 3, size, size)
        output = model(input_tensor)
        print(f"Input: {size}×{size} → Output: {output.shape}")


def example_6_batch_processing():
    """Process multiple images in a batch."""
    print("\n" + "=" * 80)
    print("Example 6: Batch Processing")
    print("=" * 80)
    
    model = build_model_from_yaml('yolov5n.yaml')
    model.eval()
    
    batch_sizes = [1, 4, 8, 16]
    for bs in batch_sizes:
        batch = torch.randn(bs, 3, 640, 640)
        output = model(batch)
        print(f"Batch size: {bs:2d} → Output shape: {output.shape}")


if __name__ == '__main__':
    import sys
    
    # Run all examples or specific one
    if len(sys.argv) > 1:
        example_num = int(sys.argv[1])
        examples = [
            example_1_basic_usage,
            example_2_custom_layers,
            example_3_inspect_model,
            example_4_save_load_model,
            example_5_different_input_size,
            example_6_batch_processing
        ]
        examples[example_num - 1]()
    else:
        print("Running all examples...\n")
        example_1_basic_usage()
        example_2_custom_layers()
        example_3_inspect_model()
        # example_4_save_load_model()  # Commented to avoid file I/O
        example_5_different_input_size()
        example_6_batch_processing()
        
        print("\n" + "=" * 80)
        print("All examples completed!")
        print("=" * 80)