"""
Debug script for model_parser.
Run this file directly or use VS Code debugger.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from yaml_model_parser import build_model_from_yaml, ModelParser
import torch

def main():
    # Choose which model to debug
    yaml_path = 'models/resnet18.yaml'
    # yaml_path = 'models/autoencoder_simple.yaml'
    # yaml_path = 'models/unet_autoencoder.yaml'

    print(f"Building model from {yaml_path}...")

    # Create parser
    parser = ModelParser(yaml_path)
    print(f"Architecture detected: {parser.arch_type}")

    # Parse model
    model, save_indices = parser.parse(input_channels=3)

    print(f"\nModel created successfully!")
    print(f"Save indices: {save_indices}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(1, 3, 224, 224)
    output = model(x)

    if isinstance(output, dict):
        print(f"Multi-output model:")
        for key, value in output.items():
            print(f"  {key}: {value.shape}")
    else:
        print(f"Single output: {output.shape}")

    print("\n✓ All tests passed!")

if __name__ == '__main__':
    main()
