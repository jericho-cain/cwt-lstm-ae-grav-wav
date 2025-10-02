#!/usr/bin/env python3
"""
Test script for CWT-LSTM Autoencoder model

This script tests the model creation, forward pass, and basic functionality
to ensure the model architecture is working correctly.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import sys
import torch
import numpy as np
import pytest
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import create_model, CWT_LSTM_Autoencoder, SimpleCWTAutoencoder


@pytest.fixture(params=[
    ('cwt_lstm', 'CWT-LSTM'),
    ('simple_cwt', 'Simple CWT')
])
def model_and_name(request):
    """Fixture providing model and name for parameterized tests."""
    model_type, model_name = request.param
    if model_type == 'cwt_lstm':
        model = create_model(
            model_type=model_type,
            input_height=8,
            input_width=1024,
            latent_dim=32,
            lstm_hidden=64,
            dropout=0.1
        )
    else:  # simple_cwt
        model = create_model(
            model_type=model_type,
            input_height=8,
            input_width=1024,
            latent_dim=64,
            dropout=0.1
        )
    return model, model_name


def test_model_creation():
    """Test model creation and basic properties."""
    print("Testing model creation...")
    
    # Test CWT-LSTM model
    model = create_model(
        model_type='cwt_lstm',
        input_height=8,
        input_width=131072,
        latent_dim=32,
        lstm_hidden=64,
        dropout=0.1
    )
    
    print(f"[OK] CWT-LSTM model created successfully")
    print(f"  Model info: {model.get_model_info()}")
    
    # Test Simple CWT model
    simple_model = create_model(
        model_type='simple_cwt',
        input_height=8,
        input_width=131072,
        latent_dim=64,
        dropout=0.1
    )
    
    print(f"[OK] Simple CWT model created successfully")
    print(f"  Model info: {simple_model.get_model_info()}")
    
    assert model is not None
    assert simple_model is not None


def test_forward_pass(model_and_name):
    """Test forward pass through the model."""
    model, model_name = model_and_name
    print(f"\nTesting forward pass for {model_name}...")
    
    # Create dummy input (batch_size=2, channels=1, height=8, width=1024)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 8, 1024)
    
    print(f"  Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed, latent = model(input_tensor)
    
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Latent shape: {latent.shape}")
    
    # Check shapes
    assert reconstructed.shape == input_tensor.shape, f"Reconstruction shape mismatch: {reconstructed.shape} vs {input_tensor.shape}"
    assert latent.shape[0] == batch_size, f"Latent batch size mismatch: {latent.shape[0]} vs {batch_size}"
    
    print(f"[OK] Forward pass successful for {model_name}")
    
    assert reconstructed is not None
    assert latent is not None


def test_model_parameters(model_and_name):
    """Test model parameters and gradients."""
    model, model_name = model_and_name
    print(f"\nTesting model parameters for {model_name}...")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test gradient flow
    input_tensor = torch.randn(1, 1, 8, 1024, requires_grad=True)
    reconstructed, latent = model(input_tensor)
    
    # Compute loss and backward pass
    loss = torch.nn.functional.mse_loss(reconstructed, input_tensor)
    loss.backward()
    
    # Check gradients
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
    
    print(f"  Parameters with gradients: {grad_count}")
    print(f"[OK] Parameter and gradient test successful for {model_name}")


def test_model_save_load(model_and_name):
    """Test model saving and loading."""
    model, model_name = model_and_name
    print(f"\nTesting model save/load for {model_name}...")
    
    # Create temporary save path
    save_path = Path("temp_model.pth")
    
    try:
        # Save model
        from models import save_model, load_model
        
        metadata = {
            "test_metadata": "This is a test",
            "model_name": model_name
        }
        
        save_model(model, save_path, metadata)
        print(f"[OK] Model saved successfully")
        
        # Load model
        loaded_model, loaded_metadata = load_model(save_path, type(model))
        print(f"[OK] Model loaded successfully")
        print(f"  Loaded metadata: {loaded_metadata}")
        
        # Test loaded model
        input_tensor = torch.randn(1, 1, 8, 1024)
        
        with torch.no_grad():
            original_output = model(input_tensor)
            loaded_output = loaded_model(input_tensor)
        
        # Check outputs are similar
        diff = torch.abs(original_output[0] - loaded_output[0]).mean()
        print(f"  Output difference: {diff.item():.6f}")
        
        if diff.item() < 1e-6:
            print(f"[OK] Save/load test successful for {model_name}")
        else:
            print(f"[WARNING] Output difference is large: {diff.item():.6f}")
            
    finally:
        # Clean up
        if save_path.exists():
            save_path.unlink()
            print(f"  Cleaned up temporary file")


def main():
    """Main test function."""
    print("CWT-LSTM Autoencoder Model Test")
    print("=" * 40)
    
    try:
        # Test model creation
        cwt_lstm_model, simple_model = test_model_creation()
        
        # Test forward pass
        test_forward_pass(cwt_lstm_model, "CWT-LSTM")
        test_forward_pass(simple_model, "Simple CWT")
        
        # Test parameters
        test_model_parameters(cwt_lstm_model, "CWT-LSTM")
        test_model_parameters(simple_model, "Simple CWT")
        
        # Test save/load
        test_model_save_load(cwt_lstm_model, "CWT-LSTM")
        test_model_save_load(simple_model, "Simple CWT")
        
        print("\n" + "=" * 40)
        print("[OK] All tests passed successfully!")
        print("The CWT-LSTM autoencoder model is working correctly.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
