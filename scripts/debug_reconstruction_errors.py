#!/usr/bin/env python3
"""
Debug Reconstruction Errors

This script loads the trained model and examines reconstruction errors
to understand why they're identical.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_and_data():
    """Load the trained model and test data."""
    # Load model
    model_path = Path("runs/run_20251003_131551_734775b0/models/final_model.pth")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None, None, None
    
    # Load model
    from src.models.cwtlstm import load_model, CWT_LSTM_Autoencoder
    
    model, metadata = load_model(
        model_path,
        CWT_LSTM_Autoencoder,
        latent_dim=32,
        lstm_hidden=64,
        dropout=0.1
    )
    
    model.eval()
    logger.info(f"Model loaded: {model.get_model_info()}")
    
    # Load test data
    processed_dir = Path("data/processed")
    cwt_files = list(processed_dir.glob("*.npy"))
    
    # Load manifest
    manifest_path = Path("data/download_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    gps_to_type = {}
    for download in manifest['downloads']:
        if download.get('successful', False):
            gps_time = download.get('start_gps')
            segment_type = download.get('segment_type')
            if gps_time and segment_type:
                gps_to_type[gps_time] = segment_type
    
    # Split files
    signal_files = []
    noise_files = []
    
    for file_path in cwt_files:
        try:
            filename_parts = file_path.stem.split('_')
            if len(filename_parts) >= 2:
                gps_time = int(filename_parts[1])
                segment_type = gps_to_type.get(gps_time, 'noise')
                
                if segment_type == 'signal':
                    signal_files.append(file_path)
                elif segment_type == 'noise':
                    noise_files.append(file_path)
        except (ValueError, IndexError):
            continue
    
    # Load sample data
    signal_data = []
    noise_data = []
    
    # Load first 5 signals and 5 noise samples
    for i, file_path in enumerate(signal_files[:5]):
        data = np.load(file_path)
        signal_data.append(data)
    
    for i, file_path in enumerate(noise_files[:5]):
        data = np.load(file_path)
        noise_data.append(data)
    
    signal_data = np.array(signal_data)
    noise_data = np.array(noise_data)
    
    logger.info(f"Loaded {len(signal_data)} signal samples, {len(noise_data)} noise samples")
    logger.info(f"Signal shape: {signal_data.shape}, Noise shape: {noise_data.shape}")
    
    return model, signal_data, noise_data

def compute_reconstruction_errors(model, data, title):
    """Compute reconstruction errors for data."""
    logger.info(f"\nComputing reconstruction errors for {title}")
    
    # Convert to tensor and add channel dimension
    data_tensor = torch.FloatTensor(data).unsqueeze(1)  # Add channel dimension
    
    reconstruction_errors = []
    reconstructions = []
    
    with torch.no_grad():
        for i in range(len(data_tensor)):
            sample = data_tensor[i:i+1]  # Single sample
            
            # Forward pass
            reconstructed, latent = model(sample)
            
            # Compute MSE reconstruction error
            mse = torch.mean((reconstructed - sample)**2, dim=(1, 2, 3))
            reconstruction_errors.append(mse.item())
            
            # Store reconstruction for analysis
            reconstructions.append(reconstructed.cpu().numpy())
    
    reconstruction_errors = np.array(reconstruction_errors)
    reconstructions = np.array(reconstructions)
    
    logger.info(f"{title} Reconstruction Errors:")
    logger.info(f"  Mean: {np.mean(reconstruction_errors):.6e}")
    logger.info(f"  Std:  {np.std(reconstruction_errors):.6e}")
    logger.info(f"  Min:  {np.min(reconstruction_errors):.6e}")
    logger.info(f"  Max:  {np.max(reconstruction_errors):.6e}")
    logger.info(f"  Range: {np.max(reconstruction_errors) - np.min(reconstruction_errors):.6e}")
    
    # Check if all errors are identical
    if np.std(reconstruction_errors) < 1e-10:
        logger.warning(f"⚠️  {title} reconstruction errors are essentially identical!")
        logger.warning(f"   This suggests the model is outputting constant values.")
    else:
        logger.info(f"✅ {title} reconstruction errors show variation")
    
    return reconstruction_errors, reconstructions

def analyze_model_outputs(model, data, title):
    """Analyze what the model is actually outputting."""
    logger.info(f"\nAnalyzing model outputs for {title}")
    
    data_tensor = torch.FloatTensor(data).unsqueeze(1)
    
    with torch.no_grad():
        reconstructed, latent = model(data_tensor)
        
        # Analyze reconstructions
        recon_np = reconstructed.cpu().numpy()
        
        logger.info(f"{title} Reconstruction Analysis:")
        logger.info(f"  Shape: {recon_np.shape}")
        logger.info(f"  Mean: {np.mean(recon_np):.6e}")
        logger.info(f"  Std:  {np.std(recon_np):.6e}")
        logger.info(f"  Min:  {np.min(recon_np):.6e}")
        logger.info(f"  Max:  {np.max(recon_np):.6e}")
        
        # Check if reconstructions are identical across samples
        sample_means = np.mean(recon_np, axis=(1, 2, 3))
        sample_stds = np.std(recon_np, axis=(1, 2, 3))
        
        logger.info(f"  Sample means: {sample_means}")
        logger.info(f"  Sample stds:  {sample_stds}")
        
        if np.std(sample_means) < 1e-10:
            logger.warning(f"⚠️  {title} reconstructions are identical across samples!")
        else:
            logger.info(f"✅ {title} reconstructions vary across samples")
        
        # Analyze latent representations
        latent_np = latent.cpu().numpy()
        logger.info(f"{title} Latent Analysis:")
        logger.info(f"  Shape: {latent_np.shape}")
        logger.info(f"  Mean: {np.mean(latent_np):.6e}")
        logger.info(f"  Std:  {np.std(latent_np):.6e}")
        
        latent_means = np.mean(latent_np, axis=1)
        if np.std(latent_means) < 1e-10:
            logger.warning(f"⚠️  {title} latent representations are identical!")
        else:
            logger.info(f"✅ {title} latent representations vary")

def main():
    """Main debugging function."""
    logger.info("Debugging Reconstruction Errors")
    logger.info("=" * 50)
    
    # Load model and data
    model, signal_data, noise_data = load_model_and_data()
    if model is None:
        return
    
    # Analyze model outputs
    analyze_model_outputs(model, signal_data, "Signals")
    analyze_model_outputs(model, noise_data, "Noise")
    
    # Compute reconstruction errors
    signal_errors, signal_recons = compute_reconstruction_errors(model, signal_data, "Signals")
    noise_errors, noise_recons = compute_reconstruction_errors(model, noise_data, "Noise")
    
    # Compare signal vs noise errors
    logger.info("\n" + "="*50)
    logger.info("SIGNAL vs NOISE COMPARISON")
    logger.info("="*50)
    
    logger.info(f"Signal errors: {signal_errors}")
    logger.info(f"Noise errors:  {noise_errors}")
    
    # Check if signal and noise errors are distinguishable
    all_errors = np.concatenate([signal_errors, noise_errors])
    error_std = np.std(all_errors)
    
    if error_std < 1e-10:
        logger.error("❌ CRITICAL ISSUE: Signal and noise reconstruction errors are identical!")
        logger.error("   The model cannot distinguish between signals and noise.")
        logger.error("   This explains the poor performance metrics.")
    else:
        logger.info("✅ Signal and noise errors are distinguishable")
    
    # Plot reconstruction errors
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(signal_errors, alpha=0.7, label='Signals', bins=10, color='red')
    plt.hist(noise_errors, alpha=0.7, label='Noise', bins=10, color='blue')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    plt.plot(signal_errors, 'ro-', label='Signals', markersize=8)
    plt.plot(noise_errors, 'bo-', label='Noise', markersize=8)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors by Sample')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    plt.scatter(range(len(signal_errors)), signal_errors, c='red', label='Signals', s=50)
    plt.scatter(range(len(noise_errors)), noise_errors, c='blue', label='Noise', s=50)
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error')
    plt.title('Reconstruction Errors Scatter Plot')
    plt.legend()
    plt.yscale('log')
    
    plt.subplot(2, 2, 4)
    all_errors_combined = np.concatenate([signal_errors, noise_errors])
    all_labels = np.concatenate([np.ones(len(signal_errors)), np.zeros(len(noise_errors))])
    plt.scatter(all_labels, all_errors_combined, alpha=0.7)
    plt.xlabel('Label (1=Signal, 0=Noise)')
    plt.ylabel('Reconstruction Error')
    plt.title('Errors vs Labels')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('reconstruction_error_debug.png', dpi=300, bbox_inches='tight')
    logger.info("Reconstruction error plots saved to 'reconstruction_error_debug.png'")
    
    # Final diagnosis
    logger.info("\n" + "="*50)
    logger.info("FINAL DIAGNOSIS")
    logger.info("="*50)
    
    if error_std < 1e-10:
        logger.error("❌ INFRASTRUCTURE ISSUE CONFIRMED:")
        logger.error("   1. Model produces identical reconstruction errors")
        logger.error("   2. Model cannot distinguish between signals and noise")
        logger.error("   3. This is NOT a data preprocessing issue")
        logger.error("   4. This is a model architecture or training issue")
        
        logger.info("\nPossible causes:")
        logger.info("   - Model bottleneck too wide (reconstruction too easy)")
        logger.info("   - Model learned identity function")
        logger.info("   - Learning rate too high (converged to trivial solution)")
        logger.info("   - Loss function inappropriate for data scale")
        logger.info("   - Model architecture mismatch with data complexity")
    else:
        logger.info("✅ Model appears to be working correctly")

if __name__ == "__main__":
    main()
