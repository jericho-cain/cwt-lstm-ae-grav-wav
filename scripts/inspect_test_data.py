#!/usr/bin/env python3
"""
Inspect test data to verify gravitational wave signals are present and distinguishable from noise.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import yaml
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data_sample():
    """Load a small sample of test data for inspection."""
    processed_dir = Path("data/processed")
    manifest_path = Path("data/download_manifest.json")
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Find one noise and one signal file
    noise_file = None
    signal_file = None
    
    for segment_data in manifest['downloads']:
        label = segment_data.get('label', '')
        
        if label.startswith('test_noise_') and noise_file is None:
            start_gps = segment_data.get('start_gps')
            detector = segment_data.get('detector', 'H1')
            if start_gps:
                filename = f"{detector}_{start_gps}_32s_cwt.npy"
                file_path = processed_dir / filename
                if file_path.exists():
                    noise_file = file_path
                    logger.info(f"Found noise file: {filename}")
        
        elif label.startswith('GW') and signal_file is None:
            start_gps = segment_data.get('start_gps')
            detector = segment_data.get('detector', 'H1')
            if start_gps:
                filename = f"{detector}_{start_gps}_32s_cwt.npy"
                file_path = processed_dir / filename
                if file_path.exists():
                    signal_file = file_path
                    logger.info(f"Found signal file: {filename} (label: {label})")
        
        if noise_file and signal_file:
            break
    
    if not noise_file or not signal_file:
        raise ValueError("Could not find both noise and signal files")
    
    # Load the data
    noise_data = np.load(noise_file)
    signal_data = np.load(signal_file)
    
    logger.info(f"Noise data shape: {noise_data.shape}")
    logger.info(f"Signal data shape: {signal_data.shape}")
    
    return noise_data, signal_data, noise_file.name, signal_file.name

def inspect_raw_data():
    """Inspect raw strain data to see if signals are present."""
    raw_dir = Path("data/raw")
    manifest_path = Path("data/download_manifest.json")
    
    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Find one noise and one signal file
    noise_file = None
    signal_file = None
    
    for segment_data in manifest['downloads']:
        label = segment_data.get('label', '')
        
        if label.startswith('test_noise_') and noise_file is None:
            file_path = segment_data.get('file_path')
            if file_path and Path(file_path).exists():
                noise_file = Path(file_path)
                logger.info(f"Found raw noise file: {file_path}")
        
        elif label.startswith('GW') and signal_file is None:
            file_path = segment_data.get('file_path')
            if file_path and Path(file_path).exists():
                signal_file = Path(file_path)
                logger.info(f"Found raw signal file: {file_path} (label: {label})")
        
        if noise_file and signal_file:
            break
    
    if not noise_file or not signal_file:
        raise ValueError("Could not find both raw noise and signal files")
    
    # Load the raw data
    noise_data = np.load(noise_file)
    signal_data = np.load(signal_file)
    
    logger.info(f"Raw noise data keys: {list(noise_data.keys())}")
    logger.info(f"Raw signal data keys: {list(signal_data.keys())}")
    
    # Extract strain data (assuming it's in 'strain' key)
    if 'strain' in noise_data:
        noise_strain = noise_data['strain']
        signal_strain = signal_data['strain']
    else:
        # Try to find the strain data
        noise_keys = list(noise_data.keys())
        signal_keys = list(signal_data.keys())
        logger.info(f"Noise keys: {noise_keys}")
        logger.info(f"Signal keys: {signal_keys}")
        
        # Use the first array we find
        noise_strain = noise_data[noise_keys[0]]
        signal_strain = signal_data[signal_keys[0]]
    
    logger.info(f"Noise strain shape: {noise_strain.shape}")
    logger.info(f"Signal strain shape: {signal_strain.shape}")
    
    return noise_strain, signal_strain, noise_file.name, signal_file.name

def create_comparison_plots():
    """Create comparison plots of noise vs signal data."""
    # Create output directory
    output_dir = Path("results/data_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load CWT data
        noise_cwt, signal_cwt, noise_name, signal_name = load_test_data_sample()
        
        # Create CWT comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot CWT scalograms (remove extra dimension)
        noise_cwt_2d = noise_cwt.squeeze(0)  # Remove batch dimension
        signal_cwt_2d = signal_cwt.squeeze(0)  # Remove batch dimension
        
        im1 = axes[0, 0].imshow(np.abs(noise_cwt_2d), aspect='auto', cmap='viridis')
        axes[0, 0].set_title(f'Noise CWT: {noise_name}')
        axes[0, 0].set_ylabel('Frequency Scale')
        axes[0, 0].set_xlabel('Time')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(np.abs(signal_cwt_2d), aspect='auto', cmap='viridis')
        axes[0, 1].set_title(f'Signal CWT: {signal_name}')
        axes[0, 1].set_ylabel('Frequency Scale')
        axes[0, 1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot power spectra
        noise_power = np.mean(np.abs(noise_cwt_2d)**2, axis=1)
        signal_power = np.mean(np.abs(signal_cwt_2d)**2, axis=1)
        
        axes[1, 0].plot(noise_power, label='Noise')
        axes[1, 0].plot(signal_power, label='Signal')
        axes[1, 0].set_title('Average Power vs Frequency Scale')
        axes[1, 0].set_xlabel('Frequency Scale Index')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].legend()
        axes[1, 0].set_yscale('log')
        
        # Plot time series (if available)
        try:
            noise_raw, signal_raw, noise_raw_name, signal_raw_name = inspect_raw_data()
            
            # Plot a portion of the time series
            time_points = min(10000, len(noise_raw), len(signal_raw))
            time_axis = np.arange(time_points) / 4096  # Assuming 4096 Hz sample rate
            
            axes[1, 1].plot(time_axis, noise_raw[:time_points], alpha=0.7, label='Noise')
            axes[1, 1].plot(time_axis, signal_raw[:time_points], alpha=0.7, label='Signal')
            axes[1, 1].set_title('Raw Strain Data (first 10k samples)')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Strain')
            axes[1, 1].legend()
            
        except Exception as e:
            logger.warning(f"Could not load raw data: {e}")
            axes[1, 1].text(0.5, 0.5, 'Raw data not available', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(output_dir / "noise_vs_signal_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plots saved to {output_dir}")
        
        # Calculate some statistics
        noise_stats = {
            'mean': np.mean(np.abs(noise_cwt_2d)),
            'std': np.std(np.abs(noise_cwt_2d)),
            'max': np.max(np.abs(noise_cwt_2d)),
            'min': np.min(np.abs(noise_cwt_2d))
        }
        
        signal_stats = {
            'mean': np.mean(np.abs(signal_cwt_2d)),
            'std': np.std(np.abs(signal_cwt_2d)),
            'max': np.max(np.abs(signal_cwt_2d)),
            'min': np.min(np.abs(signal_cwt_2d))
        }
        
        logger.info("CWT Statistics:")
        logger.info(f"Noise - Mean: {noise_stats['mean']:.6f}, Std: {noise_stats['std']:.6f}, Max: {noise_stats['max']:.6f}")
        logger.info(f"Signal - Mean: {signal_stats['mean']:.6f}, Std: {signal_stats['std']:.6f}, Max: {signal_stats['max']:.6f}")
        
        # Check if signals are distinguishable
        signal_to_noise_ratio = signal_stats['max'] / noise_stats['max']
        logger.info(f"Signal-to-noise ratio (max values): {signal_to_noise_ratio:.3f}")
        
        if signal_to_noise_ratio < 1.5:
            logger.warning("Signal and noise appear very similar - this could explain poor model performance!")
        else:
            logger.info("Signal and noise appear distinguishable")
        
        return noise_stats, signal_stats
        
    except Exception as e:
        logger.error(f"Error creating comparison plots: {e}")
        raise

def main():
    """Main function."""
    logger.info("Starting test data inspection...")
    
    try:
        noise_stats, signal_stats = create_comparison_plots()
        logger.info("Data inspection completed successfully!")
        
    except Exception as e:
        logger.error(f"Data inspection failed: {e}")
        raise

if __name__ == "__main__":
    main()
