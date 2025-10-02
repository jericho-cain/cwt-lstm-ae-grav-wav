#!/usr/bin/env python3
"""
Reprocess a small sample of data with the updated CWT parameters.
"""

import sys
import os
import logging
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import yaml
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def reprocess_sample():
    """Reprocess a small sample of data with updated CWT parameters."""
    from preprocessing.cwt import fixed_preprocess_with_cwt
    
    # Load config
    with open("config/balanced_gw_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Load raw data
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
    
    # Extract strain data
    noise_strain = noise_data['strain']
    signal_strain = signal_data['strain']
    
    logger.info(f"Noise strain shape: {noise_strain.shape}")
    logger.info(f"Signal strain shape: {signal_strain.shape}")
    
    # Reprocess with updated CWT parameters
    cwt_config = config['preprocessing']['cwt']
    
    logger.info("Reprocessing noise data...")
    noise_cwt = fixed_preprocess_with_cwt(
        noise_strain.reshape(1, -1),  # Add batch dimension
        sample_rate=cwt_config['sample_rate'],
        target_height=cwt_config['target_height'],
        use_analytic=cwt_config['use_analytic'],
        fmin=cwt_config['fmin'],
        fmax=cwt_config['fmax']
    )
    
    logger.info("Reprocessing signal data...")
    signal_cwt = fixed_preprocess_with_cwt(
        signal_strain.reshape(1, -1),  # Add batch dimension
        sample_rate=cwt_config['sample_rate'],
        target_height=cwt_config['target_height'],
        use_analytic=cwt_config['use_analytic'],
        fmin=cwt_config['fmin'],
        fmax=cwt_config['fmax']
    )
    
    logger.info(f"Reprocessed noise CWT shape: {noise_cwt.shape}")
    logger.info(f"Reprocessed signal CWT shape: {signal_cwt.shape}")
    
    # Calculate statistics
    noise_stats = {
        'mean': np.mean(np.abs(noise_cwt)),
        'std': np.std(np.abs(noise_cwt)),
        'max': np.max(np.abs(noise_cwt)),
        'min': np.min(np.abs(noise_cwt))
    }
    
    signal_stats = {
        'mean': np.mean(np.abs(signal_cwt)),
        'std': np.std(np.abs(signal_cwt)),
        'max': np.max(np.abs(signal_cwt)),
        'min': np.min(np.abs(signal_cwt))
    }
    
    logger.info("Reprocessed CWT Statistics:")
    logger.info(f"Noise - Mean: {noise_stats['mean']:.6f}, Std: {noise_stats['std']:.6f}, Max: {noise_stats['max']:.6f}")
    logger.info(f"Signal - Mean: {signal_stats['mean']:.6f}, Std: {signal_stats['std']:.6f}, Max: {signal_stats['max']:.6f}")
    
    # Check if signals are distinguishable
    signal_to_noise_ratio = signal_stats['max'] / noise_stats['max']
    logger.info(f"Signal-to-noise ratio (max values): {signal_to_noise_ratio:.3f}")
    
    if signal_to_noise_ratio < 1.5:
        logger.warning("Signal and noise still appear very similar!")
    else:
        logger.info("Signal and noise now appear distinguishable!")
    
    # Save reprocessed data for comparison
    output_dir = Path("results/data_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "reprocessed_noise_cwt.npy", noise_cwt)
    np.save(output_dir / "reprocessed_signal_cwt.npy", signal_cwt)
    
    logger.info(f"Reprocessed data saved to {output_dir}")
    
    return noise_stats, signal_stats

def main():
    """Main function."""
    logger.info("Starting sample reprocessing with updated CWT parameters...")
    
    try:
        noise_stats, signal_stats = reprocess_sample()
        logger.info("Sample reprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Sample reprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
