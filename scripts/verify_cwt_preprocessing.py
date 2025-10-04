#!/usr/bin/env python3
"""
Verify CWT Preprocessing Results

This script loads processed CWT data and compares signal vs noise samples
to verify the preprocessing is working correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_manifest():
    """Load the download manifest to identify signal vs noise files."""
    manifest_path = Path("data/download_manifest.json")
    if not manifest_path.exists():
        logger.error("Manifest file not found!")
        return {}
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Create mapping from GPS time to segment type
    gps_to_type = {}
    for download in manifest['downloads']:
        if download.get('successful', False):
            gps_time = download.get('start_gps')
            segment_type = download.get('segment_type')
            if gps_time and segment_type:
                gps_to_type[gps_time] = segment_type
    
    return gps_to_type

def find_sample_files(processed_dir, gps_to_type, num_samples=3):
    """Find sample signal and noise files from processed data."""
    cwt_files = list(processed_dir.glob("*.npy"))
    
    signal_files = []
    noise_files = []
    
    for file_path in cwt_files:
        try:
            # Extract GPS time from filename (H1_<GPS>_32s_cwt.npy)
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
    
    # Sort files and pick different ones (skip first few)
    signal_files.sort()
    noise_files.sort()
    
    # Pick different files - skip the first 2 and take the next ones
    selected_signals = signal_files[2:2+num_samples] if len(signal_files) > 2 else signal_files[:num_samples]
    selected_noise = noise_files[2:2+num_samples] if len(noise_files) > 2 else noise_files[:num_samples]
    
    return selected_signals, selected_noise

def analyze_cwt_data(data, title):
    """Analyze CWT data and return statistics."""
    stats = {
        'shape': data.shape,
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'range': np.max(data) - np.min(data),
        'non_zero_count': np.count_nonzero(data),
        'total_elements': data.size,
        'non_zero_ratio': np.count_nonzero(data) / data.size
    }
    
    logger.info(f"\n{title} Statistics:")
    logger.info(f"  Shape: {stats['shape']}")
    logger.info(f"  Range: {stats['min']:.2e} to {stats['max']:.2e}")
    logger.info(f"  Mean: {stats['mean']:.2e}")
    logger.info(f"  Std: {stats['std']:.2e}")
    logger.info(f"  Non-zero ratio: {stats['non_zero_ratio']:.1%}")
    
    return stats

def plot_spectrograms(signal_data, noise_data, signal_title, noise_title):
    """Plot spectrograms for visual comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Signal spectrogram
    im1 = axes[0, 0].imshow(signal_data, aspect='auto', origin='lower', 
                           extent=[0, 32, 50, 200], cmap='viridis')
    axes[0, 0].set_title(f'{signal_title} - CWT Spectrogram')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axes[0, 0], label='Magnitude')
    
    # Noise spectrogram
    im2 = axes[0, 1].imshow(noise_data, aspect='auto', origin='lower',
                           extent=[0, 32, 50, 200], cmap='viridis')
    axes[0, 1].set_title(f'{noise_title} - CWT Spectrogram')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=axes[0, 1], label='Magnitude')
    
    # Use shared color scale for better comparison
    vmin = min(np.min(signal_data), np.min(noise_data))
    vmax = max(np.max(signal_data), np.max(noise_data))
    
    im3 = axes[1, 0].imshow(signal_data, aspect='auto', origin='lower',
                           extent=[0, 32, 50, 200], cmap='viridis', 
                           vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'{signal_title} - Shared Scale')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im3, ax=axes[1, 0], label='Magnitude')
    
    im4 = axes[1, 1].imshow(noise_data, aspect='auto', origin='lower',
                           extent=[0, 32, 50, 200], cmap='viridis',
                           vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'{noise_title} - Shared Scale')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im4, ax=axes[1, 1], label='Magnitude')
    
    plt.tight_layout()
    plt.savefig('cwt_verification_plots.png', dpi=300, bbox_inches='tight')
    logger.info("Spectrograms saved to 'cwt_verification_plots.png'")
    
    return vmin, vmax

def main():
    """Main verification function."""
    logger.info("Verifying CWT Preprocessing Results")
    logger.info("=" * 50)
    
    # Load manifest
    gps_to_type = load_manifest()
    logger.info(f"Loaded manifest with {len(gps_to_type)} entries")
    
    # Find processed data
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        logger.error("Processed data directory not found!")
        return
    
    # Find sample files
    signal_files, noise_files = find_sample_files(processed_dir, gps_to_type, num_samples=3)
    
    logger.info(f"Found {len(signal_files)} signal files and {len(noise_files)} noise files")
    
    if not signal_files or not noise_files:
        logger.error("Need both signal and noise files for comparison!")
        return
    
    # Load and analyze samples
    logger.info("\n" + "="*50)
    logger.info("SIGNAL ANALYSIS")
    logger.info("="*50)
    
    signal_stats = []
    for i, signal_file in enumerate(signal_files[:2]):  # Analyze first 2 signals
        logger.info(f"\nSignal {i+1}: {signal_file.name}")
        signal_data = np.load(signal_file)
        stats = analyze_cwt_data(signal_data, f"Signal {i+1}")
        signal_stats.append(stats)
    
    logger.info("\n" + "="*50)
    logger.info("NOISE ANALYSIS")
    logger.info("="*50)
    
    noise_stats = []
    for i, noise_file in enumerate(noise_files[:2]):  # Analyze first 2 noise samples
        logger.info(f"\nNoise {i+1}: {noise_file.name}")
        noise_data = np.load(noise_file)
        stats = analyze_cwt_data(noise_data, f"Noise {i+1}")
        noise_stats.append(stats)
    
    # Compare statistics
    logger.info("\n" + "="*50)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*50)
    
    signal_means = [s['mean'] for s in signal_stats]
    noise_means = [s['mean'] for s in noise_stats]
    signal_stds = [s['std'] for s in signal_stats]
    noise_stds = [s['std'] for s in noise_stats]
    
    logger.info(f"Signal means: {[f'{m:.2e}' for m in signal_means]}")
    logger.info(f"Noise means:  {[f'{m:.2e}' for m in noise_means]}")
    logger.info(f"Signal stds:  {[f'{s:.2e}' for s in signal_stds]}")
    logger.info(f"Noise stds:   {[f'{s:.2e}' for s in noise_stds]}")
    
    # Check for variation
    signal_mean_var = np.std(signal_means) / np.mean(signal_means) if np.mean(signal_means) != 0 else 0
    noise_mean_var = np.std(noise_means) / np.mean(noise_means) if np.mean(noise_means) != 0 else 0
    
    logger.info(f"\nVariation Analysis:")
    logger.info(f"Signal mean variation: {signal_mean_var:.1%}")
    logger.info(f"Noise mean variation:  {noise_mean_var:.1%}")
    
    if signal_mean_var < 0.01 and noise_mean_var < 0.01:
        logger.warning("⚠️  VERY LOW VARIATION - This could explain identical reconstruction errors!")
    elif signal_mean_var < 0.05 or noise_mean_var < 0.05:
        logger.warning("⚠️  Low variation detected")
    else:
        logger.info("✅ Good variation detected")
    
    # Plot spectrograms
    logger.info("\n" + "="*50)
    logger.info("GENERATING SPECTROGRAMS")
    logger.info("="*50)
    
    # Use first signal and noise for plotting
    signal_data = np.load(signal_files[0])
    noise_data = np.load(noise_files[0])
    
    signal_title = f"Signal ({signal_files[0].name})"
    noise_title = f"Noise ({noise_files[0].name})"
    
    vmin, vmax = plot_spectrograms(signal_data, noise_data, signal_title, noise_title)
    
    logger.info(f"Shared color scale: {vmin:.2e} to {vmax:.2e}")
    
    # Final assessment
    logger.info("\n" + "="*50)
    logger.info("FINAL ASSESSMENT")
    logger.info("="*50)
    
    if signal_mean_var < 0.01 and noise_mean_var < 0.01:
        logger.error("❌ CRITICAL ISSUE: Data has almost no variation!")
        logger.error("   This explains why the model produces identical reconstruction errors.")
        logger.error("   The CWT preprocessing may be normalizing/scaling data too aggressively.")
    else:
        logger.info("✅ CWT preprocessing appears to be working correctly")
        logger.info("   Data shows sufficient variation for model learning")

if __name__ == "__main__":
    main()
