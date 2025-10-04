#!/usr/bin/env python3
"""
Create frequency-band comparison plots matching the paper's format.
Shows zoomed frequency bands and time segments like the paper spectrograms.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_frequency_band(cwt_data, fs, target_fmin, target_fmax, scales, freqs):
    """Extract specific frequency band from CWT data"""
    
    # Find scales corresponding to target frequency range
    # CWT frequencies are: freqs = fs / scales
    # So scales = fs / freqs
    
    # Find indices for target frequency range
    valid_indices = (freqs >= target_fmin) & (freqs <= target_fmax)
    
    if not np.any(valid_indices):
        logger.warning(f"No scales found for frequency range {target_fmin}-{target_fmax} Hz")
        return None, None, None
    
    # Extract frequency band
    band_data = cwt_data[valid_indices, :]
    band_freqs = freqs[valid_indices]
    band_scales = scales[valid_indices]
    
    logger.info(f"Extracted frequency band {target_fmin}-{target_fmax} Hz: {band_data.shape}")
    logger.info(f"Actual frequency range: {band_freqs.min():.1f}-{band_freqs.max():.1f} Hz")
    
    return band_data, band_freqs, band_scales

def extract_time_segment(cwt_data, total_duration=32, segment_duration=4, segment_start=0):
    """Extract specific time segment from CWT data"""
    
    # Calculate time indices
    total_samples = cwt_data.shape[1]
    samples_per_second = total_samples / total_duration
    
    start_sample = int(segment_start * samples_per_second)
    end_sample = int((segment_start + segment_duration) * samples_per_second)
    
    # Ensure we don't go out of bounds
    start_sample = max(0, start_sample)
    end_sample = min(total_samples, end_sample)
    
    # Extract time segment
    segment_data = cwt_data[:, start_sample:end_sample]
    
    logger.info(f"Extracted time segment {segment_start}-{segment_start + segment_duration}s: {segment_data.shape}")
    
    return segment_data

def create_frequency_band_plots(noise_cwt, signal_cwt, freqs, fs=4096, scales=None):
    """Create frequency-band comparison plots matching the paper format"""
    
    # Define frequency bands like the paper
    frequency_bands = [
        (20, 50, "20-50 Hz"),
        (50, 100, "50-100 Hz"), 
        (100, 200, "100-200 Hz")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (fmin, fmax, title) in enumerate(frequency_bands):
        
        # Extract frequency band for both noise and signal
        noise_band, noise_freqs, _ = extract_frequency_band(noise_cwt, fs, fmin, fmax, scales, freqs)
        signal_band, signal_freqs, _ = extract_frequency_band(signal_cwt, fs, fmin, fmax, scales, freqs)
        
        if noise_band is None or signal_band is None:
            continue
        
        # Extract 4-second time segment (like the paper)
        noise_segment = extract_time_segment(noise_band, total_duration=32, segment_duration=4, segment_start=14)
        signal_segment = extract_time_segment(signal_band, total_duration=32, segment_duration=4, segment_start=14)
        
        # Use proper extent for time and frequency scales
        time_extent = [0, 4]  # 4-second segment
        freq_extent = [fmin, fmax]  # Frequency band
        
        # Plot noise (top row) with proper extent
        im1 = axes[0, i].imshow(noise_segment, aspect='auto', origin='lower', 
                               extent=[*time_extent, *freq_extent], cmap='viridis')
        axes[0, i].set_title(f'Noise - {title}')
        axes[0, i].set_ylabel('Frequency (Hz)')
        
        # Plot signal (bottom row) with proper extent
        im2 = axes[1, i].imshow(signal_segment, aspect='auto', origin='lower', 
                               extent=[*time_extent, *freq_extent], cmap='viridis')
        axes[1, i].set_title(f'Gravitational Wave - {title}')
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Frequency (Hz)')
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[0, i], label='Magnitude')
        plt.colorbar(im2, ax=axes[1, i], label='Magnitude')
    
    # Set main title
    fig.suptitle('Frequency Band Comparison: Noise vs Gravitational Waves (Real LIGO Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('paper_style_frequency_bands.png', dpi=150, bbox_inches='tight')
    logger.info("Saved paper-style frequency band plots: paper_style_frequency_bands.png")

def main():
    """Create paper-style plots from legacy CWT data"""
    
    # Load legacy CWT data
    legacy_dir = Path("data/legacy_cwt_test")
    
    noise_file = legacy_dir / "legacy_H1_1126124017_32s_cwt.npy"
    signal_file = legacy_dir / "legacy_H1_1164686041_32s_cwt.npy"
    
    if not noise_file.exists() or not signal_file.exists():
        logger.error("Legacy CWT files not found. Run test_legacy_cwt.py first.")
        return
    
    # Load data
    noise_cwt = np.load(noise_file)
    signal_cwt = np.load(signal_file)
    
    logger.info(f"Loaded noise CWT: {noise_cwt.shape}")
    logger.info(f"Loaded signal CWT: {signal_cwt.shape}")
    
    # Generate frequency array (assuming standard CWT scales)
    # Legacy CWT uses 64 scales from 20-512 Hz
    freqs = np.logspace(np.log10(20), np.log10(512), 64)
    scales = 4096 / freqs  # fs = 4096 Hz
    
    logger.info(f"Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    
    # Create paper-style plots
    create_frequency_band_plots(noise_cwt, signal_cwt, freqs, fs=4096, scales=scales)
    
    print(f"\n=== PAPER-STYLE PLOTS CREATED ===")
    print(f"Frequency bands: 20-50 Hz, 50-100 Hz, 100-200 Hz")
    print(f"Time segments: 4-second windows")
    print(f"This should match the paper's spectrogram format")

if __name__ == "__main__":
    main()
