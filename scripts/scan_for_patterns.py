#!/usr/bin/env python3
"""
Scan CWT data to find regions with largest differences/patterns
and plot around those areas to find gravitational wave features.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_for_patterns(noise_cwt, signal_cwt, window_size=4):
    """Scan the CWT data to find regions with largest differences/patterns"""
    
    total_time = 32  # seconds
    total_samples = noise_cwt.shape[1]
    samples_per_second = total_samples / total_time
    window_samples = int(window_size * samples_per_second)
    
    logger.info(f"Scanning {total_time}s data with {window_size}s windows")
    logger.info(f"Window size: {window_samples} samples")
    
    # Calculate sliding window statistics
    noise_stats = []
    signal_stats = []
    time_windows = []
    
    step_size = window_samples // 4  # Overlap windows for better coverage
    
    for start_sample in range(0, total_samples - window_samples, step_size):
        end_sample = start_sample + window_samples
        
        # Extract window
        noise_window = noise_cwt[:, start_sample:end_sample]
        signal_window = signal_cwt[:, start_sample:end_sample]
        
        # Calculate statistics
        noise_mean = np.mean(noise_window)
        noise_std = np.std(noise_window)
        noise_max = np.max(noise_window)
        noise_range = np.max(noise_window) - np.min(noise_window)
        
        signal_mean = np.mean(signal_window)
        signal_std = np.std(signal_window)
        signal_max = np.max(signal_window)
        signal_range = np.max(signal_window) - np.min(signal_window)
        
        # Calculate differences
        mean_diff = abs(signal_mean - noise_mean)
        std_diff = abs(signal_std - noise_std)
        max_diff = abs(signal_max - noise_max)
        range_diff = abs(signal_range - noise_range)
        
        # Store results
        time_start = start_sample / samples_per_second
        time_windows.append(time_start)
        
        noise_stats.append({
            'mean': noise_mean, 'std': noise_std, 'max': noise_max, 'range': noise_range,
            'start_sample': start_sample, 'end_sample': end_sample
        })
        
        signal_stats.append({
            'mean': signal_mean, 'std': signal_std, 'max': signal_max, 'range': signal_range,
            'mean_diff': mean_diff, 'std_diff': std_diff, 'max_diff': max_diff, 'range_diff': range_diff,
            'start_sample': start_sample, 'end_sample': end_sample
        })
    
    return time_windows, noise_stats, signal_stats

def find_interesting_regions(time_windows, signal_stats, top_n=3):
    """Find the most interesting regions based on differences"""
    
    # Sort by different criteria
    max_diff_regions = sorted(enumerate(signal_stats), key=lambda x: x[1]['max_diff'], reverse=True)[:top_n]
    range_diff_regions = sorted(enumerate(signal_stats), key=lambda x: x[1]['range_diff'], reverse=True)[:top_n]
    std_diff_regions = sorted(enumerate(signal_stats), key=lambda x: x[1]['std_diff'], reverse=True)[:top_n]
    
    logger.info(f"\n=== TOP {top_n} REGIONS BY MAX DIFFERENCE ===")
    for i, (idx, stats) in enumerate(max_diff_regions):
        time_start = time_windows[idx]
        logger.info(f"{i+1}. Time {time_start:.1f}s: max_diff={stats['max_diff']:.6f}, range_diff={stats['range_diff']:.6f}")
    
    logger.info(f"\n=== TOP {top_n} REGIONS BY RANGE DIFFERENCE ===")
    for i, (idx, stats) in enumerate(range_diff_regions):
        time_start = time_windows[idx]
        logger.info(f"{i+1}. Time {time_start:.1f}s: range_diff={stats['range_diff']:.6f}, max_diff={stats['max_diff']:.6f}")
    
    return max_diff_regions, range_diff_regions, std_diff_regions

def plot_interesting_regions(noise_cwt, signal_cwt, interesting_regions, freqs, fs=4096):
    """Plot the most interesting regions"""
    
    # Define frequency bands
    frequency_bands = [
        (20, 50, "20-50 Hz"),
        (50, 100, "50-100 Hz"), 
        (100, 200, "100-200 Hz")
    ]
    
    # Create plots for each interesting region
    for region_idx, (window_idx, stats) in enumerate(interesting_regions):
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        start_sample = stats['start_sample']
        end_sample = stats['end_sample']
        
        # Extract time segment
        noise_segment = noise_cwt[:, start_sample:end_sample]
        signal_segment = signal_cwt[:, start_sample:end_sample]
        
        time_start = start_sample / (noise_cwt.shape[1] / 32)  # Convert to seconds
        time_duration = (end_sample - start_sample) / (noise_cwt.shape[1] / 32)
        
        logger.info(f"Plotting region {region_idx + 1}: {time_start:.1f}s - {time_start + time_duration:.1f}s")
        
        for i, (fmin, fmax, title) in enumerate(frequency_bands):
            
            # Extract frequency band
            valid_indices = (freqs >= fmin) & (freqs <= fmax)
            
            if not np.any(valid_indices):
                continue
            
            noise_band = noise_segment[valid_indices, :]
            signal_band = signal_segment[valid_indices, :]
            
            # Plot noise (top row)
            im1 = axes[0, i].imshow(noise_band, aspect='auto', origin='lower', 
                                   extent=[time_start, time_start + time_duration, fmin, fmax], 
                                   cmap='viridis')
            axes[0, i].set_title(f'Noise - {title}')
            axes[0, i].set_ylabel('Frequency (Hz)')
            
            # Plot signal (bottom row)
            im2 = axes[1, i].imshow(signal_band, aspect='auto', origin='lower', 
                                   extent=[time_start, time_start + time_duration, fmin, fmax], 
                                   cmap='viridis')
            axes[1, i].set_title(f'Gravitational Wave - {title}')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('Frequency (Hz)')
            
            # Add colorbars
            plt.colorbar(im1, ax=axes[0, i], label='Magnitude')
            plt.colorbar(im2, ax=axes[1, i], label='Magnitude')
        
        # Set main title
        fig.suptitle(f'Region {region_idx + 1}: Time {time_start:.1f}s-{time_start + time_duration:.1f}s (Max Diff: {stats["max_diff"]:.6f})', 
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f'interesting_region_{region_idx + 1}.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved interesting region {region_idx + 1}: interesting_region_{region_idx + 1}.png")

def main():
    """Scan data and plot interesting regions"""
    
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
    
    # Generate frequency array
    freqs = np.logspace(np.log10(20), np.log10(512), 64)
    
    # Scan for patterns
    time_windows, noise_stats, signal_stats = scan_for_patterns(noise_cwt, signal_cwt, window_size=4)
    
    # Find interesting regions
    max_diff_regions, range_diff_regions, std_diff_regions = find_interesting_regions(time_windows, signal_stats, top_n=3)
    
    # Plot the most interesting regions
    logger.info(f"\n=== PLOTTING MOST INTERESTING REGIONS ===")
    plot_interesting_regions(noise_cwt, signal_cwt, max_diff_regions, freqs)
    
    print(f"\n=== SCAN COMPLETE ===")
    print(f"Found regions with largest differences between signal and noise")
    print(f"Plotted the top 3 most interesting regions")
    print(f"This should reveal where the gravitational wave features are located!")

if __name__ == "__main__":
    main()
