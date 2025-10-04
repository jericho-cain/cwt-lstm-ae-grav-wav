#!/usr/bin/env python3
"""
Analyze multiple noise and signal segments to identify timing patterns and shifts.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cwt_data(file_path):
    """Load CWT data from NumPy file."""
    try:
        cwt_data = np.load(file_path)
        # Extract metadata from filename
        filename = file_path.stem
        parts = filename.split('_')
        metadata = {
            'detector': parts[0],
            'gps_start': int(parts[1]),
            'duration': int(parts[2].replace('s', '')),
            'event_name': 'noise' if len(parts) == 3 else 'signal'
        }
        return cwt_data, metadata
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None

def plot_spectrogram(cwt_data, title, save_path=None):
    """Plot CWT spectrogram."""
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(cwt_data), aspect='auto', origin='lower', 
               extent=[0, 32, 50, 200], cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def analyze_segment(cwt_data, metadata, segment_type, segment_id):
    """Analyze a single segment and return key statistics."""
    if cwt_data is None:
        return None
    
    # Basic statistics
    stats = {
        'segment_id': segment_id,
        'segment_type': segment_type,
        'shape': cwt_data.shape,
        'mean': np.mean(cwt_data),
        'std': np.std(cwt_data),
        'min': np.min(cwt_data),
        'max': np.max(cwt_data),
        'range': np.max(cwt_data) - np.min(cwt_data),
        'gps_start': metadata.get('gps_start', 'unknown'),
        'gps_end': metadata.get('gps_end', 'unknown'),
        'event_name': metadata.get('event_name', 'noise'),
        'detector': metadata.get('detector', 'unknown')
    }
    
    # Look for prominent features (high amplitude regions)
    # Find time bins with highest average amplitude across frequencies
    time_avg_amplitude = np.mean(np.abs(cwt_data), axis=0)
    peak_time_bins = np.argsort(time_avg_amplitude)[-5:]  # Top 5 time bins
    
    stats['peak_time_bins'] = peak_time_bins
    stats['peak_amplitudes'] = time_avg_amplitude[peak_time_bins]
    
    # Convert time bins to seconds (assuming 32s duration, 32768 samples)
    time_per_sample = 32.0 / 32768
    peak_times_seconds = peak_time_bins * time_per_sample
    stats['peak_times_seconds'] = peak_times_seconds
    
    return stats

def main():
    """Analyze multiple segments to identify patterns."""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        logger.error("Processed data directory not found")
        return
    
    # Find signal and noise files
    signal_files = list(processed_dir.glob("*signal*.npy"))
    noise_files = list(processed_dir.glob("*noise*.npy"))
    
    # If no explicit signal/noise files, look for GW event files
    if not signal_files and not noise_files:
        # Look for files that might be signals (GW events have specific GPS times)
        all_files = list(processed_dir.glob("*.npy"))
        signal_files = []
        noise_files = []
        
        # GW150914 is at GPS time 1126259462
        gw150914_gps = 1126259462
        for file_path in all_files:
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    gps_time = int(parts[1])
                    # If GPS time is close to GW150914, it's likely a signal
                    if abs(gps_time - gw150914_gps) < 100000:  # Within ~1 day
                        signal_files.append(file_path)
                    else:
                        noise_files.append(file_path)
                except ValueError:
                    noise_files.append(file_path)
    
    logger.info(f"Found {len(signal_files)} signal files and {len(noise_files)} noise files")
    
    # Analyze first 5 of each type
    signal_analyses = []
    noise_analyses = []
    
    logger.info("Analyzing signal segments...")
    for i, file_path in enumerate(signal_files[:3]):  # Just 3 for visualization
        logger.info(f"  Analyzing signal {i+1}: {file_path.name}")
        cwt_data, metadata = load_cwt_data(file_path)
        analysis = analyze_segment(cwt_data, metadata, "signal", f"signal_{i+1}")
        if analysis:
            signal_analyses.append(analysis)
            # Plot spectrogram
            plot_spectrogram(cwt_data, f"Signal {i+1}: {file_path.name}")
    
    logger.info("Analyzing noise segments...")
    for i, file_path in enumerate(noise_files[:3]):  # Just 3 for visualization
        logger.info(f"  Analyzing noise {i+1}: {file_path.name}")
        cwt_data, metadata = load_cwt_data(file_path)
        analysis = analyze_segment(cwt_data, metadata, "noise", f"noise_{i+1}")
        if analysis:
            noise_analyses.append(analysis)
            # Plot spectrogram
            plot_spectrogram(cwt_data, f"Noise {i+1}: {file_path.name}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SIGNAL SEGMENTS ANALYSIS")
    print("="*80)
    for analysis in signal_analyses:
        print(f"\n{analysis['segment_id']} ({analysis['event_name']}):")
        print(f"  GPS: {analysis['gps_start']} - {analysis['gps_end']}")
        print(f"  Shape: {analysis['shape']}")
        print(f"  Range: {analysis['min']:.2e} to {analysis['max']:.2e}")
        print(f"  Mean: {analysis['mean']:.2e}, Std: {analysis['std']:.2e}")
        print(f"  Peak times (seconds): {analysis['peak_times_seconds']}")
        print(f"  Peak amplitudes: {analysis['peak_amplitudes']}")
    
    print("\n" + "="*80)
    print("NOISE SEGMENTS ANALYSIS")
    print("="*80)
    for analysis in noise_analyses:
        print(f"\n{analysis['segment_id']}:")
        print(f"  GPS: {analysis['gps_start']} - {analysis['gps_end']}")
        print(f"  Shape: {analysis['shape']}")
        print(f"  Range: {analysis['min']:.2e} to {analysis['max']:.2e}")
        print(f"  Mean: {analysis['mean']:.2e}, Std: {analysis['std']:.2e}")
        print(f"  Peak times (seconds): {analysis['peak_times_seconds']}")
        print(f"  Peak amplitudes: {analysis['peak_amplitudes']}")
    
    # Look for patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    # Check if signals consistently have features at specific times
    signal_peak_times = []
    for analysis in signal_analyses:
        signal_peak_times.extend(analysis['peak_times_seconds'])
    
    noise_peak_times = []
    for analysis in noise_analyses:
        noise_peak_times.extend(analysis['peak_times_seconds'])
    
    print(f"Signal peak times: {signal_peak_times}")
    print(f"Noise peak times: {noise_peak_times}")
    
    # Check for consistent timing patterns
    signal_peak_times = np.array(signal_peak_times)
    noise_peak_times = np.array(noise_peak_times)
    
    print(f"\nSignal peak time statistics:")
    print(f"  Mean: {np.mean(signal_peak_times):.2f}s")
    print(f"  Std: {np.std(signal_peak_times):.2f}s")
    print(f"  Range: {np.min(signal_peak_times):.2f}s to {np.max(signal_peak_times):.2f}s")
    
    print(f"\nNoise peak time statistics:")
    print(f"  Mean: {np.mean(noise_peak_times):.2f}s")
    print(f"  Std: {np.std(noise_peak_times):.2f}s")
    print(f"  Range: {np.min(noise_peak_times):.2f}s to {np.max(noise_peak_times):.2f}s")
    
    # Check for amplitude differences
    signal_amplitudes = [analysis['max'] for analysis in signal_analyses]
    noise_amplitudes = [analysis['max'] for analysis in noise_analyses]
    
    print(f"\nAmplitude comparison:")
    print(f"  Signal max amplitudes: {signal_amplitudes}")
    print(f"  Noise max amplitudes: {noise_amplitudes}")
    print(f"  Signal mean max: {np.mean(signal_amplitudes):.2e}")
    print(f"  Noise mean max: {np.mean(noise_amplitudes):.2e}")

if __name__ == "__main__":
    main()
