#!/usr/bin/env python3
"""
Verify Gravitational Wave Signals in Raw Data
Analyze raw strain data to confirm signal presence before CWT processing
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_manifest():
    """Load the download manifest to identify signal vs noise segments"""
    manifest_path = Path("data/download_manifest.json")
    if not manifest_path.exists():
        print("No manifest found")
        return {}
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Create GPS time to segment type mapping
    gps_to_type = {}
    for entry in manifest.get('downloads', []):
        gps_time = entry.get('start_gps')
        segment_type = entry.get('segment_type', 'noise')
        event_name = entry.get('event', 'unknown')
        if gps_time:
            gps_to_type[gps_time] = {'type': segment_type, 'event': event_name}
    
    return gps_to_type

def analyze_raw_strain_data():
    """Analyze raw strain data to verify signal presence"""
    print("=== Raw Strain Data Analysis ===")
    
    # Load manifest
    gps_to_info = load_manifest()
    
    # Load raw data files
    raw_dir = Path("data/raw")
    raw_files = list(raw_dir.glob("*.npz"))
    
    if not raw_files:
        print("No raw files found")
        return
    
    print(f"Found {len(raw_files)} raw files")
    
    # Separate signal and noise files
    signal_files = []
    noise_files = []
    
    for raw_file in raw_files:
        try:
            # Extract GPS time from filename (e.g., H1_1126259462_32s.npz)
            filename_parts = raw_file.stem.split('_')
            if len(filename_parts) >= 2:
                gps_time = int(filename_parts[1].split('s')[0])
                info = gps_to_info.get(gps_time, {'type': 'noise', 'event': 'unknown'})
                
                if info['type'] == 'signal':
                    signal_files.append((raw_file, info['event']))
                else:
                    noise_files.append((raw_file, 'noise'))
        except (ValueError, IndexError):
            noise_files.append((raw_file, 'noise'))
    
    print(f"Found {len(signal_files)} signal files, {len(noise_files)} noise files")
    
    if not signal_files:
        print("No signal files found for analysis")
        return
    
    # Analyze first few signal and noise files
    print(f"\n=== Signal Analysis ===")
    signal_stats = []
    
    for i, (signal_file, event) in enumerate(signal_files[:5]):  # Analyze first 5 signals
        print(f"\nSignal {i+1}: {signal_file.name} (Event: {event})")
        
        try:
            data = np.load(signal_file)
            strain = data['strain']
            
            # Basic statistics
            stats = {
                'file': signal_file.name,
                'event': event,
                'shape': strain.shape,
                'mean': strain.mean(),
                'std': strain.std(),
                'min': strain.min(),
                'max': strain.max(),
                'range': strain.max() - strain.min()
            }
            
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:.6e}")
            print(f"  Std: {stats['std']:.6e}")
            print(f"  Range: {stats['min']:.6e} to {stats['max']:.6e}")
            print(f"  Amplitude: {stats['range']:.6e}")
            
            signal_stats.append(stats)
            
        except Exception as e:
            print(f"  Error loading {signal_file.name}: {e}")
    
    print(f"\n=== Noise Analysis ===")
    noise_stats = []
    
    for i, (noise_file, event) in enumerate(noise_files[:5]):  # Analyze first 5 noise files
        print(f"\nNoise {i+1}: {noise_file.name}")
        
        try:
            data = np.load(noise_file)
            strain = data['strain']
            
            # Basic statistics
            stats = {
                'file': noise_file.name,
                'event': 'noise',
                'shape': strain.shape,
                'mean': strain.mean(),
                'std': strain.std(),
                'min': strain.min(),
                'max': strain.max(),
                'range': strain.max() - strain.min()
            }
            
            print(f"  Shape: {stats['shape']}")
            print(f"  Mean: {stats['mean']:.6e}")
            print(f"  Std: {stats['std']:.6e}")
            print(f"  Range: {stats['min']:.6e} to {stats['max']:.6e}")
            print(f"  Amplitude: {stats['range']:.6e}")
            
            noise_stats.append(stats)
            
        except Exception as e:
            print(f"  Error loading {noise_file.name}: {e}")
    
    # Compare statistics
    if signal_stats and noise_stats:
        print(f"\n=== Statistical Comparison ===")
        
        signal_means = [s['mean'] for s in signal_stats]
        noise_means = [s['mean'] for s in noise_stats]
        
        signal_stds = [s['std'] for s in signal_stats]
        noise_stds = [s['std'] for s in noise_stats]
        
        signal_ranges = [s['range'] for s in signal_stats]
        noise_ranges = [s['range'] for s in noise_stats]
        
        print(f"Signal mean: {np.mean(signal_means):.6e} ± {np.std(signal_means):.6e}")
        print(f"Noise mean: {np.mean(noise_means):.6e} ± {np.std(noise_means):.6e}")
        print(f"Mean difference: {abs(np.mean(signal_means) - np.mean(noise_means)):.6e}")
        
        print(f"Signal std: {np.mean(signal_stds):.6e} ± {np.std(signal_stds):.6e}")
        print(f"Noise std: {np.mean(noise_stds):.6e} ± {np.std(noise_stds):.6e}")
        print(f"Std difference: {abs(np.mean(signal_stds) - np.mean(noise_stds)):.6e}")
        
        print(f"Signal range: {np.mean(signal_ranges):.6e} ± {np.std(signal_ranges):.6e}")
        print(f"Noise range: {np.mean(noise_ranges):.6e} ± {np.std(noise_ranges):.6e}")
        print(f"Range difference: {abs(np.mean(signal_ranges) - np.mean(noise_ranges)):.6e}")
        
        # Check for significant differences
        mean_diff = abs(np.mean(signal_means) - np.mean(noise_means))
        std_diff = abs(np.mean(signal_stds) - np.mean(noise_stds))
        range_diff = abs(np.mean(signal_ranges) - np.mean(noise_ranges))
        
        if mean_diff > 1e-20 or std_diff > 1e-20 or range_diff > 1e-20:
            print("SUCCESS: Significant differences found between signal and noise!")
        else:
            print("WARNING: No significant differences in raw strain data")
    
    return signal_stats, noise_stats

def analyze_time_domain_features():
    """Analyze time-domain features that might indicate gravitational wave signals"""
    print(f"\n=== Time-Domain Feature Analysis ===")
    
    # Load manifest
    gps_to_info = load_manifest()
    
    # Load raw data files
    raw_dir = Path("data/raw")
    raw_files = list(raw_dir.glob("*.npz"))
    
    # Find signal and noise files
    signal_file = None
    noise_file = None
    
    for raw_file in raw_files:
        try:
            filename_parts = raw_file.stem.split('_')
            if len(filename_parts) >= 2:
                gps_time = int(filename_parts[1].split('s')[0])
                info = gps_to_info.get(gps_time, {'type': 'noise', 'event': 'unknown'})
                
                if info['type'] == 'signal' and signal_file is None:
                    signal_file = raw_file
                elif info['type'] == 'noise' and noise_file is None:
                    noise_file = raw_file
                
                if signal_file and noise_file:
                    break
        except (ValueError, IndexError):
            continue
    
    if not signal_file or not noise_file:
        print("Could not find both signal and noise files for comparison")
        return
    
    print(f"Analyzing signal file: {signal_file.name}")
    print(f"Analyzing noise file: {noise_file.name}")
    
    # Load data
    signal_data = np.load(signal_file)
    noise_data = np.load(noise_file)
    
    signal_strain = signal_data['strain']
    noise_strain = noise_data['strain']
    
    sample_rate = signal_data.get('sample_rate', 4096)
    duration = signal_data.get('duration', 32)
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {duration} seconds")
    print(f"Signal length: {len(signal_strain)} samples")
    print(f"Noise length: {len(noise_strain)} samples")
    
    # Time-domain analysis
    t_signal = np.linspace(0, duration, len(signal_strain))
    t_noise = np.linspace(0, duration, len(noise_strain))
    
    # 1. Power spectral density
    from scipy.signal import welch
    
    f_signal, psd_signal = welch(signal_strain, fs=sample_rate, nperseg=1024)
    f_noise, psd_noise = welch(noise_strain, fs=sample_rate, nperseg=1024)
    
    print(f"\n--- Power Spectral Density Analysis ---")
    print(f"Signal PSD range: {psd_signal.min():.6e} to {psd_signal.max():.6e}")
    print(f"Noise PSD range: {psd_noise.min():.6e} to {psd_noise.max():.6e}")
    
    # Look for frequency content differences
    freq_range_20_200 = (f_signal >= 20) & (f_signal <= 200)
    signal_power_20_200 = np.mean(psd_signal[freq_range_20_200])
    noise_power_20_200 = np.mean(psd_noise[freq_range_20_200])
    
    print(f"Signal power (20-200 Hz): {signal_power_20_200:.6e}")
    print(f"Noise power (20-200 Hz): {noise_power_20_200:.6e}")
    print(f"Power ratio: {signal_power_20_200 / noise_power_20_200:.3f}")
    
    # 2. Time-domain variance analysis
    window_size = 1024  # ~0.25 seconds
    signal_variances = []
    noise_variances = []
    
    for i in range(0, len(signal_strain) - window_size, window_size):
        signal_variances.append(np.var(signal_strain[i:i+window_size]))
        noise_variances.append(np.var(noise_strain[i:i+window_size]))
    
    signal_variances = np.array(signal_variances)
    noise_variances = np.array(noise_variances)
    
    print(f"\n--- Time-Domain Variance Analysis ---")
    print(f"Signal variance: {signal_variances.mean():.6e} ± {signal_variances.std():.6e}")
    print(f"Noise variance: {noise_variances.mean():.6e} ± {noise_variances.std():.6e}")
    print(f"Variance ratio: {signal_variances.mean() / noise_variances.mean():.3f}")
    
    # 3. Look for chirp-like patterns (frequency evolution)
    print(f"\n--- Chirp Pattern Analysis ---")
    
    # Simple frequency tracking using zero-crossing rate
    def estimate_frequency(signal, window_size=512):
        """Estimate instantaneous frequency using zero-crossing rate"""
        frequencies = []
        for i in range(0, len(signal) - window_size, window_size//2):
            window = signal[i:i+window_size]
            # Count zero crossings
            zero_crossings = np.sum(np.diff(np.sign(window)) != 0)
            # Estimate frequency
            freq = zero_crossings * sample_rate / (2 * window_size)
            frequencies.append(freq)
        return np.array(frequencies)
    
    signal_freqs = estimate_frequency(signal_strain)
    noise_freqs = estimate_frequency(noise_strain)
    
    print(f"Signal frequency range: {signal_freqs.min():.1f} to {signal_freqs.max():.1f} Hz")
    print(f"Noise frequency range: {noise_freqs.min():.1f} to {noise_freqs.max():.1f} Hz")
    
    # Check for frequency evolution (chirp signature)
    signal_freq_trend = np.polyfit(range(len(signal_freqs)), signal_freqs, 1)[0]
    noise_freq_trend = np.polyfit(range(len(noise_freqs)), noise_freqs, 1)[0]
    
    print(f"Signal frequency trend: {signal_freq_trend:.3f} Hz/sample")
    print(f"Noise frequency trend: {noise_freq_trend:.3f} Hz/sample")
    
    if abs(signal_freq_trend) > abs(noise_freq_trend) * 2:
        print("POTENTIAL CHIRP DETECTED: Signal shows frequency evolution!")
    else:
        print("No clear chirp pattern detected in frequency evolution")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Time series
    axes[0, 0].plot(t_signal, signal_strain, 'b-', alpha=0.7)
    axes[0, 0].set_title('Signal - Time Series')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    
    axes[1, 0].plot(t_noise, noise_strain, 'r-', alpha=0.7)
    axes[1, 0].set_title('Noise - Time Series')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Strain')
    
    # PSD
    axes[0, 1].loglog(f_signal, psd_signal, 'b-')
    axes[0, 1].set_title('Signal - Power Spectral Density')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_xlim(10, 1000)
    
    axes[1, 1].loglog(f_noise, psd_noise, 'r-')
    axes[1, 1].set_title('Noise - Power Spectral Density')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD')
    axes[1, 1].set_xlim(10, 1000)
    
    # Frequency evolution
    axes[0, 2].plot(signal_freqs, 'b-')
    axes[0, 2].set_title('Signal - Frequency Evolution')
    axes[0, 2].set_xlabel('Time Window')
    axes[0, 2].set_ylabel('Frequency (Hz)')
    
    axes[1, 2].plot(noise_freqs, 'r-')
    axes[1, 2].set_title('Noise - Frequency Evolution')
    axes[1, 2].set_xlabel('Time Window')
    axes[1, 2].set_ylabel('Frequency (Hz)')
    
    plt.tight_layout()
    
    output_dir = Path("results/spectrogram_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "raw_data_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to {output_dir / 'raw_data_analysis.png'}")

def main():
    """Main analysis function"""
    print("=== Gravitational Wave Signal Verification ===")
    
    # Basic statistical analysis
    signal_stats, noise_stats = analyze_raw_strain_data()
    
    # Time-domain feature analysis
    analyze_time_domain_features()
    
    print(f"\n=== Summary ===")
    print("This analysis examines raw strain data to verify the presence of")
    print("gravitational wave signals before any CWT processing.")
    print("Key indicators to look for:")
    print("1. Statistical differences between signal and noise")
    print("2. Power spectral density differences")
    print("3. Time-domain variance patterns")
    print("4. Frequency evolution (chirp signatures)")

if __name__ == "__main__":
    main()
