#!/usr/bin/env python3
"""
Visualize Corrected CWT Spectrograms
Show noise vs signal spectrograms using the corrected CWT preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.cwt import CWTPreprocessor

def visualize_corrected_spectrograms():
    """Visualize noise vs signal spectrograms with corrected CWT"""
    print("=== Visualizing Corrected CWT Spectrograms ===")
    
    # Load manifest
    manifest_path = Path("data/download_manifest.json")
    if not manifest_path.exists():
        print("No manifest found")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Create GPS time to segment type mapping
    gps_to_info = {}
    for entry in manifest.get('downloads', []):
        gps_time = entry.get('start_gps')
        segment_type = entry.get('segment_type', 'noise')
        event_name = entry.get('event', 'unknown')
        if gps_time:
            gps_to_info[gps_time] = {'type': segment_type, 'event': event_name}
    
    # Load raw data files
    raw_dir = Path("data/raw")
    raw_files = list(raw_dir.glob("*.npz"))
    
    if not raw_files:
        print("No raw files found")
        return
    
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
        print("Could not find both signal and noise files")
        return
    
    print(f"Signal file: {signal_file.name}")
    print(f"Noise file: {noise_file.name}")
    
    # Load data
    signal_data = np.load(signal_file)
    noise_data = np.load(noise_file)
    
    signal_strain = signal_data['strain']
    noise_strain = noise_data['strain']
    
    print(f"Signal strain shape: {signal_strain.shape}")
    print(f"Noise strain shape: {noise_strain.shape}")
    
    # Initialize corrected CWT preprocessor
    preprocessor = CWTPreprocessor(
        sample_rate=4096,
        target_height=64,
        target_width=32768,  # Minimal downsampling
        use_analytic=False,
        fmin=50.0,  # Focused frequency range
        fmax=200.0,  # Focused frequency range
        wavelet='morl'
    )
    
    # Process both signals
    print("Processing signal with corrected CWT...")
    signal_cwt = preprocessor.process(signal_strain)
    
    print("Processing noise with corrected CWT...")
    noise_cwt = preprocessor.process(noise_strain)
    
    print(f"Signal CWT shape: {signal_cwt.shape}")
    print(f"Noise CWT shape: {noise_cwt.shape}")
    
    # Show shared scale range
    global_min = min(signal_cwt.min(), noise_cwt.min())
    global_max = max(signal_cwt.max(), noise_cwt.max())
    print(f"Shared color scale range: {global_min:.6e} to {global_max:.6e}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Raw strain data
    t_signal = np.linspace(0, 32, len(signal_strain))
    t_noise = np.linspace(0, 32, len(noise_strain))
    
    axes[0, 0].plot(t_signal, signal_strain, 'b-', alpha=0.7, linewidth=0.5)
    axes[0, 0].set_title('Signal - Raw Strain Data')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t_noise, noise_strain, 'r-', alpha=0.7, linewidth=0.5)
    axes[1, 0].set_title('Noise - Raw Strain Data')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Strain')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Corrected CWT spectrograms - USE SHARED COLOR SCALE
    # Find the global min/max for consistent scaling
    global_min = min(signal_cwt.min(), noise_cwt.min())
    global_max = max(signal_cwt.max(), noise_cwt.max())
    
    im1 = axes[0, 1].imshow(signal_cwt, aspect='auto', origin='lower', cmap='viridis', 
                           vmin=global_min, vmax=global_max)
    axes[0, 1].set_title('Signal - Corrected CWT (50-200 Hz)')
    axes[0, 1].set_xlabel('Time Sample')
    axes[0, 1].set_ylabel('Frequency Scale')
    plt.colorbar(im1, ax=axes[0, 1], label='Magnitude')
    
    im2 = axes[1, 1].imshow(noise_cwt, aspect='auto', origin='lower', cmap='viridis',
                           vmin=global_min, vmax=global_max)
    axes[1, 1].set_title('Noise - Corrected CWT (50-200 Hz)')
    axes[1, 1].set_xlabel('Time Sample')
    axes[1, 1].set_ylabel('Frequency Scale')
    plt.colorbar(im2, ax=axes[1, 1], label='Magnitude')
    
    # Difference spectrogram
    cwt_diff = signal_cwt - noise_cwt
    im3 = axes[0, 2].imshow(cwt_diff, aspect='auto', origin='lower', cmap='RdBu_r')
    axes[0, 2].set_title('Difference (Signal - Noise)')
    axes[0, 2].set_xlabel('Time Sample')
    axes[0, 2].set_ylabel('Frequency Scale')
    plt.colorbar(im3, ax=axes[0, 2], label='Difference')
    
    # Statistics comparison
    stats_names = ['Mean', 'Std', 'Min', 'Max', 'Range']
    signal_stats = [signal_cwt.mean(), signal_cwt.std(), signal_cwt.min(), signal_cwt.max(), signal_cwt.max() - signal_cwt.min()]
    noise_stats = [noise_cwt.mean(), noise_cwt.std(), noise_cwt.min(), noise_cwt.max(), noise_cwt.max() - noise_cwt.min()]
    
    x_pos = np.arange(len(stats_names))
    axes[1, 2].bar(x_pos - 0.2, signal_stats, 0.4, label='Signal', alpha=0.7, color='blue')
    axes[1, 2].bar(x_pos + 0.2, noise_stats, 0.4, label='Noise', alpha=0.7, color='red')
    axes[1, 2].set_title('CWT Statistics Comparison')
    axes[1, 2].set_xlabel('Statistic')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(stats_names)
    axes[1, 2].legend()
    axes[1, 2].set_yscale('log')
    
    # Time-averaged frequency profiles
    signal_freq_profile = np.mean(signal_cwt, axis=1)
    noise_freq_profile = np.mean(noise_cwt, axis=1)
    freqs = np.logspace(np.log10(50), np.log10(200), 64)
    
    axes[2, 0].semilogx(freqs, signal_freq_profile, 'b-', label='Signal', linewidth=2)
    axes[2, 0].semilogx(freqs, noise_freq_profile, 'r-', label='Noise', linewidth=2)
    axes[2, 0].set_title('Time-Averaged Frequency Profile')
    axes[2, 0].set_xlabel('Frequency (Hz)')
    axes[2, 0].set_ylabel('Average Magnitude')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Frequency-averaged time profiles
    signal_time_profile = np.mean(signal_cwt, axis=0)
    noise_time_profile = np.mean(noise_cwt, axis=0)
    time_samples = np.arange(len(signal_time_profile))
    
    axes[2, 1].plot(time_samples, signal_time_profile, 'b-', label='Signal', linewidth=2)
    axes[2, 1].plot(time_samples, noise_time_profile, 'r-', label='Noise', linewidth=2)
    axes[2, 1].set_title('Frequency-Averaged Time Profile')
    axes[2, 1].set_xlabel('Time Sample')
    axes[2, 1].set_ylabel('Average Magnitude')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Difference analysis
    mean_diff = abs(signal_cwt.mean() - noise_cwt.mean())
    std_diff = abs(signal_cwt.std() - noise_cwt.std())
    range_diff = abs((signal_cwt.max() - signal_cwt.min()) - (noise_cwt.max() - noise_cwt.min()))
    
    diff_stats = [mean_diff, std_diff, range_diff]
    diff_names = ['Mean Diff', 'Std Diff', 'Range Diff']
    
    axes[2, 2].bar(diff_names, diff_stats, alpha=0.7, color='green')
    axes[2, 2].set_title('Difference Statistics')
    axes[2, 2].set_ylabel('Absolute Difference')
    axes[2, 2].set_yscale('log')
    
    # Add text summary
    mean_ratio = signal_cwt.mean() / noise_cwt.mean() if noise_cwt.mean() > 0 else float('inf')
    std_ratio = signal_cwt.std() / noise_cwt.std() if noise_cwt.std() > 0 else float('inf')
    
    summary_text = f"""Summary:
Signal/Noise Ratios:
Mean: {mean_ratio:.3f}
Std: {std_ratio:.3f}

Differences:
Mean: {mean_diff:.2e}
Std: {std_diff:.2e}
Range: {range_diff:.2e}"""
    
    axes[2, 2].text(0.02, 0.98, summary_text, transform=axes[2, 2].transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("results/spectrogram_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "corrected_cwt_spectrograms.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved corrected CWT spectrograms to {output_dir / 'corrected_cwt_spectrograms.png'}")
    
    # Print summary
    print(f"\n=== Corrected CWT Analysis Summary ===")
    print(f"Signal CWT range: {signal_cwt.min():.6e} to {signal_cwt.max():.6e}")
    print(f"Noise CWT range: {noise_cwt.min():.6e} to {noise_cwt.max():.6e}")
    print(f"Signal CWT mean: {signal_cwt.mean():.6e}")
    print(f"Noise CWT mean: {noise_cwt.mean():.6e}")
    print(f"Signal CWT std: {signal_cwt.std():.6e}")
    print(f"Noise CWT std: {noise_cwt.std():.6e}")
    print(f"Mean ratio (signal/noise): {mean_ratio:.3f}")
    print(f"Std ratio (signal/noise): {std_ratio:.3f}")
    
    if mean_ratio > 1.1 or std_ratio > 1.1:
        print("*** SIGNIFICANT DIFFERENCES DETECTED ***")
    elif mean_ratio > 1.05 or std_ratio > 1.05:
        print("* Moderate differences detected")
    else:
        print("No significant differences detected")

def main():
    """Main visualization function"""
    print("=== Corrected CWT Spectrogram Visualization ===")
    
    try:
        visualize_corrected_spectrograms()
        
        print(f"\n=== Visualization Complete ===")
        print("The corrected CWT preprocessing focuses on the 50-200 Hz frequency range")
        print("where gravitational waves show the strongest differences from noise.")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
