#!/usr/bin/env python3
"""
Analyze Gravitational Wave Signals in Specific Frequency Bands
Look for signal characteristics in the frequency range where GWs are most prominent
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
import pywt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.ndimage import zoom

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def analyze_frequency_bands():
    """Analyze signal vs noise in specific frequency bands where GWs are prominent"""
    print("=== Frequency Band Analysis ===")
    
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
    
    print(f"Analyzing signal file: {signal_file.name}")
    print(f"Analyzing noise file: {noise_file.name}")
    
    # Load data
    signal_data = np.load(signal_file)
    noise_data = np.load(noise_file)
    
    signal_strain = signal_data['strain']
    noise_strain = noise_data['strain']
    
    sample_rate = int(signal_data.get('sample_rate', 4096))
    
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Signal length: {len(signal_strain)} samples")
    print(f"Noise length: {len(noise_strain)} samples")
    
    # High-pass filter
    sos = butter(4, 20, btype='high', fs=sample_rate, output='sos')
    signal_filtered = sosfiltfilt(sos, signal_strain)
    noise_filtered = sosfiltfilt(sos, noise_strain)
    
    print(f"\n=== Raw Signal Analysis ===")
    print(f"Signal raw: mean={signal_strain.mean():.6e}, std={signal_strain.std():.6e}")
    print(f"Noise raw: mean={noise_strain.mean():.6e}, std={noise_strain.std():.6e}")
    print(f"Signal filtered: mean={signal_filtered.mean():.6e}, std={signal_filtered.std():.6e}")
    print(f"Noise filtered: mean={noise_filtered.mean():.6e}, std={noise_filtered.std():.6e}")
    
    # Power spectral density analysis
    print(f"\n=== Power Spectral Density Analysis ===")
    
    f_signal, psd_signal = welch(signal_filtered, fs=sample_rate, nperseg=1024)
    f_noise, psd_noise = welch(noise_filtered, fs=sample_rate, nperseg=1024)
    
    # Analyze specific frequency bands where GWs are prominent
    frequency_bands = [
        (20, 50, "Low frequency (20-50 Hz)"),
        (50, 100, "Mid-low frequency (50-100 Hz)"),
        (100, 200, "Mid frequency (100-200 Hz)"),
        (200, 400, "High frequency (200-400 Hz)"),
        (400, 512, "Very high frequency (400-512 Hz)")
    ]
    
    print(f"\n=== Frequency Band Analysis ===")
    for fmin, fmax, name in frequency_bands:
        # Find frequency indices
        freq_mask = (f_signal >= fmin) & (f_signal <= fmax)
        
        if np.any(freq_mask):
            signal_power = np.mean(psd_signal[freq_mask])
            noise_power = np.mean(psd_noise[freq_mask])
            power_ratio = signal_power / noise_power if noise_power > 0 else float('inf')
            
            print(f"{name}:")
            print(f"  Signal power: {signal_power:.6e}")
            print(f"  Noise power: {noise_power:.6e}")
            print(f"  Power ratio: {power_ratio:.3f}")
            
            if power_ratio > 1.5:
                print(f"  *** SIGNIFICANT DIFFERENCE DETECTED ***")
            elif power_ratio > 1.1:
                print(f"  * Moderate difference")
            else:
                print(f"  No significant difference")
    
    # CWT analysis with different frequency ranges
    print(f"\n=== CWT Analysis by Frequency Range ===")
    
    cwt_ranges = [
        (20, 100, "Low-Mid (20-100 Hz)"),
        (100, 200, "Mid (100-200 Hz)"),
        (200, 400, "High (200-400 Hz)")
    ]
    
    for fmin, fmax, name in cwt_ranges:
        print(f"\n{name} CWT Analysis:")
        
        # Generate scales for this frequency range
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), 32)  # Fewer scales for focused analysis
        scales = sample_rate / freqs
        
        try:
            # Compute CWT for signal
            signal_coeffs, _ = pywt.cwt(signal_filtered, scales, 'morl', sampling_period=1/sample_rate)
            signal_cwt = np.abs(signal_coeffs)
            
            # Compute CWT for noise
            noise_coeffs, _ = pywt.cwt(noise_filtered, scales, 'morl', sampling_period=1/sample_rate)
            noise_cwt = np.abs(noise_coeffs)
            
            # Compare statistics
            signal_mean = signal_cwt.mean()
            noise_mean = noise_cwt.mean()
            signal_std = signal_cwt.std()
            noise_std = noise_cwt.std()
            
            mean_diff = abs(signal_mean - noise_mean)
            std_diff = abs(signal_std - noise_std)
            
            print(f"  Signal CWT: mean={signal_mean:.6e}, std={signal_std:.6e}")
            print(f"  Noise CWT: mean={noise_mean:.6e}, std={noise_std:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  Std difference: {std_diff:.6e}")
            
            # Check for significant differences
            if mean_diff > signal_mean * 0.1 or std_diff > signal_std * 0.1:
                print(f"  *** SIGNIFICANT CWT DIFFERENCES IN {name} ***")
            else:
                print(f"  No significant CWT differences")
                
        except Exception as e:
            print(f"  CWT analysis failed: {e}")
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Raw data
    t_signal = np.linspace(0, 32, len(signal_strain))
    t_noise = np.linspace(0, 32, len(noise_strain))
    
    axes[0, 0].plot(t_signal, signal_strain, 'b-', alpha=0.7)
    axes[0, 0].set_title('Signal - Raw Strain')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    
    axes[1, 0].plot(t_noise, noise_strain, 'r-', alpha=0.7)
    axes[1, 0].set_title('Noise - Raw Strain')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Strain')
    
    # Filtered data
    axes[0, 1].plot(t_signal, signal_filtered, 'b-', alpha=0.7)
    axes[0, 1].set_title('Signal - Filtered (>20 Hz)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Strain')
    
    axes[1, 1].plot(t_noise, noise_filtered, 'r-', alpha=0.7)
    axes[1, 1].set_title('Noise - Filtered (>20 Hz)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Strain')
    
    # PSD comparison
    axes[0, 2].loglog(f_signal, psd_signal, 'b-', label='Signal')
    axes[0, 2].loglog(f_noise, psd_noise, 'r-', label='Noise')
    axes[0, 2].set_title('Power Spectral Density')
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('PSD')
    axes[0, 2].legend()
    axes[0, 2].set_xlim(10, 1000)
    
    # Frequency band power comparison
    band_powers_signal = []
    band_powers_noise = []
    band_names = []
    
    for fmin, fmax, name in frequency_bands:
        freq_mask = (f_signal >= fmin) & (f_signal <= fmax)
        if np.any(freq_mask):
            signal_power = np.mean(psd_signal[freq_mask])
            noise_power = np.mean(psd_noise[freq_mask])
            band_powers_signal.append(signal_power)
            band_powers_noise.append(noise_power)
            band_names.append(f"{fmin}-{fmax}")
    
    x_pos = np.arange(len(band_names))
    axes[1, 2].bar(x_pos - 0.2, band_powers_signal, 0.4, label='Signal', alpha=0.7, color='blue')
    axes[1, 2].bar(x_pos + 0.2, band_powers_noise, 0.4, label='Noise', alpha=0.7, color='red')
    axes[1, 2].set_title('Power by Frequency Band')
    axes[1, 2].set_xlabel('Frequency Band (Hz)')
    axes[1, 2].set_ylabel('Average Power')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(band_names, rotation=45)
    axes[1, 2].legend()
    
    # CWT comparison for mid-frequency range (100-200 Hz)
    fmin, fmax = 100, 200
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), 32)
    scales = sample_rate / freqs
    
    try:
        signal_coeffs, _ = pywt.cwt(signal_filtered, scales, 'morl', sampling_period=1/sample_rate)
        noise_coeffs, _ = pywt.cwt(noise_filtered, scales, 'morl', sampling_period=1/sample_rate)
        
        signal_cwt = np.abs(signal_coeffs)
        noise_cwt = np.abs(noise_coeffs)
        
        axes[2, 0].imshow(signal_cwt, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 0].set_title(f'Signal CWT ({fmin}-{fmax} Hz)')
        axes[2, 0].set_xlabel('Time Sample')
        axes[2, 0].set_ylabel('Frequency Scale')
        
        axes[2, 1].imshow(noise_cwt, aspect='auto', origin='lower', cmap='viridis')
        axes[2, 1].set_title(f'Noise CWT ({fmin}-{fmax} Hz)')
        axes[2, 1].set_xlabel('Time Sample')
        axes[2, 1].set_ylabel('Frequency Scale')
        
        # Difference
        cwt_diff = signal_cwt - noise_cwt
        im = axes[2, 2].imshow(cwt_diff, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[2, 2].set_title(f'CWT Difference ({fmin}-{fmax} Hz)')
        axes[2, 2].set_xlabel('Time Sample')
        axes[2, 2].set_ylabel('Frequency Scale')
        plt.colorbar(im, ax=axes[2, 2])
        
    except Exception as e:
        axes[2, 0].text(0.5, 0.5, f'CWT failed: {e}', ha='center', va='center')
        axes[2, 1].text(0.5, 0.5, f'CWT failed: {e}', ha='center', va='center')
        axes[2, 2].text(0.5, 0.5, f'CWT failed: {e}', ha='center', va='center')
    
    plt.tight_layout()
    
    output_dir = Path("results/spectrogram_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "frequency_band_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved analysis plot to {output_dir / 'frequency_band_analysis.png'}")

def main():
    """Main analysis function"""
    print("=== Gravitational Wave Frequency Band Analysis ===")
    
    try:
        analyze_frequency_bands()
        
        print(f"\n=== Analysis Complete ===")
        print("This analysis examines signal characteristics in specific")
        print("frequency bands where gravitational waves are most prominent.")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
