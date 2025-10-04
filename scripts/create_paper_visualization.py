#!/usr/bin/env python3
"""
Create CWT visualizations matching the paper's approach.
Based on create_cwt_comparison.py from the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy import signal
from pathlib import Path

def load_data_file(file_path):
    """Load strain data from npz file"""
    data = np.load(file_path)
    
    # Handle different file formats
    if 'strain' in data:
        strain = data['strain']
    else:
        # If strain is the only array, use it
        strain = data[list(data.keys())[0]]
    
    return strain

def apply_paper_cwt(strain_data, sample_rate=4096):
    """
    Apply CWT exactly as done in the paper's visualization script.
    Returns raw magnitude without log transform or normalization.
    """
    # Use the same scales as in the paper
    scales = np.logspace(0, 2, 64)  # 64 scales from 1 to 100
    
    # Perform CWT using Morlet wavelet
    coefficients, _ = pywt.cwt(strain_data, scales, 'morl', sampling_period=1/sample_rate)
    
    # Convert scales to frequencies (as done in paper)
    frequencies = pywt.scale2frequency('morl', scales) * sample_rate
    
    # Return raw magnitude (no log transform, no normalization)
    scalogram = np.abs(coefficients)
    
    return scalogram, frequencies

def create_frequency_band_plots(scalogram, frequencies, title_prefix="", save_name="cwt_analysis.png"):
    """
    Create frequency band comparison plots matching the paper's style.
    """
    # Define frequency bands as used in the paper
    freq_ranges = [
        (20, 50),    # Low frequency band
        (50, 100),   # Mid frequency band  
        (100, 200)   # High frequency band
    ]
    
    # Find indices for each frequency band
    band_indices = []
    for low_freq, high_freq in freq_ranges:
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        band_indices.append(np.where(mask)[0])
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each frequency band
    for i, (band_idx, freq_range) in enumerate(zip(band_indices, freq_ranges)):
        ax = axes[i]
        
        # Extract frequency band data
        band_data = scalogram[band_idx, :]
        band_freqs = frequencies[band_idx]
        
        # Calculate time points (assuming 4-second duration as in paper)
        time_points = np.linspace(0, 4, band_data.shape[1])
        
        # Create the plot exactly as in the paper
        im = ax.imshow(band_data, aspect='auto', origin='lower',
                       extent=[0, 4, band_freqs[0], band_freqs[-1]],
                       cmap='viridis')
        
        ax.set_title(f'{title_prefix} {freq_range[0]}-{freq_range[1]} Hz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='|CWT Coefficient|')
    
    # Add overall title
    fig.suptitle(f'Frequency Band Analysis: {title_prefix}', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    
    print(f"Plot saved as '{save_name}'")

def analyze_ec2_gw150914():
    """Analyze the EC2 GW150914 file using paper's approach"""
    print("=== Analyzing EC2 GW150914 using Paper's CWT Approach ===")
    
    # Load the EC2 file
    file_path = Path("H1_1126259462_32s.npz")
    if not file_path.exists():
        print(f"Error: {file_path} not found!")
        return
    
    strain_data = load_data_file(file_path)
    print(f"Loaded strain data: shape={strain_data.shape}, range={strain_data.min():.6e} to {strain_data.max():.6e}")
    
    # Apply paper's CWT approach
    scalogram, frequencies = apply_paper_cwt(strain_data)
    print(f"CWT computed: shape={scalogram.shape}, range={scalogram.min():.6e} to {scalogram.max():.6e}")
    print(f"Frequency range: {frequencies.min():.2f} to {frequencies.max():.2f} Hz")
    
    # Create frequency band plots
    create_frequency_band_plots(scalogram, frequencies, 
                               title_prefix="GW150914 Signal", 
                               save_name="ec2_gw150914_paper_style.png")
    
    return scalogram, frequencies

def compare_with_our_data():
    """Compare EC2 data with our current data using paper's approach"""
    print("\n=== Comparing EC2 vs Our Data using Paper's Approach ===")
    
    # Load our data
    our_file = Path("data/raw/H1_1164686041_32s.npz")
    if not our_file.exists():
        print(f"Our data file not found: {our_file}")
        return
    
    our_strain = load_data_file(our_file)
    print(f"Our data: shape={our_strain.shape}, range={our_strain.min():.6e} to {our_strain.max():.6e}")
    
    # Apply paper's CWT to our data
    our_scalogram, frequencies = apply_paper_cwt(our_strain)
    print(f"Our CWT: shape={our_scalogram.shape}, range={our_scalogram.min():.6e} to {our_scalogram.max():.6e}")
    
    # Create comparison plots
    create_frequency_band_plots(our_scalogram, frequencies,
                               title_prefix="Our Signal Data",
                               save_name="our_signal_paper_style.png")

def main():
    """Main function"""
    print("Creating CWT visualizations using paper's approach...")
    
    # Analyze EC2 GW150914
    analyze_ec2_gw150914()
    
    # Compare with our data
    compare_with_our_data()
    
    print("\nAnalysis complete! Check the generated PNG files.")

if __name__ == "__main__":
    main()
