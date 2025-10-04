#!/usr/bin/env python3
"""
Analyze the EC2 GW150914 file and compare with our current CWT preprocessing.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.cwt import cwt_clean
from preprocessing.legacy_cwt import legacy_cwt_clean

def load_ec2_file():
    """Load the EC2 GW150914 file"""
    file_path = Path("H1_1126259462_32s.npz")
    if not file_path.exists():
        print(f"Error: {file_path} not found!")
        return None
    
    data = np.load(file_path)
    strain = data['strain']
    gps_time = data['gps_time']
    
    print(f"EC2 file loaded:")
    print(f"  GPS time: {gps_time}")
    print(f"  Shape: {strain.shape}")
    print(f"  Data type: {strain.dtype}")
    print(f"  Range: {strain.min():.6e} to {strain.max():.6e}")
    print(f"  Mean: {strain.mean():.6e}, Std: {strain.std():.6e}")
    
    return strain

def process_with_both_cwt(strain_data):
    """Process with both current and legacy CWT"""
    print("\nProcessing with CURRENT CWT...")
    try:
        current_cwt, freqs, scales, coi = cwt_clean(
            strain_data, fs=4096, fmin=20.0, fmax=512.0, n_scales=64
        )
        print(f"  Current CWT shape: {current_cwt.shape}")
        print(f"  Current CWT range: {current_cwt.min():.6e} to {current_cwt.max():.6e}")
        print(f"  Current CWT mean: {current_cwt.mean():.6e}, std: {current_cwt.std():.6e}")
    except Exception as e:
        print(f"  Current CWT failed: {e}")
        current_cwt = None
    
    print("\nProcessing with LEGACY CWT...")
    try:
        legacy_cwt, freqs, scales, coi = legacy_cwt_clean(
            strain_data, fs=4096, fmin=20.0, fmax=512.0, n_scales=64
        )
        print(f"  Legacy CWT shape: {legacy_cwt.shape}")
        print(f"  Legacy CWT range: {legacy_cwt.min():.6e} to {legacy_cwt.max():.6e}")
        print(f"  Legacy CWT mean: {legacy_cwt.mean():.6e}, std: {legacy_cwt.std():.6e}")
    except Exception as e:
        print(f"  Legacy CWT failed: {e}")
        legacy_cwt = None
    
    return current_cwt, legacy_cwt

def plot_spectrograms(current_cwt, legacy_cwt):
    """Plot both CWT results for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Current CWT plots
    if current_cwt is not None:
        # Individual scale
        im1 = axes[0, 0].imshow(current_cwt, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Current CWT (Individual Scale)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Scale')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Normalized scale
        vmin, vmax = current_cwt.min(), current_cwt.max()
        im2 = axes[1, 0].imshow(current_cwt, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Current CWT (Normalized Scale)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Scale')
        plt.colorbar(im2, ax=axes[1, 0])
    
    # Legacy CWT plots
    if legacy_cwt is not None:
        # Individual scale
        im3 = axes[0, 1].imshow(legacy_cwt, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Legacy CWT (Individual Scale)')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Scale')
        plt.colorbar(im3, ax=axes[0, 1])
        
        # Normalized scale
        vmin, vmax = legacy_cwt.min(), legacy_cwt.max()
        im4 = axes[1, 1].imshow(legacy_cwt, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('Legacy CWT (Normalized Scale)')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Scale')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('ec2_gw150914_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlot saved as 'ec2_gw150914_analysis.png'")

def main():
    print("=== EC2 GW150914 File Analysis ===")
    
    # Load EC2 file
    strain_data = load_ec2_file()
    if strain_data is None:
        return
    
    # Process with both CWT methods
    current_cwt, legacy_cwt = process_with_both_cwt(strain_data)
    
    # Plot results
    plot_spectrograms(current_cwt, legacy_cwt)
    
    # Summary
    print("\n=== SUMMARY ===")
    if current_cwt is not None and legacy_cwt is not None:
        print("Both CWT methods processed successfully!")
        print(f"Shape difference: {current_cwt.shape} vs {legacy_cwt.shape}")
        print(f"Range difference: Current {current_cwt.min():.6e} to {current_cwt.max():.6e}")
        print(f"                 Legacy  {legacy_cwt.min():.6e} to {legacy_cwt.max():.6e}")
    else:
        print("One or both CWT methods failed!")

if __name__ == "__main__":
    main()
