#!/usr/bin/env python3
"""
Test script to compare legacy vs current CWT preprocessing on the same files.
This will help us understand why the loss values are different.
"""

import numpy as np
import pywt
import logging
from scipy import signal
from scipy.ndimage import zoom
from pathlib import Path
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def continuous_wavelet_transform(x, fs, wavelet='morl', n_scales=64):
    """
    Legacy continuous wavelet transform function.
    Copied exactly from legacy_scripts/cwt_lstm_autoencoder_CWT_TEST.py
    """
    # Generate scales
    freqs = np.logspace(np.log10(20), np.log10(512), n_scales)
    scales = fs / freqs
    
    # Compute CWT
    coefficients, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)
    
    return np.abs(coefficients), freqs

def legacy_preprocess_with_cwt(strain_data, sample_rate, target_height=64):
    """
    Legacy preprocessing function.
    Copied exactly from legacy_scripts/cwt_lstm_autoencoder_CWT_TEST.py
    """
    cwt_data = []
    successful_samples = 0
    failed_samples = 0
    
    logger.info(f"Computing legacy CWT for {len(strain_data)} samples...")
    
    for i, strain in enumerate(strain_data):
        try:
            # Check for problematic data
            if np.any(np.isnan(strain)) or np.any(np.isinf(strain)):
                logger.warning(f"  Sample {i+1}: Contains NaN/inf values, skipping")
                failed_samples += 1
                continue
                
            if np.std(strain) < 1e-20:  # Essentially constant data
                logger.warning(f"  Sample {i+1}: No variation in data (std={np.std(strain):.2e}), skipping")
                failed_samples += 1
                continue
            
            # Apply basic preprocessing
            # High-pass filter to remove low-frequency noise
            sos = signal.butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
            filtered = signal.sosfilt(sos, strain)
            
            # Whiten the data
            whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
            
            # Compute CWT
            scalogram, freqs = continuous_wavelet_transform(whitened, sample_rate)
            
            # Check CWT result
            if np.any(np.isnan(scalogram)) or np.any(np.isinf(scalogram)):
                logger.warning(f"  Sample {i+1}: CWT produced NaN/inf values, skipping")
                failed_samples += 1
                continue
            
            # Resize to target height if needed
            if scalogram.shape[0] != target_height:
                # Simple interpolation to target size
                zoom_factor = target_height / scalogram.shape[0]
                scalogram = zoom(scalogram, (zoom_factor, 1), order=1)
            
            # Log transform and normalize (crucial for neural networks)
            log_scalogram = np.log10(scalogram + 1e-10)
            normalized = (log_scalogram - np.mean(log_scalogram)) / (np.std(log_scalogram) + 1e-10)
            normalized = normalized.astype(np.float32)  # Ensure float32
            
            # Final check
            if np.any(np.isnan(normalized)) or np.any(np.isinf(normalized)):
                logger.warning(f"  Sample {i+1}: Final normalization produced NaN/inf, skipping")
                failed_samples += 1
                continue
            
            cwt_data.append(normalized)
            successful_samples += 1
            
        except Exception as e:
            logger.warning(f"  Sample {i+1}: Error during processing: {e}")
            failed_samples += 1
            continue
    
    logger.info(f"  Legacy CWT preprocessing complete: {successful_samples} successful, {failed_samples} failed")
    
    return np.array(cwt_data)

def load_raw_data(file_path):
    """Load raw strain data from file"""
    try:
        if file_path.suffix == '.npz':
            # Load .npz file - extract the data array
            data_dict = np.load(file_path)
            # Get the first (and likely only) array from the .npz file
            data = data_dict[list(data_dict.keys())[0]]
        else:
            data = np.load(file_path)
        
        logger.info(f"Loaded raw data: {data.shape}, range={data.min():.6e} to {data.max():.6e}")
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def analyze_cwt_differences(legacy_cwt, current_cwt, file_type):
    """Analyze differences between legacy and current CWT preprocessing"""
    
    print(f"\n=== {file_type.upper()} CWT ANALYSIS ===")
    print(f"Legacy CWT shape: {legacy_cwt.shape}")
    print(f"Current CWT shape: {current_cwt.shape}")
    
    print(f"\nLegacy CWT stats:")
    print(f"  Range: {legacy_cwt.min():.6e} to {legacy_cwt.max():.6e}")
    print(f"  Mean: {legacy_cwt.mean():.6e}")
    print(f"  Std: {legacy_cwt.std():.6e}")
    print(f"  Non-zero values: {np.count_nonzero(legacy_cwt)} / {legacy_cwt.size}")
    
    print(f"\nCurrent CWT stats:")
    print(f"  Range: {current_cwt.min():.6e} to {current_cwt.max():.6e}")
    print(f"  Mean: {current_cwt.mean():.6e}")
    print(f"  Std: {current_cwt.std():.6e}")
    print(f"  Non-zero values: {np.count_nonzero(current_cwt)} / {current_cwt.size}")
    
    # Check if shapes match for comparison
    if legacy_cwt.shape == current_cwt.shape:
        diff = np.abs(legacy_cwt - current_cwt)
        print(f"\nDirect comparison (same shape):")
        print(f"  Max absolute difference: {diff.max():.6e}")
        print(f"  Mean absolute difference: {diff.mean():.6e}")
        print(f"  Relative difference: {diff.mean() / current_cwt.mean():.6e}")
    else:
        print(f"\nShapes don't match - cannot do direct comparison")

def main():
    """Main comparison function"""
    
    # Find one noise file and one signal file from raw data
    raw_dir = Path("data/raw")
    
    # Get specific files based on manifest
    # Signal file: H1_1164686041_32s.npz (event: 161202-v1)
    # Noise file: H1_1126124017_32s.npz (confirmed noise from manifest)
    signal_file = raw_dir / "H1_1164686041_32s.npz"
    noise_file = raw_dir / "H1_1126124017_32s.npz"
    
    if not signal_file.exists():
        logger.error(f"Signal file not found: {signal_file}")
        return
    
    if not noise_file.exists():
        logger.error(f"Noise file not found: {noise_file}")
        return
    
    logger.info(f"Testing noise file: {noise_file}")
    logger.info(f"Testing signal file: {signal_file}")
    
    # Load raw data
    noise_data = load_raw_data(noise_file)
    signal_data = load_raw_data(signal_file)
    
    if noise_data is None or signal_data is None:
        logger.error("Failed to load raw data")
        return
    
    # Apply legacy CWT preprocessing
    logger.info("\n=== APPLYING LEGACY CWT PREPROCESSING ===")
    legacy_noise_cwt = legacy_preprocess_with_cwt([noise_data], sample_rate=4096)
    legacy_signal_cwt = legacy_preprocess_with_cwt([signal_data], sample_rate=4096)
    
    # Load current CWT preprocessing results
    processed_dir = Path("data/processed")
    
    # Find corresponding processed files
    # Convert .npz filename to expected processed filename pattern
    noise_processed_name = noise_file.stem.replace('.npz', '') + '_cwt.npy'
    signal_processed_name = signal_file.stem.replace('.npz', '') + '_cwt.npy'
    
    noise_processed_files = list(processed_dir.glob(noise_processed_name))
    signal_processed_files = list(processed_dir.glob(signal_processed_name))
    
    if not noise_processed_files or not signal_processed_files:
        logger.error("Could not find corresponding processed files")
        return
    
    current_noise_cwt = np.load(noise_processed_files[0])
    current_signal_cwt = np.load(signal_processed_files[0])
    
    logger.info(f"Loaded current noise CWT: {current_noise_cwt.shape}")
    logger.info(f"Loaded current signal CWT: {current_signal_cwt.shape}")
    
    # Analyze differences
    analyze_cwt_differences(legacy_noise_cwt[0], current_noise_cwt, "noise")
    analyze_cwt_differences(legacy_signal_cwt[0], current_signal_cwt, "signal")
    
    # Save legacy CWT results to separate directory for comparison
    legacy_dir = Path("data/legacy_cwt_test")
    legacy_dir.mkdir(exist_ok=True)
    
    # Save legacy results
    np.save(legacy_dir / f"legacy_{noise_file.stem}_cwt.npy", legacy_noise_cwt[0])
    np.save(legacy_dir / f"legacy_{signal_file.stem}_cwt.npy", legacy_signal_cwt[0])
    logger.info(f"Saved legacy CWT results to {legacy_dir}")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Noise comparison
    im1 = axes[0, 0].imshow(legacy_noise_cwt[0], aspect='auto', origin='lower')
    axes[0, 0].set_title('Legacy Noise CWT')
    axes[0, 0].set_ylabel('Scale')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(current_noise_cwt, aspect='auto', origin='lower')
    axes[0, 1].set_title('Current Noise CWT')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Signal comparison
    im3 = axes[1, 0].imshow(legacy_signal_cwt[0], aspect='auto', origin='lower')
    axes[1, 0].set_title('Legacy Signal CWT')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Scale')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(current_signal_cwt, aspect='auto', origin='lower')
    axes[1, 1].set_title('Current Signal CWT')
    axes[1, 1].set_xlabel('Time')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('cwt_comparison_legacy_vs_current.png', dpi=150, bbox_inches='tight')
    logger.info("Saved comparison plot: cwt_comparison_legacy_vs_current.png")
    
    print(f"\n=== SUMMARY ===")
    print(f"Legacy CWT applies log10 transform + normalization")
    print(f"Current CWT uses raw magnitude scalogram")
    print(f"This fundamental difference in data preprocessing")
    print(f"could explain why the model loss values are different!")

if __name__ == "__main__":
    main()
