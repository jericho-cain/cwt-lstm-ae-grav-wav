#!/usr/bin/env python3
"""
CWT Timing Validation Script for Both Detectors

This script validates the CWT preprocessing timing accuracy using real gravitational wave
data from both H1 and L1 detectors for the GW150914 event. It compares detected peak times
with expected event times to assess the effectiveness of the CWT timing fixes.

Purpose:
    - Validate CWT timing accuracy with real GWOSC data
    - Compare H1 vs L1 detector performance
    - Measure timing offsets against known event times
    - Provide performance comparison with mock data

Usage:
    python tests/validate_both_detectors.py

Requirements:
    - Real GWOSC data files in data/raw/ directory
    - H1_1126259450_32s.npz and L1_1126259450_32s.npz files
    - CWT preprocessing module

Output:
    - Timing accuracy report for both detectors
    - Performance comparison with mock data
    - Summary of timing offsets and recommendations

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import CWTPreprocessor, TimingValidator

def test_detector(detector_name: str, data_file: str, expected_peak_time: float):
    """Test CWT timing for a specific detector"""
    print(f"\n=== Testing {detector_name} Detector ===")
    
    # Load data
    data = np.load(data_file)
    strain_data = data['strain']
    print(f"Data shape: {strain_data.shape}")
    print(f"Data range: {strain_data.min():.2e} to {strain_data.max():.2e}")
    
    # Initialize CWT preprocessor
    preprocessor = CWTPreprocessor(sample_rate=4096)
    
    # Process the data (reshape to 2D for batch processing)
    strain_data_2d = strain_data.reshape(1, -1)
    cwt_data = preprocessor.process(strain_data_2d)
    print(f"CWT shape: {cwt_data.shape}")
    
    # Find peak time
    peak_idx, peak_time = preprocessor.find_peak_time(cwt_data[0])
    print(f"Detected peak time: {peak_time:.3f}s")
    print(f"Expected peak time: {expected_peak_time:.3f}s")
    print(f"Timing offset: {abs(peak_time - expected_peak_time)*1000:.1f}ms")
    
    return peak_time, abs(peak_time - expected_peak_time)*1000

def main():
    """Main function"""
    print("GW150914 CWT Timing Test - Both Detectors")
    print("=" * 50)
    
    # Test parameters
    expected_peak_time = 12.4  # Event at 12.4s in the 32s segment
    
    # Test H1 detector
    h1_peak, h1_offset = test_detector(
        "H1", 
        "data/raw/H1_1126259450_32s.npz", 
        expected_peak_time
    )
    
    # Test L1 detector
    l1_peak, l1_offset = test_detector(
        "L1", 
        "data/raw/L1_1126259450_32s.npz", 
        expected_peak_time
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"H1 Detector: Peak at {h1_peak:.3f}s, Offset: {h1_offset:.1f}ms")
    print(f"L1 Detector: Peak at {l1_peak:.3f}s, Offset: {l1_offset:.1f}ms")
    print(f"Average offset: {(h1_offset + l1_offset)/2:.1f}ms")
    
    # Compare with mock data performance
    print(f"\nMock data performance: ~100ms offset")
    print(f"Real data performance: {(h1_offset + l1_offset)/2:.1f}ms offset")
    print(f"Performance ratio: {(h1_offset + l1_offset)/2/100:.1f}x worse than mock data")

if __name__ == "__main__":
    main()
