#!/usr/bin/env python3
"""
Test script for PostProcessor module

This script tests the post-processing functionality to ensure timing analysis
and result enhancement work correctly.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import PostProcessor


def test_post_processor():
    """Test post-processor functionality."""
    print("Testing PostProcessor...")
    
    # Create mock detection results
    detection_results = {
        'predictions': np.array([0, 1, 0, 1, 0, 1, 0, 0, 1, 0]),
        'reconstruction_errors': np.array([0.1, 0.8, 0.2, 0.9, 0.15, 0.85, 0.12, 0.18, 0.75, 0.14]),
        'threshold': 0.5,
        'num_anomalies': 4,
        'anomaly_rate': 0.4
    }
    
    # Create mock CWT data
    cwt_data = np.random.randn(10, 8, 131072)  # 10 samples, 8x131072 CWT scalograms
    
    # Initialize post-processor
    postprocessor = PostProcessor('config/pipeline_clean_config.yaml')
    
    # Test timing addition
    print("  Testing timing addition...")
    enhanced_results = postprocessor.add_timing(detection_results, cwt_data)
    
    # Check results
    assert 'detection_times' in enhanced_results
    assert 'peak_times' in enhanced_results
    assert 'confidence_scores' in enhanced_results
    assert 'timing_analysis' in enhanced_results
    
    print(f"    Detection times: {enhanced_results['detection_times']}")
    print(f"    Peak times: {enhanced_results['peak_times']}")
    print(f"    Confidence scores: {enhanced_results['confidence_scores']}")
    
    # Test pattern analysis
    print("  Testing pattern analysis...")
    pattern_analysis = postprocessor.analyze_detection_patterns(enhanced_results)
    
    assert 'pattern_analysis' in pattern_analysis
    print(f"    Total detections: {pattern_analysis['pattern_analysis']['total_detections']}")
    print(f"    Time span: {pattern_analysis['pattern_analysis']['time_span']:.3f} seconds")
    
    # Test report generation
    print("  Testing report generation...")
    report = postprocessor.generate_detection_report(enhanced_results)
    
    assert len(report) > 0
    assert "GRAVITATIONAL WAVE DETECTION REPORT" in report
    assert "Total detections: 4" in report
    
    print("    Report generated successfully")
    print("    Report preview:")
    print("    " + "\n    ".join(report.split("\n")[:10]))
    
    print("[OK] PostProcessor tests passed!")
    assert True  # Test passes if we get here


def test_empty_detections():
    """Test post-processor with no detections."""
    print("\nTesting PostProcessor with no detections...")
    
    # Create mock detection results with no anomalies
    detection_results = {
        'predictions': np.array([0, 0, 0, 0, 0]),
        'reconstruction_errors': np.array([0.1, 0.2, 0.15, 0.12, 0.18]),
        'threshold': 0.5,
        'num_anomalies': 0,
        'anomaly_rate': 0.0
    }
    
    # Create mock CWT data
    cwt_data = np.random.randn(5, 8, 131072)
    
    # Initialize post-processor
    postprocessor = PostProcessor('config/pipeline_clean_config.yaml')
    
    # Test timing addition
    enhanced_results = postprocessor.add_timing(detection_results, cwt_data)
    
    # Check results
    assert len(enhanced_results['detection_times']) == 0
    assert len(enhanced_results['peak_times']) == 0
    assert len(enhanced_results['confidence_scores']) == 0
    assert enhanced_results['timing_analysis']['total_anomalies'] == 0
    
    # Test report generation
    report = postprocessor.generate_detection_report(enhanced_results)
    assert "No gravitational wave signals detected" in report
    
    print("[OK] Empty detections test passed!")
    assert True  # Test passes if we get here


def main():
    """Main test function."""
    print("PostProcessor Test Suite")
    print("=" * 30)
    
    try:
        test_post_processor()
        test_empty_detections()
        
        print("\n" + "=" * 30)
        print("[OK] All PostProcessor tests passed!")
        print("The post-processing module is working correctly.")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
