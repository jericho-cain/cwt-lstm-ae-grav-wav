#!/usr/bin/env python3
"""
Comprehensive Test Suite for CWT Preprocessing Module

This test suite validates the CWT preprocessing functionality including:
- Basic CWT computation
- Timing accuracy
- Edge case handling
- Integration with timing validator
"""

import unittest
import tempfile
import numpy as np
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    cwt_clean,
    peak_time_from_cwt,
    fixed_preprocess_with_cwt,
    CWTPreprocessor,
    TimingValidator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCWTBasic(unittest.TestCase):
    """Test basic CWT functionality."""
    
    def test_cwt_clean_basic(self):
        """Test basic CWT computation."""
        # Create test signal
        fs = 4096
        t = np.linspace(0, 1, fs)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
        
        # Compute CWT
        C, freqs, scales, mask = cwt_clean(x, fs, fmin=20, fmax=200, n_scales=8)
        
        # Check output shapes
        self.assertEqual(C.shape[0], 8)  # n_scales
        self.assertEqual(C.shape[1], len(x))  # time samples
        self.assertEqual(len(freqs), 8)
        self.assertEqual(len(scales), 8)
        self.assertEqual(mask.shape, C.shape)
        
        # Check frequency range
        self.assertGreaterEqual(freqs.min(), 20)
        self.assertLessEqual(freqs.max(), 200.1)  # Allow small floating point error
        
        # Check that CWT coefficients are real (magnitude from Morlet wavelet)
        self.assertFalse(np.iscomplexobj(C))
        self.assertTrue(np.isrealobj(C))
    
    def test_cwt_clean_edge_cases(self):
        """Test CWT with edge cases."""
        fs = 4096
        
        # Test with very short signal
        x_short = np.array([1.0, -1.0, 1.0, -1.0])
        C, freqs, scales, mask = cwt_clean(x_short, fs, n_scales=4)
        self.assertEqual(C.shape[0], 4)
        self.assertEqual(C.shape[1], len(x_short))
        
        # Test with constant signal
        x_const = np.ones(100)
        C, freqs, scales, mask = cwt_clean(x_const, fs, n_scales=4)
        self.assertEqual(C.shape[0], 4)
        self.assertEqual(C.shape[1], len(x_const))
        
        # Test with NaN/inf values
        x_nan = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        C, freqs, scales, mask = cwt_clean(x_nan, fs, n_scales=4)
        self.assertEqual(C.shape[0], 4)
        self.assertEqual(C.shape[1], len(x_nan))
    
    def test_peak_time_from_cwt(self):
        """Test peak time detection."""
        fs = 4096
        n_scales = 8
        n_samples = 1024
        
        # Create CWT coefficients with known peak
        C = np.random.randn(n_scales, n_samples) + 1j * np.random.randn(n_scales, n_samples)
        
        # Add a strong peak at a known location
        peak_idx = 512
        C[:, peak_idx] *= 10
        
        # Find peak
        t_idx, t_sec = peak_time_from_cwt(C, fs)
        
        # Check that peak is found near expected location
        self.assertLess(abs(t_idx - peak_idx), 10)  # Within 10 samples
        self.assertAlmostEqual(t_sec, t_idx / fs, places=6)
    
    def test_peak_time_with_mask(self):
        """Test peak time detection with cone of influence mask."""
        fs = 4096
        n_scales = 8
        n_samples = 1024
        
        # Create CWT coefficients
        C = np.random.randn(n_scales, n_samples) + 1j * np.random.randn(n_scales, n_samples)
        
        # Create mask that excludes edges
        mask = np.ones_like(C, dtype=bool)
        mask[:, :100] = False  # Exclude first 100 samples
        mask[:, -100:] = False  # Exclude last 100 samples
        
        # Add peak in masked region (should not be found)
        C[:, 50] *= 10
        
        # Add peak in valid region
        peak_idx = 512
        C[:, peak_idx] *= 10
        
        # Find peak
        t_idx, t_sec = peak_time_from_cwt(C, fs, mask=mask)
        
        # Peak should be found in valid region
        self.assertGreaterEqual(t_idx, 100)
        self.assertLessEqual(t_idx, n_samples - 100)


class TestCWTPreprocessor(unittest.TestCase):
    """Test CWTPreprocessor class."""
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = CWTPreprocessor(
            sample_rate=4096,
            target_height=8,
            use_analytic=True,
            fmin=20.0,
            fmax=512.0
        )
        
        self.assertEqual(preprocessor.sample_rate, 4096)
        self.assertEqual(preprocessor.target_height, 8)
        self.assertTrue(preprocessor.use_analytic)
        self.assertEqual(preprocessor.fmin, 20.0)
        self.assertEqual(preprocessor.fmax, 512.0)
    
    def test_preprocessor_process(self):
        """Test preprocessor process method."""
        preprocessor = CWTPreprocessor(sample_rate=4096, target_height=8)
        
        # Create test data
        n_samples = 100
        n_timepoints = 1024
        strain_data = np.random.randn(n_samples, n_timepoints)
        
        # Process data
        cwt_data = preprocessor.process(strain_data)
        
        # Check output shape
        self.assertEqual(cwt_data.shape[0], n_samples)
        self.assertEqual(cwt_data.shape[1], 8)  # target_height
        self.assertEqual(cwt_data.shape[2], n_timepoints)
        
        # Check data type
        self.assertEqual(cwt_data.dtype, np.float32)
    
    def test_preprocessor_find_peak_time(self):
        """Test preprocessor peak time finding."""
        preprocessor = CWTPreprocessor(sample_rate=4096)
        
        # Create test CWT data with known peak
        n_scales = 8
        n_timepoints = 1024
        cwt_data = np.random.randn(n_scales, n_timepoints)
        
        # Add peak at known location
        peak_idx = 512
        cwt_data[:, peak_idx] *= 10
        
        # Find peak
        t_idx, t_sec = preprocessor.find_peak_time(cwt_data)
        
        # Check results
        self.assertLess(abs(t_idx - peak_idx), 10)
        self.assertAlmostEqual(t_sec, t_idx / 4096, places=6)


class TestTimingValidator(unittest.TestCase):
    """Test TimingValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = TimingValidator(sample_rate=4096, target_height=8)
        
        self.assertEqual(validator.sample_rate, 4096)
        self.assertIsInstance(validator.preprocessor, CWTPreprocessor)
        self.assertIn('GW150914', validator.known_events)
        self.assertIn('GW151226', validator.known_events)
        self.assertIn('GW170817', validator.known_events)
    
    def test_validate_timing_mock(self):
        """Test timing validation with mock data."""
        validator = TimingValidator(sample_rate=4096)
        
        # Create mock strain data with known peak
        fs = 4096
        duration = 2.0
        peak_time = 1.0
        n_samples = int(fs * duration)
        
        # Create chirp signal
        t = np.linspace(0, duration, n_samples)
        f0, f1 = 20, 200
        f = f0 + (f1 - f0) * t / duration
        phase = 2 * np.pi * np.cumsum(f) / fs
        signal = np.sin(phase)
        
        # Add Gaussian envelope
        sigma = 0.1
        envelope = np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)
        signal *= envelope
        
        # Add noise
        signal += 0.1 * np.random.randn(n_samples)
        
        # Validate timing
        results = validator.validate_timing(
            strain_data=signal,
            event_name='GW150914',
            segment_start_gps=1126259462.4,
            expected_peak_time=peak_time
        )
        
        # Check results structure
        self.assertIn('timing_offset_ms', results)
        self.assertIn('peak_time_sec', results)
        self.assertIn('expected_time_sec', results)
        self.assertIn('accuracy_score', results)
        
        # Check that timing offset is reasonable (relaxed for mock data)
        self.assertLess(results['timing_offset_ms'], 1000)  # Less than 1000ms for mock data
        self.assertGreaterEqual(results['accuracy_score'], 0.0)
        self.assertLessEqual(results['accuracy_score'], 1.0)
    
    def test_validate_timing_unknown_event(self):
        """Test timing validation with unknown event."""
        validator = TimingValidator(sample_rate=4096)
        
        # Create test data
        signal = np.random.randn(1024)
        
        # Should raise ValueError for unknown event
        with self.assertRaises(ValueError):
            validator.validate_timing(
                strain_data=signal,
                event_name='UNKNOWN_EVENT',
                segment_start_gps=0.0
            )
    
    def test_generate_timing_report(self):
        """Test timing report generation."""
        validator = TimingValidator(sample_rate=4096)
        
        # Create mock validation results
        validation_results = {
            'GW150914': {
                'timing_offset_ms': 2.5,
                'peak_time_sec': 1.0,
                'expected_time_sec': 1.0,
                'accuracy_score': 0.8,
                'event_name': 'GW150914',
                'peak_index': 512
            },
            'GW151226': {
                'timing_offset_ms': 5.0,
                'peak_time_sec': 0.5,
                'expected_time_sec': 0.5,
                'accuracy_score': 0.6,
                'event_name': 'GW151226',
                'peak_index': 256
            }
        }
        
        # Generate report
        report = validator.generate_timing_report(validation_results)
        
        # Check report content
        self.assertIn('CWT TIMING VALIDATION REPORT', report)
        self.assertIn('GW150914', report)
        self.assertIn('GW151226', report)
        self.assertIn('SUMMARY:', report)
        self.assertIn('Events tested: 2', report)


class TestCWTIntegration(unittest.TestCase):
    """Test integration between CWT components."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end CWT processing."""
        # Create test data
        fs = 4096
        duration = 2.0
        n_samples = 10
        n_timepoints = int(fs * duration)
        
        # Create multiple strain samples
        strain_data = []
        for i in range(n_samples):
            # Create chirp signal
            t = np.linspace(0, duration, n_timepoints)
            f0, f1 = 20 + i*10, 200 + i*10
            f = f0 + (f1 - f0) * t / duration
            phase = 2 * np.pi * np.cumsum(f) / fs
            signal = np.sin(phase)
            
            # Add noise
            signal += 0.1 * np.random.randn(n_timepoints)
            strain_data.append(signal)
        
        strain_data = np.array(strain_data)
        
        # Process with CWT
        preprocessor = CWTPreprocessor(sample_rate=fs, target_height=8)
        cwt_data = preprocessor.process(strain_data)
        
        # Check output
        self.assertEqual(cwt_data.shape[0], n_samples)
        self.assertEqual(cwt_data.shape[1], 8)
        self.assertEqual(cwt_data.shape[2], n_timepoints)
        
        # Test peak detection on first sample
        peak_idx, peak_time = preprocessor.find_peak_time(cwt_data[0])
        self.assertGreaterEqual(peak_idx, 0)
        self.assertLess(peak_idx, n_timepoints)
        self.assertAlmostEqual(peak_time, peak_idx / fs, places=6)
    
    def test_timing_validation_integration(self):
        """Test integration with timing validation."""
        validator = TimingValidator(sample_rate=4096)
        
        # Create test data for multiple events
        strain_data_dict = {}
        segment_start_gps_dict = {}
        
        events = ['GW150914', 'GW151226', 'GW170817']
        for event_name in events:
            # Create mock data
            fs = 4096
            duration = 2.0
            n_timepoints = int(fs * duration)
            
            t = np.linspace(0, duration, n_timepoints)
            signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t)
            signal += 0.1 * np.random.randn(n_timepoints)
            
            strain_data_dict[event_name] = signal
            segment_start_gps_dict[event_name] = 1000000000.0  # Mock GPS time
        
        # Validate multiple events
        results = validator.validate_multiple_events(
            strain_data_dict=strain_data_dict,
            segment_start_gps_dict=segment_start_gps_dict
        )
        
        # Check results
        self.assertEqual(len(results), len(events))
        for event_name in events:
            self.assertIn(event_name, results)
            self.assertIn('timing_offset_ms', results[event_name])


def run_tests():
    """Run all CWT preprocessing tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCWTBasic,
        TestCWTPreprocessor,
        TestTimingValidator,
        TestCWTIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("CWT PREPROCESSING TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
