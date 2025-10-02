"""
Timing Validation Module for CWT Preprocessing

This module provides validation tools for testing CWT timing accuracy
using known gravitational wave events with precise timing information.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

from .cwt import CWTPreprocessor, peak_time_from_cwt

logger = logging.getLogger(__name__)


class TimingValidator:
    """
    Validates CWT timing accuracy using known gravitational wave events.
    
    This class provides methods to test CWT preprocessing timing accuracy
    against known gravitational wave events with precise timing information.
    
    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz
    target_height : int, optional
        Target height for CWT output, by default 8
    use_analytic : bool, optional
        Use analytic wavelet for better timing, by default True
        
    Attributes
    ----------
    sample_rate : int
        Sampling rate in Hz
    preprocessor : CWTPreprocessor
        CWT preprocessor instance
    known_events : Dict[str, Dict]
        Dictionary of known gravitational wave events
        
    Examples
    --------
    >>> validator = TimingValidator(sample_rate=4096)
    >>> results = validator.validate_timing(strain_data, event_name='GW150914')
    >>> print(f"Timing offset: {results['timing_offset_ms']:.2f} ms")
    """
    
    def __init__(
        self,
        sample_rate: int,
        target_height: int = 8,
        use_analytic: bool = True
    ) -> None:
        """
        Initialize timing validator.
        
        Parameters
        ----------
        sample_rate : int
            Sampling rate in Hz
        target_height : int, optional
            Target height for CWT output, by default 8
        use_analytic : bool, optional
            Use analytic wavelet for better timing, by default True
        """
        self.sample_rate = sample_rate
        self.preprocessor = CWTPreprocessor(
            sample_rate=sample_rate,
            target_height=target_height,
            use_analytic=use_analytic
        )
        
        # Known gravitational wave events with precise timing
        self.known_events = {
            'GW150914': {
                'gps_time': 1126259462.4,
                'duration': 0.2,  # seconds
                'description': 'First LIGO detection - binary black hole merger',
                'expected_offset_ms': 1.0  # Expected timing accuracy
            },
            'GW151226': {
                'gps_time': 1135136350.6,
                'duration': 1.0,
                'description': 'Second LIGO detection - binary black hole merger',
                'expected_offset_ms': 2.0
            },
            'GW170817': {
                'gps_time': 1187008882.4,
                'duration': 2.0,
                'description': 'Neutron star merger with electromagnetic counterpart',
                'expected_offset_ms': 5.0
            }
        }
    
    def validate_timing(
        self,
        strain_data: np.ndarray,
        event_name: str,
        segment_start_gps: float,
        expected_peak_time: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Validate CWT timing accuracy for a known gravitational wave event.
        
        Parameters
        ----------
        strain_data : np.ndarray
            Strain data array
        event_name : str
            Name of the gravitational wave event
        segment_start_gps : float
            GPS start time of the data segment
        expected_peak_time : float, optional
            Expected peak time relative to segment start, by default None
            
        Returns
        -------
        Dict[str, float]
            Validation results including:
            - timing_offset_ms : Timing offset in milliseconds
            - peak_time_sec : Detected peak time in seconds
            - expected_time_sec : Expected peak time in seconds
            - accuracy_score : Accuracy score (0-1, higher is better)
            
        Raises
        ------
        ValueError
            If event_name is not recognized
        """
        if event_name not in self.known_events:
            raise ValueError(f"Unknown event: {event_name}. Available events: {list(self.known_events.keys())}")
        
        event_info = self.known_events[event_name]
        
        # Process data with CWT
        cwt_data = self.preprocessor.process(strain_data.reshape(1, -1))
        
        # Find peak time
        peak_idx, peak_time_sec = self.preprocessor.find_peak_time(cwt_data[0])
        
        # Calculate expected peak time
        if expected_peak_time is None:
            # Assume peak is at the middle of the event duration
            expected_time_sec = event_info['duration'] / 2.0
        else:
            expected_time_sec = expected_peak_time
        
        # Calculate timing offset
        timing_offset_sec = abs(peak_time_sec - expected_time_sec)
        timing_offset_ms = timing_offset_sec * 1000.0
        
        # Calculate accuracy score
        expected_offset_ms = event_info['expected_offset_ms']
        accuracy_score = max(0.0, 1.0 - (timing_offset_ms / expected_offset_ms))
        
        results = {
            'timing_offset_ms': timing_offset_ms,
            'peak_time_sec': peak_time_sec,
            'expected_time_sec': expected_time_sec,
            'accuracy_score': accuracy_score,
            'event_name': event_name,
            'peak_index': peak_idx
        }
        
        logger.info(f"Timing validation for {event_name}:")
        logger.info(f"  Peak time: {peak_time_sec:.3f}s")
        logger.info(f"  Expected: {expected_time_sec:.3f}s")
        logger.info(f"  Offset: {timing_offset_ms:.2f}ms")
        logger.info(f"  Accuracy score: {accuracy_score:.3f}")
        
        return results
    
    def validate_multiple_events(
        self,
        strain_data_dict: Dict[str, np.ndarray],
        segment_start_gps_dict: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate timing for multiple gravitational wave events.
        
        Parameters
        ----------
        strain_data_dict : Dict[str, np.ndarray]
            Dictionary mapping event names to strain data
        segment_start_gps_dict : Dict[str, float]
            Dictionary mapping event names to segment start GPS times
            
        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of validation results for each event
        """
        results = {}
        
        for event_name in strain_data_dict.keys():
            if event_name in segment_start_gps_dict:
                try:
                    results[event_name] = self.validate_timing(
                        strain_data_dict[event_name],
                        event_name,
                        segment_start_gps_dict[event_name]
                    )
                except Exception as e:
                    logger.error(f"Failed to validate {event_name}: {e}")
                    results[event_name] = {'error': str(e)}
            else:
                logger.warning(f"No GPS start time provided for {event_name}")
                results[event_name] = {'error': 'No GPS start time provided'}
        
        return results
    
    def generate_timing_report(self, validation_results: Dict[str, Dict[str, float]]) -> str:
        """
        Generate a human-readable timing validation report.
        
        Parameters
        ----------
        validation_results : Dict[str, Dict[str, float]]
            Results from timing validation
            
        Returns
        -------
        str
            Formatted timing validation report
        """
        report = []
        report.append("=" * 60)
        report.append("CWT TIMING VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Sample rate: {self.sample_rate} Hz")
        report.append(f"Wavelet type: {'Analytic' if self.preprocessor.use_analytic else 'Real'}")
        report.append("")
        
        total_events = 0
        successful_validations = 0
        total_offset_ms = 0.0
        
        for event_name, results in validation_results.items():
            if 'error' in results:
                report.append(f"[ERROR] {event_name}: {results['error']}")
                continue
            
            total_events += 1
            successful_validations += 1
            
            offset_ms = results['timing_offset_ms']
            accuracy_score = results['accuracy_score']
            total_offset_ms += offset_ms
            
            # Determine status
            if offset_ms <= 1.0:
                status = "[EXCELLENT]"
            elif offset_ms <= 5.0:
                status = "[GOOD]"
            elif offset_ms <= 10.0:
                status = "[ACCEPTABLE]"
            else:
                status = "[POOR]"
            
            report.append(f"{status} {event_name}:")
            report.append(f"  Timing offset: {offset_ms:.2f} ms")
            report.append(f"  Accuracy score: {accuracy_score:.3f}")
            report.append(f"  Peak time: {results['peak_time_sec']:.3f}s")
            report.append(f"  Expected: {results['expected_time_sec']:.3f}s")
            report.append("")
        
        # Summary
        if total_events > 0:
            avg_offset_ms = total_offset_ms / total_events
            report.append("SUMMARY:")
            report.append(f"  Events tested: {total_events}")
            report.append(f"  Successful validations: {successful_validations}")
            report.append(f"  Average timing offset: {avg_offset_ms:.2f} ms")
            
            if avg_offset_ms <= 2.0:
                report.append("  Overall status: [EXCELLENT] TIMING ACCURACY")
            elif avg_offset_ms <= 5.0:
                report.append("  Overall status: [GOOD] TIMING ACCURACY")
            elif avg_offset_ms <= 10.0:
                report.append("  Overall status: [ACCEPTABLE] TIMING ACCURACY")
            else:
                report.append("  Overall status: [POOR] TIMING ACCURACY")
        
        report.append("=" * 60)
        return "\n".join(report)
