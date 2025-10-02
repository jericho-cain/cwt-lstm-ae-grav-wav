"""
Preprocessing Module for Gravitational Wave Hunter v2.0

This module provides preprocessing functionality for gravitational wave data,
including Continuous Wavelet Transform (CWT) with proper timing alignment.
"""

from .cwt import (
    cwt_clean,
    peak_time_from_cwt,
    fixed_preprocess_with_cwt,
    CWTPreprocessor
)

from .timing_validator import TimingValidator

__all__ = [
    'cwt_clean',
    'peak_time_from_cwt', 
    'fixed_preprocess_with_cwt',
    'CWTPreprocessor',
    'TimingValidator'
]
