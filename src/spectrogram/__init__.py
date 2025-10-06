"""
Spectrogram visualization tools for gravitational wave data.

This module provides tools for creating high-quality CWT spectrograms
from raw LIGO data files.
"""

from .npz_cwt_plot import (
    parse_filename,
    load_npz_any,
    whiten_psd,
    whiten_gwpy_if_possible,
    cwt_compute,
    auto_peak_time_from_cwt,
    grace_lookup_t0,
    plot_cwt,
    main
)

__all__ = [
    'parse_filename',
    'load_npz_any', 
    'whiten_psd',
    'whiten_gwpy_if_possible',
    'cwt_compute',
    'auto_peak_time_from_cwt',
    'grace_lookup_t0',
    'plot_cwt',
    'main'
]
