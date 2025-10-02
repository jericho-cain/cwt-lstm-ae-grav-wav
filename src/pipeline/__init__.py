"""
Pipeline module for Gravitational Wave Hunter v2.0

This module provides pipeline management capabilities for the gravitational wave
detection system, including run management, configuration handling, and
reproducibility features.
"""

from .run_manager import RunManager

__all__ = ['RunManager']
