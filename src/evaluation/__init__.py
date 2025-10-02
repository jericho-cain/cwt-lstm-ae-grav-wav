"""
Evaluation module for Gravitational Wave Hunter v2.0

This module provides evaluation capabilities for gravitational wave detection
models, including anomaly detection, performance metrics, and visualization.
"""

from .anomaly_detector import AnomalyDetector
from .post_processor import PostProcessor

__all__ = ['AnomalyDetector', 'PostProcessor']
