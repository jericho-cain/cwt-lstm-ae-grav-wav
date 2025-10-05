"""
CWT-LSTM Autoencoder for Gravitational Wave Detection

A state-of-the-art unsupervised anomaly detection system for identifying
gravitational wave signals in LIGO detector noise using Continuous Wavelet
Transform (CWT) and Long Short-Term Memory (LSTM) autoencoders.

Author: Gravitational Wave Hunter v2.0
Version: 1.0.0
Date: October 2025
"""

__version__ = "1.0.0"
__author__ = "Gravitational Wave Hunter v2.0"
__email__ = "contact@gravitational-wave-detection.org"
__description__ = "Unsupervised Gravitational Wave Detection using CWT-LSTM Autoencoders"
__license__ = "MIT"

# Core modules
from .src.models import CWTLSTMAutoencoder
from .src.preprocessing import CWTPreprocessor
from .src.evaluation import AnomalyDetector, MetricsEvaluator
from .src.training import Trainer
from .src.pipeline import RunManager

__all__ = [
    "CWTLSTMAutoencoder",
    "CWTPreprocessor", 
    "AnomalyDetector",
    "MetricsEvaluator",
    "Trainer",
    "RunManager",
    "__version__",
    "__author__",
    "__description__"
]
