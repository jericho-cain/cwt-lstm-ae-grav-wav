"""
CWT-LSTM Autoencoder for Gravitational Wave Detection

A state-of-the-art unsupervised anomaly detection system for identifying
gravitational wave signals in LIGO detector noise using Continuous Wavelet
Transform (CWT) and Long Short-Term Memory (LSTM) autoencoders.

Author: Jericho Cain
Version: 1.0.0
Date: October 2025
"""

__version__ = "1.0.0"
__author__ = "Jericho Cain"
__email__ = "contact@gravitational-wave-detection.org"
__description__ = "Unsupervised Gravitational Wave Detection using CWT-LSTM Autoencoders"
__license__ = "MIT"

# Core modules - imported on demand to avoid test import issues
__all__ = [
    "__version__",
    "__author__",
    "__description__"
]

# Optional imports for when the package is properly installed
try:
    from .src.models import CWTLSTMAutoencoder
    from .src.preprocessing import CWTPreprocessor
    from .src.evaluation import AnomalyDetector, MetricsEvaluator
    from .src.training import Trainer
    from .src.pipeline import RunManager
    
    __all__.extend([
        "CWTLSTMAutoencoder",
        "CWTPreprocessor", 
        "AnomalyDetector",
        "MetricsEvaluator",
        "Trainer",
        "RunManager"
    ])
except ImportError:
    # Allow the package to be imported even if modules aren't available
    # This is useful for testing and development
    pass
