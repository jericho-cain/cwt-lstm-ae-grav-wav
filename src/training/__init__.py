"""
Training module for Gravitational Wave Hunter v2.0

This module provides training capabilities for CWT-LSTM autoencoders,
including data loading, training loops, validation, and model saving.
"""

from .trainer import CWTModelTrainer

__all__ = ['CWTModelTrainer']
