"""
Models module for Gravitational Wave Hunter v2.0

This module provides neural network models for gravitational wave detection,
including CWT-LSTM autoencoders and simplified CWT autoencoders.
"""

from .cwtlstm import (
    CWT_LSTM_Autoencoder,
    SimpleCWTAutoencoder,
    create_model,
    save_model,
    load_model
)

__all__ = [
    'CWT_LSTM_Autoencoder',
    'SimpleCWTAutoencoder', 
    'create_model',
    'save_model',
    'load_model'
]
