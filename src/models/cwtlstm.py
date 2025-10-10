"""
CWT-LSTM Autoencoder for Gravitational Wave Detection

This module implements a clean, production-ready LSTM autoencoder that operates on
Continuous Wavelet Transform (CWT) scalograms for unsupervised anomaly detection
in gravitational wave data.

The model learns to reconstruct normal noise patterns and identifies anomalies
(potential GW signals) through high reconstruction error.

Author: Jericho Cain
Date: October 2, 2025
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class CWT_LSTM_Autoencoder(nn.Module):
    """
    LSTM Autoencoder for gravitational wave detection using CWT scalograms.
    
    A hybrid neural network architecture that combines 2D convolutional layers
    for spatial feature extraction with LSTM layers for temporal modeling.
    Designed for unsupervised anomaly detection in gravitational wave data.
    
    The model learns to reconstruct normal noise patterns and identifies
    anomalies (potential GW signals) through high reconstruction error.
    
    Parameters
    ----------
    input_height : int
        Height of input CWT scalograms
    input_width : int
        Width of input CWT scalograms
    latent_dim : int, optional
        Dimension of the latent space representation, by default 32
    lstm_hidden : int, optional
        Hidden size for LSTM layers, by default 64
    dropout : float, optional
        Dropout rate for regularization, by default 0.1
        
    Attributes
    ----------
    input_height : int
        Height of input CWT scalograms
    input_width : int
        Width of input CWT scalograms
    latent_dim : int
        Dimension of the latent space representation
    spatial_encoder : nn.Sequential
        CNN encoder for spatial feature extraction
    temporal_encoder : nn.Linear
        Linear encoder for temporal sequence modeling
    to_latent : nn.Linear
        Linear layer mapping to latent space
    from_latent : nn.Linear
        Linear layer mapping from latent space
    spatial_decoder : nn.Sequential
        CNN decoder for spatial feature reconstruction
        
    Examples
    --------
    >>> model = CWT_LSTM_Autoencoder(input_height=8, input_width=131072, latent_dim=32)
    >>> x = torch.randn(1, 1, 8, 131072)  # Batch of CWT scalograms
    >>> reconstructed, latent = model(x)
    >>> print(f"Reconstructed shape: {reconstructed.shape}")
    >>> print(f"Latent shape: {latent.shape}")
    """
    
    def __init__(
        self, 
        input_height: int, 
        input_width: int, 
        latent_dim: int = 32, 
        lstm_hidden: int = 64,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        
        # Encoder: 2D CNN to extract spatial features - memory efficient
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Immediate downsampling to avoid large intermediates
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Final small spatial dimensions
        )
        
        # Simple linear encoder instead of LSTM for memory efficiency
        self.temporal_encoder = nn.Linear(
            32 * 4 * 4,  # Flattened spatial features (32 channels * 4*4 spatial)
            lstm_hidden // 2  # Reduce hidden size by half
        )
        
        # Latent space
        self.to_latent = nn.Linear(lstm_hidden // 2, latent_dim)
        
        # Decoder - simplified to match encoder
        self.from_latent = nn.Linear(latent_dim, lstm_hidden // 2)
        
        # Spatial decoder - ultra-compact to avoid memory explosion
        self.spatial_decoder = nn.Sequential(
            nn.Linear(lstm_hidden // 2, 16 * 4 * 4),  # Even smaller: 16x4x4 instead of 32x8x8
            nn.ReLU(),
            nn.Unflatten(1, (16, 4, 4)),  # Reshape to tiny fixed spatial dimensions
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((input_height, input_width)),  # Upsample to target dimensions
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input scalogram to latent representation.
        
        Processes the input through spatial and temporal encoders to produce
        a compact latent representation of the time-frequency patterns.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width)
            
        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
        """
        batch_size, channels, height, width = x.size()
        
        # Spatial encoding - process entire scalogram at once
        spatial_features = self.spatial_encoder(x)  # (batch, 32, 4, 4)
        
        # Flatten spatial features for temporal modeling
        spatial_flat = spatial_features.view(batch_size, -1)  # (batch, 32*4*4)
        
        # Temporal encoding
        temporal_out = self.temporal_encoder(spatial_flat)  # (batch, lstm_hidden//2)
        
        # Direct to latent
        latent = self.to_latent(temporal_out)  # (batch, latent_dim)
        
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation back to scalogram.
        
        Reconstructs the original CWT scalogram from the latent representation
        using a simple linear decoder to match the encoder architecture.
        
        Parameters
        ----------
        latent : torch.Tensor
            Latent representation of shape (batch_size, latent_dim)
            
        Returns
        -------
        torch.Tensor
            Reconstructed scalogram of shape (batch_size, 1, height, width)
        """
        batch_size = latent.size(0)
        
        # Simple linear decoding to match the encoder
        decoded_features = self.from_latent(latent)  # (batch, lstm_hidden//2)
        
        # Direct spatial decoding - no unnecessary sequence creation
        reconstructed = self.spatial_decoder(decoded_features)  # (batch, 1, height, width)
        
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Encodes the input to latent space and decodes back to reconstruction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed scalogram
            - latent: Latent representation
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_height': self.input_height,
            'input_width': self.input_width,
            'latent_dim': self.latent_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': 'CWT-LSTM Autoencoder'
        }


class SimpleCWTAutoencoder(nn.Module):
    """
    Simplified CWT Autoencoder for gravitational wave detection.
    
    A streamlined version of the CWT-LSTM autoencoder that uses only
    convolutional layers for easier training and understanding. This model
    is more stable to train and provides a good baseline for comparison.
    
    Parameters
    ----------
    height : int
        Height of input CWT scalograms
    width : int
        Width of input CWT scalograms
    latent_dim : int, optional
        Dimension of the latent space, by default 64
    dropout : float, optional
        Dropout rate for regularization, by default 0.1
        
    Attributes
    ----------
    height : int
        Height of input CWT scalograms
    width : int
        Width of input CWT scalograms
    encoder : nn.Sequential
        Convolutional encoder network
    decoder : nn.Sequential
        Convolutional decoder network
        
    Examples
    --------
    >>> model = SimpleCWTAutoencoder(height=8, width=131072, latent_dim=64)
    >>> x = torch.randn(1, 1, 8, 131072)  # Batch of CWT scalograms
    >>> reconstructed, latent = model(x)
    >>> print(f"Reconstructed shape: {reconstructed.shape}")
    >>> print(f"Latent shape: {latent.shape}")
    """
    
    def __init__(
        self, 
        height: int, 
        width: int, 
        latent_dim: int = 64,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((8, 8)),  # Fixed size output
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        
        self.height = height
        self.width = width
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the simplified autoencoder.
        
        Encodes the input to latent space and decodes back to reconstruction.
        
        Parameters
        ----------
        x : torch.Tensor
            Input CWT scalogram tensor of shape (batch_size, 1, height, width)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - reconstructed: Reconstructed scalogram
            - latent: Latent representation
        """
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Resize to original dimensions if needed
        if reconstructed.shape[-2:] != (self.height, self.width):
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=(self.height, self.width), mode='bilinear', align_corners=False
            )
        
        return reconstructed, latent
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model architecture information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing model architecture details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_height': self.height,
            'input_width': self.width,
            'latent_dim': 64,  # Fixed for this model
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': 'Simple CWT Autoencoder'
        }


def create_model(
    model_type: str,
    input_height: int,
    input_width: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create CWT autoencoder models.
    
    Parameters
    ----------
    model_type : str
        Type of model to create ('cwt_lstm' or 'simple_cwt')
    input_height : int
        Height of input CWT scalograms
    input_width : int
        Width of input CWT scalograms
    **kwargs
        Additional keyword arguments passed to model constructor
        
    Returns
    -------
    nn.Module
        Initialized model instance
        
    Raises
    ------
    ValueError
        If model_type is not supported
        
    Examples
    --------
    >>> model = create_model('cwt_lstm', input_height=8, input_width=131072, latent_dim=32)
    >>> print(model.get_model_info())
    """
    if model_type.lower() == 'cwt_lstm':
        return CWT_LSTM_Autoencoder(
            input_height=input_height,
            input_width=input_width,
            **kwargs
        )
    elif model_type.lower() == 'simple_cwt':
        return SimpleCWTAutoencoder(
            height=input_height,
            width=input_width,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'cwt_lstm', 'simple_cwt'")


def save_model(model: nn.Module, save_path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save model with metadata.
    
    Parameters
    ----------
    model : nn.Module
        Model to save
    save_path : Path
        Path to save the model
    metadata : Dict[str, Any], optional
        Additional metadata to save with the model
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save data
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_info': model.get_model_info(),
        'metadata': metadata or {}
    }
    
    # Save model
    torch.save(save_data, save_path)
    logger.info(f"Model saved to {save_path}")


def load_model(load_path: Path, model_class: nn.Module, **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model with metadata.
    
    Parameters
    ----------
    load_path : Path
        Path to load the model from
    model_class : nn.Module
        Model class to instantiate
    **kwargs
        Additional keyword arguments for model constructor
        
    Returns
    -------
    Tuple[nn.Module, Dict[str, Any]]
        Loaded model and metadata
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Model file not found: {load_path}")
    
    # Load model data
    save_data = torch.load(load_path, map_location='cpu')
    
    # Create model instance
    model_info = save_data['model_info']
    
    # Handle different model types
    if model_class == SimpleCWTAutoencoder:
        model = model_class(
            height=model_info['input_height'],
            width=model_info['input_width'],
            **kwargs
        )
    else:
        model = model_class(
            input_height=model_info['input_height'],
            input_width=model_info['input_width'],
            **kwargs
        )
    
    # Load state dict
    model.load_state_dict(save_data['model_state_dict'])
    
    logger.info(f"Model loaded from {load_path}")
    
    return model, save_data['metadata']
