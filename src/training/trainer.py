"""
Training Module for Gravitational Wave Hunter v2.0

This module provides training capabilities for CWT-LSTM autoencoders,
including data loading, training loops, validation, and model saving.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import yaml
from datetime import datetime
import json

from models import create_model, save_model
from preprocessing import CWTPreprocessor

logger = logging.getLogger(__name__)


class CWTModelTrainer:
    """
    Trainer for CWT-LSTM autoencoders.
    
    This class handles the complete training pipeline for gravitational wave
    detection models, including data loading, training loops, validation,
    and model saving with comprehensive logging and metrics tracking.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    run_manager : Optional[Any]
        Run manager instance for tracking experiments
        
    Attributes
    ----------
    config : Dict[str, Any]
        Loaded configuration dictionary
    model : nn.Module
        The neural network model
    device : torch.device
        Device for training (CPU/GPU)
    optimizer : torch.optim.Optimizer
        Optimizer for training
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler
    criterion : nn.Module
        Loss function
    train_loader : DataLoader
        Training data loader
    val_loader : Optional[DataLoader]
        Validation data loader
    best_val_loss : float
        Best validation loss achieved
    train_losses : List[float]
        Training losses per epoch
    val_losses : List[float]
        Validation losses per epoch
        
    Examples
    --------
    >>> trainer = CWTModelTrainer('config/download_config.yaml')
    >>> trainer.prepare_data()
    >>> trainer.setup_model()
    >>> trainer.train()
    """
    
    def __init__(self, config_path: str, run_manager: Optional[Any] = None) -> None:
        self.config_path = Path(config_path)
        self.run_manager = run_manager
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize attributes
        self.model: Optional[nn.Module] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        logger.info(f"Initialized trainer with device: {self.device}")
        
    def prepare_data(self) -> None:
        """
        Prepare training and validation data.
        
        Loads processed CWT data and creates data loaders for training.
        Splits data into training and validation sets based on configuration.
        """
        logger.info("Preparing training data...")
        
        # Get data configuration
        data_config = self.config['pipeline']['data_flow']
        model_config = self.config['model']
        
        # Load processed CWT data
        processed_dir = Path(data_config['preprocessed_data_dir'])
        if not processed_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
            
        # Find CWT data files
        cwt_files = list(processed_dir.glob("*.npy"))
        if not cwt_files:
            raise FileNotFoundError(f"No CWT data files found in {processed_dir}")
            
        logger.info(f"Found {len(cwt_files)} CWT data files")
        
        # Load and combine data
        cwt_data = []
        labels = []
        
        for file_path in cwt_files:
            data = np.load(file_path)
            cwt_data.append(data)
            
            # Determine label from filename
            if 'noise' in file_path.name.lower():
                labels.extend([0] * len(data))  # Noise = 0
            elif 'signal' in file_path.name.lower():
                labels.extend([1] * len(data))  # Signal = 1
            else:
                labels.extend([0] * len(data))  # Default to noise
                
        # Combine all data
        cwt_data = np.vstack(cwt_data)
        labels = np.array(labels)
        
        logger.info(f"Loaded CWT data: {cwt_data.shape}")
        logger.info(f"Labels: {np.sum(labels)} signals, {np.sum(1-labels)} noise")
        
        # Filter data based on training strategy
        if data_config['train_on_noise_only']:
            # Use only noise data for training
            noise_indices = np.where(labels == 0)[0]
            train_data = cwt_data[noise_indices]
            train_labels = labels[noise_indices]
            logger.info(f"Training on noise-only data: {len(train_data)} samples")
        else:
            # Use all data for training
            train_data = cwt_data
            train_labels = labels
            logger.info(f"Training on all data: {len(train_data)} samples")
            
        # Create validation split
        val_split = model_config['training']['validation_split']
        if val_split > 0:
            train_size = int(len(train_data) * (1 - val_split))
            val_size = len(train_data) - train_size
            
            train_data, val_data = random_split(
                TensorDataset(torch.FloatTensor(train_data).unsqueeze(1)),
                [train_size, val_size]
            )
            
            # Create validation data loader
            self.val_loader = DataLoader(
                val_data,
                batch_size=model_config['training']['batch_size'],
                shuffle=False
            )
            
            logger.info(f"Validation split: {val_size} samples")
        else:
            train_data = TensorDataset(torch.FloatTensor(train_data).unsqueeze(1))
            
        # Create training data loader
        self.train_loader = DataLoader(
            train_data,
            batch_size=model_config['training']['batch_size'],
            shuffle=True
        )
        
        logger.info(f"Training data loader created: {len(self.train_loader)} batches")
        
    def setup_model(self) -> None:
        """
        Setup model, optimizer, and loss function.
        
        Creates the model based on configuration, initializes optimizer
        and learning rate scheduler, and sets up the loss function.
        """
        logger.info("Setting up model...")
        
        model_config = self.config['model']
        training_config = model_config['training']
        
        # Create model
        self.model = create_model(
            model_type=model_config['model_type'],
            input_height=model_config['input_height'],
            input_width=model_config['input_width'],
            latent_dim=model_config['latent_dim'],
            lstm_hidden=model_config['lstm_hidden'],
            dropout=model_config['dropout']
        )
        
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        if training_config['optimizer'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate']
            )
        elif training_config['optimizer'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
            
        # Setup scheduler
        if training_config['scheduler'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        elif training_config['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs']
            )
        # 'none' scheduler means no scheduler
            
        # Setup loss function
        if training_config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        elif training_config['loss_function'] == 'l1':
            self.criterion = nn.L1Loss()
        elif training_config['loss_function'] == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {training_config['loss_function']}")
            
        logger.info(f"Model setup complete:")
        logger.info(f"  Model: {model_config['model_type']}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  Optimizer: {training_config['optimizer']}")
        logger.info(f"  Loss: {training_config['loss_function']}")
        
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns
        -------
        float
            Average training loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data,) in enumerate(self.train_loader):
            data = data.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latent = self.model(data)
            
            # Compute loss
            loss = self.criterion(reconstructed, data)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.debug(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}")
                
        return epoch_loss / num_batches if num_batches > 0 else 0.0
        
    def validate_epoch(self) -> float:
        """
        Validate for one epoch.
        
        Returns
        -------
        float
            Average validation loss for the epoch
        """
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, in self.val_loader:
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, latent = self.model(data)
                
                # Compute loss
                loss = self.criterion(reconstructed, data)
                
                epoch_loss += loss.item()
                num_batches += 1
                
        return epoch_loss / num_batches if num_batches > 0 else 0.0
        
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        epoch : int
            Current epoch number
        is_best : bool, optional
            Whether this is the best model so far, by default False
        """
        model_config = self.config['model']
        save_config = model_config['save']
        
        # Create save directory
        save_dir = Path(save_config['model_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = save_dir / save_config['best_model_name']
            save_model(self.model, best_path, {
                'epoch': epoch,
                'val_loss': self.best_val_loss,
                'config': self.config
            })
            
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns
        -------
        Dict[str, Any]
            Training results and metrics
        """
        logger.info("Starting training...")
        
        training_config = self.config['model']['training']
        num_epochs = training_config['num_epochs']
        patience = training_config['early_stopping_patience']
        
        # Training loop
        patience_counter = 0
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            epoch_start = datetime.now()
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                    
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Save checkpoint
            save_config = self.config['model']['save']
            if (epoch + 1) % save_config['save_every_n_epochs'] == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
                
            # Log progress
            epoch_time = datetime.now() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"Time: {epoch_time.total_seconds():.1f}s"
            )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Save final model
        final_path = Path(self.config['model']['save']['model_dir']) / self.config['model']['save']['final_model_name']
        save_model(self.model, final_path, {
            'epoch': epoch + 1,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'config': self.config
        })
        
        # Training results
        total_time = datetime.now() - start_time
        results = {
            'total_epochs': epoch + 1,
            'total_time_seconds': total_time.total_seconds(),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_info': self.model.get_model_info()
        }
        
        logger.info(f"Training completed in {total_time.total_seconds():.1f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        return results
