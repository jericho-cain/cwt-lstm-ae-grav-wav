"""
Training Module for Gravitational Wave Hunter v2.0

This module provides training capabilities for CWT-LSTM autoencoders,
including data loading, training loops, validation, and model saving.

Author: Jericho Cain
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
        
        # Test file lists for evaluation
        self.test_noise_files: List[Path] = []
        self.test_signal_files: List[Path] = []
        
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Setup training components
        self.setup_model()
        self.prepare_data()
        
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
        
        # Load manifest to get proper labels FIRST
        # Try to get manifest from downloader config, fallback to default
        downloader_config = self.config.get('downloader', {})
        data_dirs = downloader_config.get('data_directories', {})
        manifest_path = Path(data_dirs.get('manifest_file', 'data/download_manifest.json'))
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Create mapping from GPS time to segment type
            gps_to_type = {}
            for download in manifest['downloads']:
                if download.get('successful', False):
                    gps_time = download.get('start_gps')
                    segment_type = download.get('segment_type')
                    if gps_time and segment_type:
                        gps_to_type[gps_time] = segment_type
        else:
            logger.warning("No manifest found, defaulting all to noise")
            gps_to_type = {}
        
        # Split files into train/test sets BEFORE loading
        if data_config['train_on_noise_only']:
            # Separate noise and signal files
            noise_files = []
            signal_files = []
            
            for file_path in cwt_files:
                try:
                    filename_parts = file_path.stem.split('_')
                    if len(filename_parts) >= 2:
                        gps_time = int(filename_parts[1])
                        segment_type = gps_to_type.get(gps_time, 'noise')
                        if segment_type == 'noise':
                            noise_files.append(file_path)
                        elif segment_type == 'signal':
                            signal_files.append(file_path)
                except (ValueError, IndexError):
                    # Default to noise if parsing fails
                    noise_files.append(file_path)
            
        # Split noise files into train/test (80/20 split)
        np.random.seed(42)  # For reproducibility
        torch.manual_seed(42)  # For PyTorch reproducibility
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.shuffle(noise_files)
        split_idx = int(len(noise_files) * 0.8)
        train_noise_files = noise_files[:split_idx]
        test_noise_files = noise_files[split_idx:]
        
        logger.info(f"Split noise files: {len(train_noise_files)} train, {len(test_noise_files)} test")
        logger.info(f"Signal files for testing: {len(signal_files)}")
        
        # Use only training noise files for model training
        cwt_files = train_noise_files
        
        # Save test file lists for evaluation module
        self.test_noise_files = test_noise_files
        self.test_signal_files = signal_files
        
        # Load and combine data
        cwt_data = []
        labels = []
        
        # Get sampling strategy
        sampling_strategy = data_config.get('sampling_strategy', 'conservative')
        samples_per_file = {
            'conservative': 5,
            'moderate': 10, 
            'aggressive': 20
        }.get(sampling_strategy, 5)
        
        logger.info(f"Using {sampling_strategy} sampling: {samples_per_file} samples per file")
        
        for file_path in cwt_files:
            data = np.load(file_path)
            
            # Validate CWT data shape
            expected_shape = (8, 4096)  # Match EC2 dimensions (height, width)
            if data.shape != expected_shape:
                logger.error(f"CWT data validation failed for {file_path.name}")
                logger.error(f"Expected: {expected_shape}, Got: {data.shape}")
                logger.error("This indicates incorrect CWT preprocessing. Check preprocessing step.")
                raise ValueError(f"CWT data shape mismatch: expected {expected_shape}, got {data.shape}")
            
            # Apply sampling strategy - only take a subset of samples per file
            # Each file contains 1 sample of shape (height, width)
            # We need to add a batch dimension to make it (1, height, width)
            if len(data.shape) == 2:
                # Single sample: (height, width) -> (1, height, width)
                sampled_data = data.reshape(1, data.shape[0], data.shape[1])
            else:
                # Already batched: (samples, height, width)
                sampled_data = data
                
            # Apply sampling - take only the first samples_per_file samples
            if sampled_data.shape[0] > samples_per_file:
                sampled_data = sampled_data[:samples_per_file]
                
            cwt_data.append(sampled_data)
            
            # Extract GPS time from filename (H1_<GPS>_32s_cwt.npy)
            try:
                filename_parts = file_path.stem.split('_')
                if len(filename_parts) >= 2:
                    gps_time = int(filename_parts[1])
                    segment_type = gps_to_type.get(gps_time, 'noise')
                    label = 1 if segment_type == 'signal' else 0
                else:
                    label = 0  # Default to noise
            except (ValueError, IndexError):
                label = 0  # Default to noise if parsing fails
                
            labels.extend([label] * sampled_data.shape[0])
                
        # Combine all data properly
        # Each element in cwt_data is (1, 8, 4096) - one sample per file
        # We need to concatenate along the first axis to get (total_samples, 8, 4096)
        cwt_data = np.concatenate(cwt_data, axis=0)
        labels = np.array(labels)
        
        logger.info(f"Loaded CWT data: {cwt_data.shape}")
        logger.info(f"Labels: {np.sum(labels)} signals, {np.sum(1-labels)} noise")
        
        # Memory check - warn if data is too large
        memory_gb = cwt_data.nbytes / (1024**3)
        if memory_gb > 2.0:  # Warn if > 2GB
            logger.warning(f"Large dataset loaded: {memory_gb:.2f} GB. Consider reducing batch size or using fewer files.")
        
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
        val_split = self.config['training']['validation_split']
        if val_split > 0:
            train_size = int(len(train_data) * (1 - val_split))
            val_size = len(train_data) - train_size
            
            # Reshape data to (samples, channels, height, width) for 4D model input
            train_data_tensor = torch.FloatTensor(train_data).unsqueeze(1)  # Add channel dimension
            train_data, val_data = random_split(
                TensorDataset(train_data_tensor),
                [train_size, val_size]
            )
            
            # Create validation data loader
            self.val_loader = DataLoader(
                val_data,
                batch_size=self.config['training']['batch_size'],
                shuffle=False
            )
            
            logger.info(f"Validation split: {val_size} samples")
        else:
            # Reshape data to (samples, channels, height, width) for 4D model input
            train_data_tensor = torch.FloatTensor(train_data).unsqueeze(1)  # Add channel dimension
            train_data = TensorDataset(train_data_tensor)
            
        # Create training data loader
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.config['training']['batch_size'],
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
        training_config = self.config['training']
        
        # Create model
        self.model = create_model(
            model_type=model_config['type'],
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
                lr=float(training_config['learning_rate']),
                momentum=float(training_config.get('momentum', 0.9)),
                weight_decay=float(training_config.get('weight_decay', 1e-5))
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
        logger.info(f"  Model: {model_config['type']}")
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
        
        training_config = self.config['training']
        num_epochs = training_config['num_epochs']
        patience = training_config['early_stopping_patience']
        
        # Training loop
        patience_counter = 0
        start_time = datetime.now()
        
        # Get early stopping configuration
        monitor = training_config.get('early_stopping_monitor', 'val_loss')
        min_delta = training_config.get('early_stopping_min_delta', 1e-3)
        
        # Initialize best loss tracking based on monitor
        if monitor == 'train_loss':
            best_loss = float('inf')
        else:  # val_loss
            best_loss = self.best_val_loss
        
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
                    
            # Check for best model with minimum improvement threshold
            current_loss = train_loss if monitor == 'train_loss' else val_loss
            improvement = best_loss - current_loss
            is_best = improvement > min_delta
            
            if is_best:
                best_loss = current_loss
                # Also update best_val_loss for backward compatibility
                if monitor == 'val_loss':
                    self.best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"  Best model! Improvement: {improvement:.6f}")
            else:
                patience_counter += 1
                logger.info(f"  No significant improvement: {improvement:.6f} (threshold: {min_delta:.6f})")
                
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
