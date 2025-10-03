"""
Anomaly Detection Module for Gravitational Wave Hunter v2.0

This module provides anomaly detection capabilities using trained autoencoders
to identify potential gravitational wave signals through reconstruction error analysis.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import yaml
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc, 
    precision_score, recall_score, f1_score, 
    accuracy_score, confusion_matrix
)

from models import load_model

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Anomaly detector for gravitational wave signals.
    
    This class uses trained autoencoders to detect anomalies (potential GW signals)
    by measuring reconstruction error. Higher reconstruction error indicates
    potential anomalies since the model was trained only on noise data.
    
    Parameters
    ----------
    model_path : str
        Path to trained model file
    config_path : str
        Path to configuration file
        
    Attributes
    ----------
    model : nn.Module
        Trained autoencoder model
    device : torch.device
        Device for inference
    threshold : float
        Reconstruction error threshold for anomaly detection
    config : Dict[str, Any]
        Loaded configuration dictionary
        
    Examples
    --------
    >>> detector = AnomalyDetector('models/best_model.pth', 'config/download_config.yaml')
    >>> results = detector.detect_anomalies(test_data)
    >>> print(f"Detected {np.sum(results['predictions'])} anomalies")
    """
    
    def __init__(self, model_path: str, config_path: str) -> None:
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model: Optional[torch.nn.Module] = None
        self.threshold: Optional[float] = None
        
        logger.info(f"Initialized anomaly detector with device: {self.device}")
        
    def load_model(self) -> None:
        """
        Load trained model from file.
        
        Loads the model architecture and weights from the specified path.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        logger.info(f"Loading model from {self.model_path}")
        
        # Load model (we need to determine the model class from config)
        model_config = self.config['model']
        model_type = model_config['type']
        
        if model_type == 'cwt_lstm':
            from models import CWT_LSTM_Autoencoder
            model_class = CWT_LSTM_Autoencoder
        elif model_type == 'simple_cwt':
            from models import SimpleCWTAutoencoder
            model_class = SimpleCWTAutoencoder
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Load model
        self.model, metadata = load_model(
            self.model_path, 
            model_class,
            latent_dim=model_config['latent_dim'],
            lstm_hidden=model_config['lstm_hidden'],
            dropout=model_config['dropout']
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model info: {self.model.get_model_info()}")
        
    def compute_reconstruction_errors(self, data: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction errors for input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input CWT data of shape (n_samples, height, width)
            
        Returns
        -------
        np.ndarray
            Reconstruction errors for each sample
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        logger.info(f"Computing reconstruction errors for {len(data)} samples")
        
        # Convert to tensor and add channel dimension
        data_tensor = torch.FloatTensor(data).unsqueeze(1).to(self.device)
        
        reconstruction_errors = []
        
        with torch.no_grad():
            for i in range(0, len(data_tensor), 8):  # Process in smaller batches to avoid memory issues
                batch = data_tensor[i:i+8]
                
                # Forward pass
                reconstructed, _ = self.model(batch)
                
                # Compute MSE reconstruction error
                mse = torch.mean((reconstructed - batch)**2, dim=(1, 2, 3))
                reconstruction_errors.extend(mse.cpu().numpy())
                
        return np.array(reconstruction_errors)
        
    def set_threshold(self, reconstruction_errors: np.ndarray, labels: Optional[np.ndarray] = None) -> float:
        """
        Set anomaly detection threshold.
        
        Parameters
        ----------
        reconstruction_errors : np.ndarray
            Reconstruction errors for all samples
        labels : np.ndarray, optional
            True labels (0=noise, 1=signal) for threshold optimization
            
        Returns
        -------
        float
            Set threshold value
        """
        anomaly_config = self.config['model']['anomaly_detection']
        
        if anomaly_config['reconstruction_error_threshold'] is not None:
            # Use fixed threshold
            self.threshold = anomaly_config['reconstruction_error_threshold']
            logger.info(f"Using fixed threshold: {self.threshold}")
        else:
            # Use percentile-based threshold
            percentile = anomaly_config['threshold_percentile']
            self.threshold = np.percentile(reconstruction_errors, percentile)
            logger.info(f"Using percentile threshold ({percentile}%): {self.threshold:.6f}")
            
        return self.threshold
        
    def detect_anomalies(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect anomalies in input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input CWT data of shape (n_samples, height, width)
        labels : np.ndarray, optional
            True labels (0=noise, 1=signal) for evaluation
            
        Returns
        -------
        Dict[str, Any]
            Detection results including predictions, errors, and metrics
        """
        if self.model is None:
            self.load_model()
            
        # Compute reconstruction errors
        reconstruction_errors = self.compute_reconstruction_errors(data)
        
        # Set threshold if not already set
        if self.threshold is None:
            self.set_threshold(reconstruction_errors, labels)
            
        # Make predictions
        predictions = (reconstruction_errors > self.threshold).astype(int)
        
        # Prepare results
        results = {
            'predictions': predictions,
            'reconstruction_errors': reconstruction_errors,
            'threshold': self.threshold,
            'num_anomalies': np.sum(predictions),
            'anomaly_rate': np.mean(predictions)
        }
        
        # Compute metrics if labels are provided
        if labels is not None:
            metrics = self.compute_metrics(labels, predictions, reconstruction_errors)
            results.update(metrics)
            
        logger.info(f"Anomaly detection complete:")
        logger.info(f"  Total samples: {len(data)}")
        logger.info(f"  Anomalies detected: {results['num_anomalies']}")
        logger.info(f"  Anomaly rate: {results['anomaly_rate']:.1%}")
        
        return results
        
    def compute_metrics(self, labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """
        Compute evaluation metrics.
        
        Parameters
        ----------
        labels : np.ndarray
            True labels (0=noise, 1=signal)
        predictions : np.ndarray
            Predicted labels (0=noise, 1=signal)
        scores : np.ndarray
            Reconstruction error scores
            
        Returns
        -------
        Dict[str, Any]
            Evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # ROC and PR curves
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            
            precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
            pr_auc = auc(recall_curve, precision_curve)
        else:
            roc_auc = 0.5
            pr_auc = 0.5
            
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm.tolist(),
            'true_positives': int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            'false_positives': int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            'true_negatives': int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            'false_negatives': int(cm[1, 0]) if cm.shape == (2, 2) else 0
        }
        
        logger.info(f"Evaluation metrics:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1-Score: {f1:.3f}")
        logger.info(f"  ROC-AUC: {roc_auc:.3f}")
        logger.info(f"  PR-AUC: {pr_auc:.3f}")
        
        return metrics
        
    def get_anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input CWT data of shape (n_samples, height, width)
            
        Returns
        -------
        np.ndarray
            Anomaly scores (reconstruction errors)
        """
        if self.model is None:
            self.load_model()
            
        return self.compute_reconstruction_errors(data)
        
    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Get anomaly probabilities for input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input CWT data of shape (n_samples, height, width)
            
        Returns
        -------
        np.ndarray
            Anomaly probabilities (0=noise, 1=signal)
        """
        scores = self.get_anomaly_scores(data)
        
        # Normalize scores to probabilities (0-1 range)
        # Higher scores = higher probability of anomaly
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            probabilities = (scores - min_score) / (max_score - min_score)
        else:
            probabilities = np.zeros_like(scores)
            
        return probabilities
