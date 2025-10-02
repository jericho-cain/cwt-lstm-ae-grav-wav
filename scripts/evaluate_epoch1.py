#!/usr/bin/env python3
"""
Evaluate the model using the epoch 1 checkpoint to avoid memory issues.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import yaml
import torch
import numpy as np
from evaluation import AnomalyDetector, PostProcessor, MetricsEvaluator
from pipeline import RunManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(config):
    """Load only test data for evaluation using manifest."""
    import json
    
    processed_dir = Path(config['pipeline']['data_flow']['preprocessed_data_dir'])
    manifest_path = Path(config['downloader']['data_directories']['manifest_file'])
    
    # Load manifest to get test file mappings
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Find test files from manifest
    test_files = []
    test_labels = []
    
    for segment_data in manifest['downloads']:
        label = segment_data.get('label', '')
        
        # Check if this is a test file
        if label.startswith('test_noise_') or label.startswith('GW'):
            # Get GPS time from segment data
            start_gps = segment_data.get('start_gps')
            detector = segment_data.get('detector', 'H1')
            
            if start_gps:
                # Construct expected filename
                filename = f"{detector}_{start_gps}_32s_cwt.npy"
                file_path = processed_dir / filename
                
                if file_path.exists():
                    test_files.append(file_path)
                    
                    # Determine label
                    if label.startswith('test_noise_'):
                        test_labels.append(0)  # Noise
                    else:
                        test_labels.append(1)  # Signal
    
    logger.info(f"Found {len(test_files)} test CWT data files")
    logger.info(f"  - Test noise: {test_labels.count(0)}")
    logger.info(f"  - Signals: {test_labels.count(1)}")
    
    if not test_files:
        raise ValueError("No test CWT data files found")
    
    # Load data
    data_list = []
    
    for file_path in test_files:
        data = np.load(file_path)
        data_list.append(data)
    
    # Stack data and remove extra dimension
    test_data = np.stack(data_list, axis=0)
    test_labels = np.array(test_labels)
    
    # Remove the extra dimension if present (should be (batch, height, width), not (batch, 1, height, width))
    if test_data.ndim == 4 and test_data.shape[1] == 1:
        test_data = test_data.squeeze(1)
    
    logger.info(f"Loaded test data: {test_data.shape}")
    logger.info(f"Test labels: {np.sum(test_labels)} signals, {len(test_labels) - np.sum(test_labels)} noise")
    
    return test_data, test_labels

def evaluate_epoch1_model(config):
    """Evaluate using the epoch 1 checkpoint."""
    logger.info("Starting evaluation with epoch 1 checkpoint...")
    
    # Load test data
    test_data, test_labels = load_test_data(config)
    
    # Initialize anomaly detector with epoch 1 checkpoint
    epoch1_path = "models/checkpoint_epoch_1.pth"
    if not os.path.exists(epoch1_path):
        raise FileNotFoundError(f"Epoch 1 checkpoint not found: {epoch1_path}")
    
    logger.info(f"Loading model from {epoch1_path}")
    
    # Load checkpoint and extract model state
    checkpoint = torch.load(epoch1_path, map_location='cpu')
    
    # Create model and load state dict
    from models import create_model
    model_config = config['model']
    model = create_model(
        model_type=model_config['model_type'],
        input_height=model_config['input_height'],
        input_width=model_config['input_width'],
        latent_dim=model_config['latent_dim'],
        lstm_hidden=model_config['lstm_hidden'],
        dropout=model_config['dropout']
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize detector with loaded model
    detector = AnomalyDetector("models/checkpoint_epoch_1.pth", "config/balanced_gw_config.yaml")
    detector.model = model
    
    # Compute reconstruction errors in smaller batches
    logger.info("Computing reconstruction errors...")
    reconstruction_errors = detector.compute_reconstruction_errors(test_data)
    
    # Set threshold
    threshold = detector.set_threshold(reconstruction_errors, test_labels)
    logger.info(f"Anomaly threshold: {threshold:.6f}")
    
    # Detect anomalies
    predictions = (reconstruction_errors > threshold).astype(int)
    
    # Calculate basic metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"Results:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    # Create results directory
    results_dir = Path("results/epoch1_evaluation")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results = {
        'reconstruction_errors': reconstruction_errors.tolist(),
        'predictions': predictions.tolist(),
        'true_labels': test_labels.tolist(),
        'threshold': float(threshold),
        'metrics': {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy)
        }
    }
    
    import json
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")
    
    # Try to create plots if we have enough data
    if len(test_data) <= 1000:  # Only for smaller datasets
        try:
            logger.info("Creating evaluation plots...")
            metrics_evaluator = MetricsEvaluator()
            
            # Calculate comprehensive metrics
            metrics_results = metrics_evaluator.calculate_metrics(
                y_true=test_labels,
                y_scores=reconstruction_errors
            )
            
            # Create plots
            metrics_evaluator.create_comprehensive_plots(str(results_dir))
            
            logger.info("Plots created successfully")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
    
    return results

def main():
    """Main function."""
    # Load config
    config_path = "config/balanced_gw_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        results = evaluate_epoch1_model(config)
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
