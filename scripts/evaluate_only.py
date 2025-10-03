#!/usr/bin/env python3
"""
Evaluation-only script for Gravitational Wave Detection Pipeline

This script runs only the evaluation and plotting components without retraining.
It uses an existing trained model and test data.

Usage:
    python scripts/evaluate_only.py --model models/final_model.pth --config config/pipeline_clean_config.yaml
"""

import argparse
import json
import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules
from evaluation.anomaly_detector import AnomalyDetector
from evaluation.metrics import MetricsEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data (noise + signals) for evaluation.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Test data and labels
    """
    processed_data_dir = Path(config['pipeline']['data_flow']['preprocessed_data_dir'])
    
    # Load manifest to get GPS time to segment type mapping
    manifest_path = Path("data/download_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Create GPS time to segment type mapping
        gps_to_type = {}
        for entry in manifest.get('downloads', []):
            if entry.get('successful', False):
                gps_time = entry.get('gps_time')
                segment_type = entry.get('segment_type', 'noise')
                if gps_time is not None:
                    gps_to_type[gps_time] = segment_type
    else:
        logger.warning("No manifest found, assuming all segments are noise")
        gps_to_type = {}
    
    # Find all CWT files
    cwt_files = list(processed_data_dir.glob("*_cwt.npy"))
    logger.info(f"Found {len(cwt_files)} CWT data files")
    
    # Split files into train/test noise and signals
    train_noise_files = []
    test_noise_files = []
    signal_files = []
    
    for file_path in cwt_files:
        # Extract GPS time from filename (H1_<GPS>_32s_cwt.npy)
        try:
            filename_parts = file_path.stem.split('_')
            if len(filename_parts) >= 2:
                gps_time = int(filename_parts[1].split('s')[0])
                segment_type = gps_to_type.get(gps_time, 'noise')
                
                if segment_type == 'signal':
                    signal_files.append(file_path)
                else:  # noise
                    # Use a simple hash-based split for consistency
                    if hash(str(file_path)) % 5 == 0:  # 20% for test
                        test_noise_files.append(file_path)
                    else:  # 80% for train
                        train_noise_files.append(file_path)
        except (ValueError, IndexError):
            # Default to noise if parsing fails
            if hash(str(file_path)) % 5 == 0:
                test_noise_files.append(file_path)
            else:
                train_noise_files.append(file_path)
    
    logger.info(f"Split noise files: {len(train_noise_files)} train, {len(test_noise_files)} test")
    logger.info(f"Signal files for testing: {len(signal_files)}")
    
    # Load test data (test noise + signals)
    test_files = test_noise_files + signal_files
    test_data = []
    test_labels = []
    
    sampling_strategy = config['pipeline']['data_flow'].get('sampling_strategy', 'conservative')
    samples_per_file = {'conservative': 5, 'moderate': 10, 'aggressive': 20}[sampling_strategy]
    
    logger.info(f"Using {sampling_strategy} sampling: {samples_per_file} samples per file")
    
    for file_path in test_files:
        data = np.load(file_path)
        
        # Each file contains 1 sample of shape (height, width)
        if len(data.shape) == 2:
            sampled_data = data.reshape(1, data.shape[0], data.shape[1])
        else:
            sampled_data = data
            
        test_data.append(sampled_data)
        
        # Determine label based on file type
        if file_path in signal_files:
            label = 1  # signal
        else:
            label = 0  # noise
            
        test_labels.extend([label] * sampled_data.shape[0])
    
    # Combine all data
    test_data = np.concatenate(test_data, axis=0)
    test_labels = np.array(test_labels)
    
    logger.info(f"Loaded test data: {test_data.shape}, labels: {np.sum(test_labels)} signals, {np.sum(1-test_labels)} noise")
    
    return test_data, test_labels


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluation-only script for Gravitational Wave Detection")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output-dir", default="results/evaluation_only", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f) if args.config.endswith('.json') else __import__('yaml').safe_load(open(args.config, 'r'))
    
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info("Loading test data...")
    test_data, test_labels = load_test_data(config)
    
    if len(test_data) == 0:
        logger.error("No test data found!")
        return
    
    # Initialize anomaly detector
    logger.info("Initializing anomaly detector...")
    detector = AnomalyDetector(args.model, args.config)
    
    # Run anomaly detection
    logger.info("Running anomaly detection...")
    results = detector.detect_anomalies(test_data, test_labels)
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    metrics_evaluator = MetricsEvaluator()
    
    # Calculate metrics for plotting
    metrics_evaluator.calculate_metrics(
        y_true=test_labels,
        y_scores=results['reconstruction_errors'],
        y_pred=results['predictions']
    )
    
    # Create comprehensive plots
    metrics_evaluator.create_comprehensive_plots(str(output_dir))
    
    logger.info(f"Evaluation plots saved to {output_dir}")
    logger.info("Evaluation completed successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Test samples: {len(test_data)}")
    print(f"Signals: {np.sum(test_labels)}")
    print(f"Noise: {np.sum(1-test_labels)}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    print(f"ROC-AUC: {results['roc_auc']:.3f}")
    print(f"PR-AUC: {results['pr_auc']:.3f}")
    print(f"Anomalies detected: {results['num_anomalies']}")
    print(f"Anomaly rate: {results['anomaly_rate']:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()
