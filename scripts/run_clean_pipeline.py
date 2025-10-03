#!/usr/bin/env python3
"""
Clean Pipeline Script for Gravitational Wave Hunter v2.0

This script runs the complete pipeline using the clean GWOSC downloader,
from data downloading to model training and evaluation.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from downloader.gwosc_downloader import CleanGWOSCDownloader
from training import CWTModelTrainer
from evaluation import AnomalyDetector, PostProcessor, MetricsEvaluator
from preprocessing import CWTPreprocessor
from pipeline import RunManager


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level, by default "INFO"
    log_file : str, optional
        Log file path, by default None
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_data(config: Dict[str, Any], config_path: str, download_signals: bool = True, 
                 download_noise: bool = True) -> Dict[str, int]:
    """
    Download data using clean GWOSC downloader.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    download_signals : bool, optional
        Whether to download signals, by default True
    download_noise : bool, optional
        Whether to download noise, by default True
        
    Returns
    -------
    Dict[str, int]
        Download results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize downloader with the same config file
    downloader = CleanGWOSCDownloader(config_path)
    
    # Download data
    if download_signals and download_noise:
        results = downloader.download_all()
    elif download_signals:
        results = downloader.download_signals()
    elif download_noise:
        results = downloader.download_noise()
    else:
        logger.warning("No data download requested")
        return {'successful': 0, 'failed': 0, 'skipped': 0}
    
    logger.info(f"Download completed: {results}")
    return results


def preprocess_data(config: Dict[str, Any]) -> None:
    """
    Preprocess raw data using CWT.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Get directories
    raw_data_dir = Path(config['downloader']['data_directories']['raw_data'])
    processed_data_dir = Path(config['downloader']['data_directories']['processed_data'])
    
    # Create processed directory
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    cwt_config = config['preprocessing']['cwt']
    preprocessor = CWTPreprocessor(
        sample_rate=cwt_config['sample_rate'],
        target_height=cwt_config['target_height'],
        use_analytic=cwt_config['use_analytic'],
        fmin=cwt_config['fmin'],
        fmax=cwt_config['fmax']
    )
    
    # Process H1 files only
    all_raw_files = list(raw_data_dir.glob("*.npz"))
    raw_files = [f for f in all_raw_files if f.name.startswith("H1_")]
    logger.info(f"Processing {len(raw_files)} H1 files (out of {len(all_raw_files)} total)")
    
    for i, raw_file in enumerate(raw_files):
        try:
            # Check if already processed
            processed_file = processed_data_dir / f"{raw_file.stem}_cwt.npy"
            if processed_file.exists():
                if (i + 1) % 100 == 0:
                    logger.info(f"Skipped {i + 1}/{len(raw_files)} files (already processed)")
                continue
            
            # Load raw data
            data = np.load(raw_file)
            strain = data['strain']
            
            # Process with CWT
            cwt_data = preprocessor.process(strain.reshape(1, -1))[0]
        
            # Save processed data
            np.save(processed_file, cwt_data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(raw_files)} files")
                
        except Exception as e:
            logger.error(f"Failed to process {raw_file}: {e}")
    
    logger.info(f"Preprocessing completed. Processed files saved to {processed_data_dir}")


def train_model(config: Dict[str, Any], run_dir: Path) -> str:
    """
    Train the CWT-LSTM autoencoder model.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    run_dir : Path
        Run directory for saving model
        
    Returns
    -------
    str
        Path to trained model
    """
    logger = logging.getLogger(__name__)
    
    # Initialize trainer
    trainer = CWTModelTrainer("config/pipeline_clean_config.yaml")
    
    # Update model save directory to run-specific path
    model_config = config['model']['save']
    model_config['model_dir'] = str(run_dir / "models")
    
    # Train model
    training_results = trainer.train()
    
    # Extract model path from config
    model_config = config['model']['save']
    model_path = Path(model_config['model_dir']) / model_config['final_model_name']
    
    logger.info(f"Model training completed. Model saved to {model_path}")
    return str(model_path)


def evaluate_model(config: Dict[str, Any], model_path: str, run_dir: Path) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    model_path : str
        Path to trained model
    run_dir : Path
        Run directory for saving results
        
    Returns
    -------
    Dict[str, Any]
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    
    # Initialize detector
    detector = AnomalyDetector(model_path, "config/pipeline_clean_config.yaml")
    
    # Load test data for evaluation
    logger.info("Loading test data for evaluation...")
    processed_dir = Path("data/processed")
    
    # Load manifest to get proper labels (same logic as trainer)
    manifest_path = Path("data/download_manifest.json")
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
    
    # Find all CWT files and split into train/test (same logic as trainer)
    cwt_files = list(processed_dir.glob("*.npy"))
    logger.info(f"Found {len(cwt_files)} CWT data files")
    
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
    
    # Split noise files into train/test (80/20 split) - same as trainer
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(noise_files)
    split_idx = int(len(noise_files) * 0.8)
    train_noise_files = noise_files[:split_idx]
    test_noise_files = noise_files[split_idx:]
    
    logger.info(f"Split noise files: {len(train_noise_files)} train, {len(test_noise_files)} test")
    logger.info(f"Signal files for testing: {len(signal_files)}")
    
    # Load and combine test data
    test_data = []
    test_labels = []
    
    # Load test noise data (label = 0)
    for file_path in test_noise_files:
        data = np.load(file_path)
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        test_data.append(data)
        test_labels.extend([0] * data.shape[0])
    
    # Load signal data (label = 1)
    for file_path in signal_files:
        data = np.load(file_path)
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        test_data.append(data)
        test_labels.extend([1] * data.shape[0])
    
    # Combine all test data
    if test_data:
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.array(test_labels)
        logger.info(f"Loaded test data: {test_data.shape}, labels: {np.sum(test_labels)} signals, {np.sum(1-test_labels)} noise")
    else:
        logger.warning("No test data found!")
        return {}
    
    # Run anomaly detection
    results = detector.detect_anomalies(test_data, test_labels)
    
    # Save results to run directory
    results_file = run_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Generate plots using MetricsEvaluator
    from evaluation.metrics import MetricsEvaluator
    
    metrics_evaluator = MetricsEvaluator()
    
    # Calculate metrics for plotting (using the same data)
    metrics_evaluator.calculate_metrics(
        y_true=test_labels,
        y_scores=results['reconstruction_errors'],
        y_pred=results['predictions']
    )
    
    # Create comprehensive plots
    plots_dir = run_dir / "results"
    metrics_evaluator.create_comprehensive_plots(str(plots_dir))
    
    logger.info(f"Evaluation plots saved to {plots_dir}")
    
    # The results already contain all metrics from detect_anomalies
    metrics = results
    
    logger.info(f"Model evaluation completed. Results saved to {run_dir}")
    return metrics


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Clean Gravitational Wave Hunter Pipeline")
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--skip-download', action='store_true', help='Skip data download')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Skip data preprocessing')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-evaluation', action='store_true', help='Skip model evaluation')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize run manager
        run_manager = RunManager()
        run_dir = run_manager.create_run()
        logger.info(f"Created run directory: {run_dir}")
        
        # Download data
        if not args.skip_download:
            logger.info("Starting data download...")
            download_results = download_data(config, args.config)
            logger.info(f"Download results: {download_results}")
        else:
            logger.info("Skipping data download")
        
        # Preprocess data
        if not args.skip_preprocessing:
            logger.info("Starting data preprocessing...")
            preprocess_data(config)
        else:
            logger.info("Skipping data preprocessing")
        
        # Train model
        if not args.skip_training:
            logger.info("Starting model training...")
            model_path = train_model(config, run_dir)
        else:
            logger.info("Skipping model training")
            model_path = None
        
        # Evaluate model
        if not args.skip_evaluation and model_path:
            logger.info("Starting model evaluation...")
            evaluation_results = evaluate_model(config, model_path, run_dir)
            logger.info(f"Evaluation results: {evaluation_results}")
        else:
            logger.info("Skipping model evaluation")
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
