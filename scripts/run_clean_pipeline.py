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
    preprocessor = CWTPreprocessor(config['preprocessing']['cwt'])
    
    # Process all raw files
    raw_files = list(raw_data_dir.glob("*.npz"))
    logger.info(f"Processing {len(raw_files)} raw files")
    
    for i, raw_file in enumerate(raw_files):
        try:
            # Load raw data
            data = np.load(raw_file)
            strain = data['strain']
            
            # Process with CWT
            cwt_data = preprocessor.process(strain.reshape(1, -1))[0]
            
            # Save processed data
            processed_file = processed_data_dir / f"{raw_file.stem}_cwt.npy"
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
    trainer = CWTModelTrainer(config)
    
    # Train model
    model_path = trainer.train(run_dir)
    
    logger.info(f"Model training completed. Model saved to {model_path}")
    return model_path


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
    detector = AnomalyDetector(model_path, config)
    
    # Run anomaly detection
    results = detector.detect_anomalies()
    
    # Post-process results
    post_processor = PostProcessor()
    processed_results = post_processor.process_results(results)
    
    # Calculate metrics
    metrics_evaluator = MetricsEvaluator()
    metrics = metrics_evaluator.evaluate(processed_results, run_dir)
    
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
