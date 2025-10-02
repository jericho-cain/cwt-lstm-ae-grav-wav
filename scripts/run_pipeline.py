#!/usr/bin/env python3
"""
End-to-End Pipeline Script for Gravitational Wave Hunter v2.0

This script runs the complete pipeline from data preprocessing to model training
and evaluation, with comprehensive logging and run management.

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
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def preprocess_data(config: Dict[str, Any], run_manager: RunManager) -> None:
    """
    Preprocess raw data using CWT.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    run_manager : RunManager
        Run manager instance
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    # Get preprocessing configuration
    preprocessing_config = config['preprocessing']['cwt']
    
    # Initialize CWT preprocessor
    preprocessor = CWTPreprocessor(
        sample_rate=preprocessing_config['sample_rate'],
        target_height=preprocessing_config['target_height'],
        use_analytic=preprocessing_config['use_analytic'],
        fmin=preprocessing_config['fmin'],
        fmax=preprocessing_config['fmax']
    )
    
    # Get data directories
    raw_data_dir = Path(config['downloader']['data_directories']['raw_data'])
    processed_data_dir = Path(config['downloader']['data_directories']['processed_data'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Find raw data files
    raw_files = list(raw_data_dir.glob("*.npz"))
    if not raw_files:
        logger.warning(f"No raw data files found in {raw_data_dir}")
        return
        
    logger.info(f"Found {len(raw_files)} raw data files")
    
    # Process each file
    for file_path in raw_files:
        logger.info(f"Processing {file_path.name}...")
        
        try:
            # Load strain data
            data = np.load(file_path)
            strain_data = data['strain']
            
            # Reshape for batch processing
            if strain_data.ndim == 1:
                strain_data = strain_data.reshape(1, -1)
                
            # Process with CWT
            cwt_data = preprocessor.process(strain_data)
            
            # Save processed data
            output_file = processed_data_dir / f"{file_path.stem}_cwt.npy"
            np.save(output_file, cwt_data)
            
            logger.info(f"Saved processed data: {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
            
    logger.info("Data preprocessing completed")


def train_model(config: Dict[str, Any], run_manager: RunManager) -> Dict[str, Any]:
    """
    Train the CWT-LSTM autoencoder model.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    run_manager : RunManager
        Run manager instance
        
    Returns
    -------
    Dict[str, Any]
        Training results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = CWTModelTrainer(
        config_path=run_manager.config_path,
        run_manager=run_manager
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Setup model
    trainer.setup_model()
    
    # Add model info to run metadata
    run_manager.add_model_info(trainer.model.get_model_info())
    
    # Train model
    results = trainer.train()
    
    # Add training results to run metadata
    run_manager.add_training_results(results)
    
    logger.info("Model training completed")
    return results


def evaluate_model(config: Dict[str, Any], run_manager: RunManager) -> Dict[str, Any]:
    """
    Evaluate the trained model for anomaly detection.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    run_manager : RunManager
        Run manager instance
        
    Returns
    -------
    Dict[str, Any]
        Evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting model evaluation...")
    
    # Get model path
    model_config = config['model']
    model_path = Path(model_config['save']['model_dir']) / model_config['save']['best_model_name']
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return {}
        
    # Initialize anomaly detector
    detector = AnomalyDetector(
        model_path=str(model_path),
        config_path=str(run_manager.config_path)
    )
    
    # Load test data
    processed_data_dir = Path(config['downloader']['data_directories']['processed_data'])
    test_files = list(processed_data_dir.glob("*_cwt.npy"))
    
    if not test_files:
        logger.warning("No test data found for evaluation")
        return {}
        
    # Load and combine test data
    test_data = []
    test_labels = []
    
    for file_path in test_files:
        data = np.load(file_path)
        test_data.append(data)
        
        # Determine labels from filename
        if 'noise' in file_path.name.lower():
            test_labels.extend([0] * len(data))
        elif 'signal' in file_path.name.lower():
            test_labels.extend([1] * len(data))
        else:
            test_labels.extend([0] * len(data))
            
    test_data = np.vstack(test_data)
    test_labels = np.array(test_labels)
    
    logger.info(f"Loaded test data: {test_data.shape}")
    logger.info(f"Test labels: {np.sum(test_labels)} signals, {np.sum(1-test_labels)} noise")
    
    # Detect anomalies
    results = detector.detect_anomalies(test_data, test_labels)
    
    # Post-process results to add timing information
    postprocessor = PostProcessor(str(run_manager.config_path))
    enhanced_results = postprocessor.add_timing(results, test_data)
    
    # Analyze detection patterns
    pattern_analysis = postprocessor.analyze_detection_patterns(enhanced_results)
    enhanced_results.update(pattern_analysis)
    
    # Generate detection report
    detection_report = postprocessor.generate_detection_report(enhanced_results)
    
    # Save detection report
    if run_manager.current_run_dir:
        report_file = run_manager.current_run_dir / "results" / "detection_report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(detection_report)
        logger.info(f"Detection report saved to {report_file}")
    
    # Calculate comprehensive metrics and create plots
    logger.info("Calculating comprehensive metrics and creating plots...")
    metrics_evaluator = MetricsEvaluator()
    
    # Calculate metrics using reconstruction errors and true labels
    metrics_results = metrics_evaluator.calculate_metrics(
        y_true=test_labels,
        y_scores=results['reconstruction_errors']
    )
    
    # Create comprehensive plots
    metrics_evaluator.create_comprehensive_plots(str(run_manager.current_run_dir / "results"))
    
    # Get summary metrics
    summary_metrics = metrics_evaluator.get_summary_metrics()
    logger.info(f"Metrics summary: {summary_metrics}")
    
    # Add metrics to results
    enhanced_results['metrics'] = summary_metrics
    enhanced_results['metrics_detailed'] = metrics_results
    
    # Add evaluation results to run metadata
    run_manager.add_evaluation_results(enhanced_results)
    
    logger.info("Model evaluation completed")
    return enhanced_results


def main() -> int:
    """
    Main pipeline function.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for failure)
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Gravitational Wave Hunter v2.0 Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/download_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom run name"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip model evaluation step"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info("Gravitational Wave Hunter v2.0 Pipeline")
        logger.info("=" * 50)
        
        # Initialize run manager
        run_manager = RunManager(
            base_dir="runs/",
            config_path=args.config
        )
        
        # Create run
        run_dir = run_manager.create_run(
            run_name=args.run_name,
            include_git_hash=True,
            include_timestamp=True
        )
        
        logger.info(f"Created run directory: {run_dir}")
        
        # Setup logging to run directory
        log_file = run_dir / "logs" / "pipeline.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(args.log_level, str(log_file))
        
        # Run pipeline steps
        if not args.skip_preprocessing:
            preprocess_data(config, run_manager)
        else:
            logger.info("Skipping data preprocessing")
            
        if not args.skip_training:
            train_results = train_model(config, run_manager)
        else:
            logger.info("Skipping model training")
            train_results = {}
            
        if not args.skip_evaluation:
            eval_results = evaluate_model(config, run_manager)
        else:
            logger.info("Skipping model evaluation")
            eval_results = {}
            
        # Mark run as completed
        run_manager.mark_completed(success=True)
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Run directory: {run_dir}")
        
        if train_results:
            logger.info(f"Training - Best val loss: {train_results.get('best_val_loss', 'N/A'):.6f}")
            
        if eval_results:
            logger.info(f"Evaluation - Anomaly rate: {eval_results.get('anomaly_rate', 'N/A'):.1%}")
            if 'accuracy' in eval_results:
                logger.info(f"Evaluation - Accuracy: {eval_results['accuracy']:.3f}")
                
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
