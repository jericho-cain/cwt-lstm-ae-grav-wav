#!/usr/bin/env python3
"""
Clean Gravitational Wave Detection Pipeline Runner

This script runs the complete end-to-end pipeline for gravitational wave detection
using the corrected CWT preprocessing approach. The pipeline includes:

1. Data Download: Downloads real gravitational wave data from GWOSC
2. Preprocessing: Applies corrected CWT preprocessing (focused 50-200 Hz range)
3. Training: Trains LSTM autoencoder on noise-only data
4. Evaluation: Tests model on mixed noise+signal data
5. Metrics: Generates comprehensive performance metrics and plots

Key Features:
- Uses focused frequency range (50-200 Hz) where GWs show strongest differences
- Preserves raw magnitude scalogram (no log transform, no normalization)
- Minimal downsampling to preserve chirp dynamics
- Proper train/test split with no data leakage

Usage:
    python scripts/run_clean_pipeline.py --config config/pipeline_clean_config.yaml

The script creates timestamped run directories with all results, metrics, and plots.

Author: Jericho Cain
Date: October 3, 2025
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import numpy as np
import json
import torch

# Add src to path (scripts/core -> scripts -> root -> src)
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from downloader.gwosc_downloader import CleanGWOSCDownloader
from training import CWTModelTrainer
from evaluation import AnomalyDetector, PostProcessor, MetricsEvaluator
from preprocessing import CWTPreprocessor
from pipeline import RunManager


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, run_dir: Optional[Path] = None) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str, optional
        Logging level, by default "INFO"
    log_file : str, optional
        Log file path, by default None
    run_dir : Path, optional
        Run directory for automatic log file creation
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (priority: explicit log_file, then run_dir, then None)
    log_path = None
    if log_file:
        log_path = log_file
    elif run_dir:
        log_path = run_dir / "logs" / "pipeline.log"
    
    if log_path:
        # Ensure log directory exists
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # More verbose logging to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_path}")


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
    
    # Check if relative re-whitened data exists
    relwhite_dir = Path('data/raw_relwhite')
    use_relwhite = relwhite_dir.exists() and len(list(relwhite_dir.glob('*.npy'))) > 2000
    
    if use_relwhite:
        logger.info("=" * 80)
        logger.info("USING RELATIVE RE-WHITENED DATA")
        logger.info("=" * 80)
        logger.info(f"Loading from: {relwhite_dir}")
        logger.info(f"Files found: {len(list(relwhite_dir.glob('*.npy')))}")
        logger.info("Whitening will be SKIPPED in CWT processing (already spectrally aligned)")
        logger.info("=" * 80)
        raw_data_dir = relwhite_dir
    else:
        logger.info("Using original GWOSC data from data/raw/")
    
    # Create processed directory
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Compute global normalization statistics from training noise
    from preprocessing.cwt import compute_global_normalization_stats
    
    logger.info("=" * 80)
    logger.info("COMPUTING GLOBAL NORMALIZATION STATISTICS")
    logger.info("=" * 80)
    
    # Load manifest to identify training noise files
    manifest_path = Path(config['downloader']['data_directories']['manifest_file'])
    global_mean = None
    global_std = None
    
    # Check if normalization params already exist
    norm_params_file = processed_data_dir / 'normalization_params.json'
    
    if norm_params_file.exists():
        logger.info(f"Loading existing normalization params from {norm_params_file}")
        with open(norm_params_file, 'r') as f:
            norm_params = json.load(f)
            global_mean = norm_params['global_mean']
            global_std = norm_params['global_std']
            logger.info(f"  Mean: {global_mean:.6e}")
            logger.info(f"  Std:  {global_std:.6e}")
            logger.info(f"  Computed from {norm_params['n_training_files']} files on {norm_params.get('computed_date', 'unknown date')}")
    else:
        # Compute new normalization stats
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Get training noise files (80% split)
            training_noise_files = []
            for download in manifest.get('downloads', []):
                if download.get('segment_type') == 'noise' and download.get('successful', False):
                    segment_id = download.get('segment_id')
                    filepath = raw_data_dir / f"{segment_id}.npz"
                    if filepath.exists():
                        training_noise_files.append(filepath)
            
            # Apply same 80/20 split as training
            np.random.seed(42)
            np.random.shuffle(training_noise_files)
            split_idx = int(len(training_noise_files) * 0.8)
            train_noise_subset = training_noise_files[:split_idx]
            
            logger.info(f"Computing global stats from {len(train_noise_subset)} training noise files (80% of {len(training_noise_files)} total)")
            
            # Compute global stats
            cwt_config = config['preprocessing']['cwt']
            global_mean, global_std = compute_global_normalization_stats(
                train_noise_subset,
                sample_rate=cwt_config['sample_rate'],
                fmin=cwt_config['fmin']
            )
            
            # Save normalization params
            import datetime
            norm_params = {
                'global_mean': float(global_mean),
                'global_std': float(global_std),
                'n_training_files': len(train_noise_subset),
                'n_total_noise_files': len(training_noise_files),
                'computed_date': datetime.datetime.now().isoformat(),
                'sample_rate': cwt_config['sample_rate'],
                'fmin': cwt_config['fmin']
            }
            
            with open(norm_params_file, 'w') as f:
                json.dump(norm_params, f, indent=2)
            
            logger.info(f"Normalization params saved to {norm_params_file}")
        else:
            logger.warning(f"No manifest found at {manifest_path}, skipping global normalization")
    
    logger.info("=" * 80)
    
    # STEP 2: Initialize preprocessor with global stats
    cwt_config = config['preprocessing']['cwt']
    preprocessor = CWTPreprocessor(
        sample_rate=cwt_config['sample_rate'],
        target_height=cwt_config['target_height'],
        target_width=cwt_config['target_width'],
        use_analytic=cwt_config['use_analytic'],
        fmin=cwt_config['fmin'],
        fmax=cwt_config['fmax'],
        global_mean=global_mean,
        global_std=global_std,
        skip_whitening=use_relwhite  # Skip if using relative re-whitened data
    )
    
    # STEP 3: Process all files using global normalization
    if use_relwhite:
        # Load .npy files (relative re-whitened strain)
        all_raw_files = list(raw_data_dir.glob("*.npy"))
        # Filter out .json sidecars
        all_raw_files = [f for f in all_raw_files if f.suffix == '.npy']
    else:
        # Load .npz files (original GWOSC strain)
        all_raw_files = list(raw_data_dir.glob("*.npz"))
    
    detectors = config['downloader']['signals']['detectors']
    raw_files = []
    for detector in detectors:
        detector_files = [f for f in all_raw_files if f.name.startswith(f"{detector}_")]
        raw_files.extend(detector_files)
    logger.info(f"Processing {len(raw_files)} {detectors[0]} files (out of {len(all_raw_files)} total)")
    
    for i, raw_file in enumerate(raw_files):
        try:
            # Check if already processed
            # For relative re-whitened: H1_1126259462.4_32s_relwhite.npy -> H1_1126259462.4_32s_relwhite_cwt.npy
            # For original: H1_1126259462.4_32s.npz -> H1_1126259462.4_32s_cwt.npy
            if use_relwhite:
                base_name = raw_file.stem  # Removes .npy
                processed_file = processed_data_dir / f"{base_name}_cwt.npy"
            else:
                processed_file = processed_data_dir / f"{raw_file.stem}_cwt.npy"
            
            if processed_file.exists():
                if (i + 1) % 100 == 0:
                    logger.info(f"Skipped {i + 1}/{len(raw_files)} files (already processed)")
                continue
            
            # Load raw data
            if use_relwhite:
                # Load relative re-whitened .npy file
                strain = np.load(raw_file)
            else:
                # Load original .npz file
                data = np.load(raw_file)
                strain = data['strain']
            
            # Process with CWT
            cwt_data = preprocessor.process(strain)
            
            # Validate CWT output shape
            expected_shape = (8, 4096)  # Match EC2 dimensions (height, width)
            if cwt_data.shape != expected_shape:
                logger.error(f"CWT shape validation failed for {raw_file.name}")
                logger.error(f"Expected: {expected_shape}, Got: {cwt_data.shape}")
                logger.error("This indicates a bug in CWT preprocessing. Stopping pipeline.")
                raise ValueError(f"CWT output shape mismatch: expected {expected_shape}, got {cwt_data.shape}")
            
            logger.debug(f"CWT validation passed: {cwt_data.shape}")
        
            # Save processed data
            np.save(processed_file, cwt_data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(raw_files)} files")
                
        except Exception as e:
            logger.error(f"Failed to process {raw_file}: {e}")
    
    logger.info(f"Preprocessing completed. Processed files saved to {processed_data_dir}")


def train_model(config: Dict[str, Any], run_dir: Path, config_path: str) -> str:
    """
    Train the CWT-LSTM autoencoder model.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    run_dir : Path
        Run directory for saving model
    config_path : str
        Path to configuration file
        
    Returns
    -------
    str
        Path to trained model
    """
    logger = logging.getLogger(__name__)
    
    # Initialize trainer with the correct config file
    trainer = CWTModelTrainer(config_path)
    
    # Train model
    training_results = trainer.train()
    
    # The model is saved to the root models directory as per config
    # We need to copy it to the run directory for evaluation
    model_config = config['model']['save']
    source_model_path = Path(model_config['model_dir']) / model_config['final_model_name']
    run_model_path = run_dir / "models" / model_config['final_model_name']
    
    # Create run models directory and copy the model
    run_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    if source_model_path.exists():
        import shutil
        shutil.copy2(source_model_path, run_model_path)
        logger.info(f"Model copied from {source_model_path} to {run_model_path}")
    else:
        logger.error(f"Source model not found: {source_model_path}")
        raise FileNotFoundError(f"Model file not found: {source_model_path}")
    
    logger.info(f"Model training completed. Model available at {run_model_path}")
    return str(run_model_path)


def evaluate_model(config: Dict[str, Any], model_path: str, run_dir: Path, config_path: str) -> Dict[str, Any]:
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
    detector = AnomalyDetector(model_path, config_path)
    
    # Load test data for evaluation
    logger.info("Loading test data for evaluation...")
    processed_dir = Path(config['pipeline']['data_flow']['preprocessed_data_dir'])
    
    # Load manifest to get proper labels (same logic as trainer)
    manifest_path = Path(config['downloader']['data_directories']['manifest_file'])
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
            # Filename format: H1_1376029162.5_32s_whiten_cwt -> GPS is in parts[1]
            # Note: PSD-whitened files have GPS center, manifest has GPS start
            if len(filename_parts) >= 2:
                gps_time = float(filename_parts[1])
                
                # Try exact match first
                segment_type = gps_to_type.get(gps_time, None)
                
                # If no match and filename suggests it's a center time (from rewhitening or relwhitening),
                # try subtracting 16s to get start time
                if segment_type is None and ('whiten' in file_path.name or 'relwhite' in file_path.name):
                    gps_start = gps_time - 16.0
                    segment_type = gps_to_type.get(gps_start, 'noise')
                elif segment_type is None:
                    segment_type = 'noise'
                
                if segment_type == 'noise':
                    noise_files.append(file_path)
                elif segment_type == 'signal':
                    signal_files.append(file_path)
        except (ValueError, IndexError):
            # Default to noise if parsing fails
            noise_files.append(file_path)
    
    # Split noise files into train/test (80/20 split) - same as trainer
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
    
    # Load and combine test data
    test_data = []
    test_labels = []
    test_filenames = []
    
    # Load test noise data (label = 0)
    for file_path in test_noise_files:
        data = np.load(file_path)
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        test_data.append(data)
        test_labels.extend([0] * data.shape[0])
        test_filenames.extend([file_path.name] * data.shape[0])
    
    # Load signal data (label = 1)
    for file_path in signal_files:
        data = np.load(file_path)
        if len(data.shape) == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
        test_data.append(data)
        test_labels.extend([1] * data.shape[0])
        test_filenames.extend([file_path.name] * data.shape[0])
    
    # Combine all test data
    if test_data:
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.array(test_labels)
        logger.info(f"Loaded test data: {test_data.shape}, labels: {np.sum(test_labels)} signals, {np.sum(1-test_labels)} noise")
    else:
        logger.warning("No test data found!")
        return {}
    
    # Get scoring strategy from config
    scoring_strategy = config['model']['anomaly_detection'].get('scoring_strategy', 'mean')
    logger.info(f"Using scoring strategy: {scoring_strategy}")
    
    # Run anomaly detection with optimal threshold
    results = detector.detect_anomalies(test_data, test_labels, use_optimal_threshold=True, scoring_strategy=scoring_strategy)
    
    # Save results (using optimal threshold only)
    combined_results = {
        'optimal_threshold_results': results,
        'threshold_info': {
            'threshold': results['threshold'],
            'threshold_type': results.get('threshold_type', 'optimal'),
            'f1_score': results.get('f1_score', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'accuracy': results.get('accuracy', 0)
        }
    }
    
    # Save results to run directory
    results_file = run_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Save full reconstruction errors and predictions for analysis
    detailed_results_file = run_dir / "detailed_results.npz"
    np.savez(
        detailed_results_file,
        reconstruction_errors=results['reconstruction_errors'],
        predictions=results['predictions'],
        labels=test_labels,
        threshold=results['threshold'],
        filenames=np.array(test_filenames, dtype=object)
    )
    logger.info(f"Detailed results (reconstruction errors, predictions, labels, filenames) saved to {detailed_results_file}")
    
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
    
    # Return optimal threshold metrics for consistency
    metrics = {
        'accuracy': results.get('accuracy', 0),
        'precision': results.get('precision', 0),
        'recall': results.get('recall', 0),
        'f1_score': results.get('f1_score', 0),
        'roc_auc': results.get('roc_auc', 0),
        'pr_auc': results.get('pr_auc', 0),
        'threshold_used': results.get('threshold_type', 'unknown'),
        'threshold_value': results.get('threshold', 0)
    }
    
    logger.info(f"Model evaluation completed. Results saved to {run_dir}")
    logger.info(f"Using {results.get('threshold_type', 'unknown')} threshold: {results.get('threshold', 0):.6f}")
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
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize run manager with correct config path
        run_manager = RunManager(config_path=args.config)
        run_dir = run_manager.create_run()
        
        # Setup logging with run directory
        setup_logging(args.log_level, args.log_file, run_dir)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Loaded configuration from {args.config}")
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
            model_path = train_model(config, run_dir, args.config)
        else:
            logger.info("Skipping model training")
            model_path = None
        
        # Evaluate model
        if not args.skip_evaluation and model_path:
            logger.info("Starting model evaluation...")
            evaluation_results = evaluate_model(config, model_path, run_dir, args.config)
            logger.info(f"Evaluation results: {evaluation_results}")
        else:
            logger.info("Skipping model evaluation")
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
