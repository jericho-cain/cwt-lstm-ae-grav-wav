"""
Generate balanced configuration with proper train/test split for noise data.

This script creates a configuration that:
- Generates 5x more noise segments than signal segments
- Splits noise into separate train/test sets (no overlap)
- Uses all signals in test set only
- Ensures proper anomaly detection evaluation
"""

import yaml
from gwosc.datasets import find_datasets, event_gps
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_all_gw_events():
    """Get all unique GW events from GWOSC."""
    datasets = find_datasets()
    gw_events = [d for d in datasets if d.startswith('GW') and len(d) > 10 and not d.startswith('GWTC')]
    
    # Get unique events (remove version suffixes)
    unique_events = set()
    for d in gw_events:
        unique_events.add(d.split('-')[0])
    
    return sorted(list(unique_events))


def get_event_gps_time(event_name):
    """Get GPS time for an event, with fallback for events without GPS data."""
    try:
        return event_gps(event_name)
    except Exception as e:
        logger.warning(f"Could not get GPS time for {event_name}: {e}")
        return None


def generate_signal_segments(events, detectors=['H1', 'L1']):
    """Generate signal segments for all events (test set only)."""
    segments = []
    
    for i, event in enumerate(events):
        logger.info(f"Processing event {i+1}/{len(events)}: {event}")
        
        gps_time = get_event_gps_time(event)
        if gps_time is None:
            logger.warning(f"Skipping {event} - no GPS time available")
            continue
        
        # Create segments for each detector
        for detector in detectors:
            # 32-second segment centered around the event
            start_gps = int(gps_time - 16)
            end_gps = int(gps_time + 16)
            
            segment = {
                'start_gps': start_gps,
                'end_gps': end_gps,
                'label': f"{event}_{detector}",
                'type': 'signal',
                'detector': detector,
                'duration': 32,
                'sample_rate': 4096,
                'known_event': event
            }
            segments.append(segment)
    
    return segments


def generate_noise_segments(signal_count, detectors=['H1', 'L1'], noise_ratio=5.0):
    """
    Generate noise segments with proper train/test split.
    
    Parameters
    ----------
    signal_count : int
        Number of signal segments (to determine noise count)
    detectors : list
        List of detectors
    noise_ratio : float
        Ratio of noise to signal segments (default 5.0)
        
    Returns
    -------
    tuple
        (train_noise_segments, test_noise_segments)
    """
    total_noise_needed = int(signal_count * noise_ratio)
    train_noise_count = int(total_noise_needed * 0.8)  # 80% for training
    test_noise_count = total_noise_needed - train_noise_count  # 20% for testing
    
    logger.info(f"Generating {total_noise_needed} noise segments:")
    logger.info(f"  Training: {train_noise_count} segments")
    logger.info(f"  Testing: {test_noise_count} segments")
    
    # Generate noise segments from different observing runs
    # O1 noise segments (2015-2016)
    o1_noise_times = list(range(1126259000, 1126260000, 100))  # 10 segments
    o1_noise_times.extend(list(range(1135136000, 1135137000, 100)))  # 10 segments
    o1_noise_times.extend(list(range(1167559800, 1167560800, 100)))  # 10 segments
    
    # O2 noise segments (2016-2017)
    o2_noise_times = list(range(1180922400, 1180923400, 100))  # 10 segments
    o2_noise_times.extend(list(range(1186741800, 1186742800, 100)))  # 10 segments
    o2_noise_times.extend(list(range(1187008800, 1187009800, 100)))  # 10 segments
    
    # O3 noise segments (2019-2020)
    o3_noise_times = list(range(1239082200, 1239083200, 100))  # 10 segments
    o3_noise_times.extend(list(range(1240215400, 1240216400, 100)))  # 10 segments
    o3_noise_times.extend(list(range(1242442900, 1242443900, 100)))  # 10 segments
    o3_noise_times.extend(list(range(1249852200, 1249853200, 100)))  # 10 segments
    
    # O4 noise segments (2023-2024)
    o4_noise_times = list(range(1685577600, 1685578600, 100))  # 10 segments
    o4_noise_times.extend(list(range(1696118400, 1696119400, 100)))  # 10 segments
    o4_noise_times.extend(list(range(1706745600, 1706746600, 100)))  # 10 segments
    
    all_noise_times = o1_noise_times + o2_noise_times + o3_noise_times + o4_noise_times
    
    # Shuffle to ensure random distribution
    random.shuffle(all_noise_times)
    
    # Split into train and test
    train_times = all_noise_times[:train_noise_count // len(detectors)]
    test_times = all_noise_times[train_noise_count // len(detectors):train_noise_count // len(detectors) + test_noise_count // len(detectors)]
    
    # Ensure we have enough noise times
    if len(train_times) < train_noise_count // len(detectors):
        # Extend with more noise times if needed
        additional_times = list(range(1700000000, 1700001000, 100))  # More O4 times
        all_noise_times.extend(additional_times)
        train_times = all_noise_times[:train_noise_count // len(detectors)]
        test_times = all_noise_times[train_noise_count // len(detectors):train_noise_count // len(detectors) + test_noise_count // len(detectors)]
    
    train_segments = []
    test_segments = []
    
    # Generate training noise segments
    for i, start_gps in enumerate(train_times):
        for detector in detectors:
            segment = {
                'start_gps': start_gps,
                'end_gps': start_gps + 32,
                'label': f"train_noise_{detector}_{i+1:04d}",
                'type': 'noise',
                'detector': detector,
                'duration': 32,
                'sample_rate': 4096,
                'split': 'train'
            }
            train_segments.append(segment)
    
    # Generate testing noise segments
    for i, start_gps in enumerate(test_times):
        for detector in detectors:
            segment = {
                'start_gps': start_gps,
                'end_gps': start_gps + 32,
                'label': f"test_noise_{detector}_{i+1:04d}",
                'type': 'noise',
                'detector': detector,
                'duration': 32,
                'sample_rate': 4096,
                'split': 'test'
            }
            test_segments.append(segment)
    
    return train_segments, test_segments


def create_balanced_config():
    """Create the balanced configuration with proper train/test split."""
    logger.info("Getting all GW events...")
    events = get_all_gw_events()
    logger.info(f"Found {len(events)} unique GW events")
    
    logger.info("Generating signal segments...")
    signal_segments = generate_signal_segments(events)
    logger.info(f"Generated {len(signal_segments)} signal segments")
    
    logger.info("Generating noise segments with train/test split...")
    train_noise_segments, test_noise_segments = generate_noise_segments(
        len(signal_segments), noise_ratio=5.0
    )
    
    logger.info(f"Generated {len(train_noise_segments)} training noise segments")
    logger.info(f"Generated {len(test_noise_segments)} testing noise segments")
    
    # Create the balanced configuration
    config = {
        'downloader': {
            'data_directories': {
                'raw_data': 'data/raw/',
                'processed_data': 'data/processed/',
                'manifest_file': 'data/download_manifest.json'
            },
            'detectors': ['H1', 'L1'],
            'observing_runs': ['O1', 'O2', 'O3a', 'O3b', 'O4a'],
            'download_parameters': {
                'max_concurrent': 4,
                'timeout_seconds': 300,
                'retry_attempts': 3,
                'retry_delay': 5
            },
            'data_quality': {
                'validate_downloads': True,
                'check_file_integrity': True,
                'max_nan_percentage': 50.0,
                'max_inf_percentage': 50.0
            },
            'signal_segments': signal_segments,  # All signals go to test set
            'noise_segments': train_noise_segments + test_noise_segments,  # Combined for download
            'train_noise_segments': train_noise_segments,  # Training set
            'test_noise_segments': test_noise_segments,    # Test set
        },
        
        # CWT Preprocessing Configuration
        'preprocessing': {
            'cwt': {
                'sample_rate': 4096,
                'target_height': 8,
                'use_analytic': True,
                'fmin': 20.0,
                'fmax': 512.0,
                'n_scales': 8,
                'wavelet': 'cmor1.5-1.0',
                'k_pad': 10.0,
                'k_coi': 6.0
            }
        },
        
        # Model Configuration
        'model': {
            'model_type': 'cwt_lstm',
            'input_height': 8,
            'input_width': 131072,
            'latent_dim': 32,
            'lstm_hidden': 64,
            'dropout': 0.1,
            'training': {
                'batch_size': 8,
                'num_epochs': 50,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'scheduler': 'reduce_on_plateau',
                'early_stopping_patience': 10,
                'validation_split': 0.2,
                'loss_function': 'mse',
                'augmentation': {
                    'enabled': False,
                    'noise_level': 0.01,
                    'time_shift': 0.1
                }
            },
            'save': {
                'model_dir': 'models/',
                'checkpoint_dir': 'models/checkpoints/',
                'best_model_name': 'best_cwt_lstm_model.pth',
                'final_model_name': 'final_cwt_lstm_model.pth',
                'save_every_n_epochs': 10
            },
            'anomaly_detection': {
                'threshold_percentile': 95.0,
                'reconstruction_error_threshold': None,
                'min_anomaly_score': 0.5
            }
        },
        
        # Pipeline Configuration
        'pipeline': {
            'run_management': {
                'create_unique_dirs': True,
                'run_metadata_file': 'run_metadata.json',
                'include_git_hash': True,
                'include_timestamp': True
            },
            'data_flow': {
                'use_preprocessed_data': True,
                'preprocessed_data_dir': 'data/processed/',
                'train_on_noise_only': True,
                'test_on_all_data': True,
                'train_test_split': {
                    'train_noise_only': True,
                    'test_noise_and_signals': True,
                    'noise_ratio': 5.0
                }
            },
            'output': {
                'results_dir': 'results/',
                'plots_dir': 'results/plots/',
                'logs_dir': 'results/logs/',
                'reports_dir': 'results/reports/'
            }
        }
    }
    
    return config


def main():
    """Generate and save the balanced configuration."""
    logger.info("Generating balanced configuration with proper train/test split...")
    
    config = create_balanced_config()
    
    # Save to file
    output_file = 'config/balanced_gw_config.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_file}")
    
    # Print summary
    signal_count = len(config['downloader']['signal_segments'])
    train_noise_count = len(config['downloader']['train_noise_segments'])
    test_noise_count = len(config['downloader']['test_noise_segments'])
    total_noise_count = train_noise_count + test_noise_count
    total_segments = signal_count + total_noise_count
    
    print(f"\nBalanced Configuration Summary:")
    print(f"  Signal segments: {signal_count}")
    print(f"  Training noise segments: {train_noise_count}")
    print(f"  Testing noise segments: {test_noise_count}")
    print(f"  Total noise segments: {total_noise_count}")
    print(f"  Total segments: {total_segments}")
    print(f"  Noise-to-signal ratio: {total_noise_count/signal_count:.1f}:1")
    
    print(f"\nTraining strategy:")
    print(f"  Train on: Noise only ({train_noise_count} segments)")
    print(f"  Test on: Noise + Signals ({test_noise_count + signal_count} segments)")
    print(f"  Expected test set: {signal_count} signals, {test_noise_count} noise")
    if test_noise_count > 0:
        print(f"  Test signal-to-noise ratio: {signal_count/test_noise_count:.1f}:1")
    else:
        print(f"  Test signal-to-noise ratio: N/A (no test noise segments)")
    
    print(f"\nData split:")
    print(f"  Training set: {train_noise_count} noise segments")
    print(f"  Test set: {test_noise_count} noise + {signal_count} signal segments")
    print(f"  No overlap between train and test noise segments")


if __name__ == "__main__":
    main()
