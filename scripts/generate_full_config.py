"""
Generate full configuration with all 241 GW events for comprehensive testing.

This script creates a configuration file that includes:
- All 241 unique GW events from GWOSC
- Proper train/test split: noise for training, noise+signals for testing
- Both H1 and L1 detectors for each event
- Comprehensive noise segments for training
"""

import yaml
from gwosc.datasets import find_datasets, event_gps
import logging

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
    """Generate signal segments for all events."""
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


def generate_noise_segments():
    """Generate noise segments for training."""
    # Generate noise segments from different observing runs
    noise_segments = []
    
    # O1 noise segments (2015-2016)
    o1_noise_times = [
        1126259000, 1126259100, 1126259200, 1126259300, 1126259400,
        1126259500, 1126259600, 1126259700, 1126259800, 1126259900,
        1135136000, 1135136100, 1135136200, 1135136300, 1135136400,
        1135136500, 1135136600, 1135136700, 1135136800, 1135136900,
        1167559800, 1167559900, 1167560000, 1167560100, 1167560200,
        1167560300, 1167560400, 1167560500, 1167560600, 1167560700
    ]
    
    # O2 noise segments (2016-2017)
    o2_noise_times = [
        1180922400, 1180922500, 1180922600, 1180922700, 1180922800,
        1180922900, 1180923000, 1180923100, 1180923200, 1180923300,
        1186741800, 1186741900, 1186742000, 1186742100, 1186742200,
        1186742300, 1186742400, 1186742500, 1186742600, 1186742700,
        1187008800, 1187008900, 1187009000, 1187009100, 1187009200,
        1187009300, 1187009400, 1187009500, 1187009600, 1187009700
    ]
    
    # O3 noise segments (2019-2020)
    o3_noise_times = [
        1239082200, 1239082300, 1239082400, 1239082500, 1239082600,
        1239082700, 1239082800, 1239082900, 1239083000, 1239083100,
        1240215400, 1240215500, 1240215600, 1240215700, 1240215800,
        1240215900, 1240216000, 1240216100, 1240216200, 1240216300,
        1242442900, 1242443000, 1242443100, 1242443200, 1242443300,
        1242443400, 1242443500, 1242443600, 1242443700, 1242443800,
        1249852200, 1249852300, 1249852400, 1249852500, 1249852600,
        1249852700, 1249852800, 1249852900, 1249853000, 1249853100
    ]
    
    all_noise_times = o1_noise_times + o2_noise_times + o3_noise_times
    
    for i, start_gps in enumerate(all_noise_times):
        for detector in ['H1', 'L1']:
            segment = {
                'start_gps': start_gps,
                'end_gps': start_gps + 32,
                'label': f"noise_{detector}_{i+1:03d}",
                'type': 'noise',
                'detector': detector,
                'duration': 32,
                'sample_rate': 4096
            }
            noise_segments.append(segment)
    
    return noise_segments


def create_full_config():
    """Create the full configuration with all events."""
    logger.info("Getting all GW events...")
    events = get_all_gw_events()
    logger.info(f"Found {len(events)} unique GW events")
    
    logger.info("Generating signal segments...")
    signal_segments = generate_signal_segments(events)
    logger.info(f"Generated {len(signal_segments)} signal segments")
    
    logger.info("Generating noise segments...")
    noise_segments = generate_noise_segments()
    logger.info(f"Generated {len(noise_segments)} noise segments")
    
    # Create the full configuration
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
            'signal_segments': signal_segments,
            'noise_segments': noise_segments
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
                'test_on_all_data': True
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
    """Generate and save the full configuration."""
    logger.info("Generating full configuration with all GW events...")
    
    config = create_full_config()
    
    # Save to file
    output_file = 'config/full_gw_config.yaml'
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Configuration saved to {output_file}")
    
    # Print summary
    signal_count = len(config['downloader']['signal_segments'])
    noise_count = len(config['downloader']['noise_segments'])
    total_segments = signal_count + noise_count
    
    print(f"\nConfiguration Summary:")
    print(f"  Signal segments: {signal_count}")
    print(f"  Noise segments: {noise_count}")
    print(f"  Total segments: {total_segments}")
    print(f"  Detectors: {config['downloader']['detectors']}")
    print(f"  Observing runs: {config['downloader']['observing_runs']}")
    
    print(f"\nTraining strategy:")
    print(f"  Train on: Noise only ({noise_count} segments)")
    print(f"  Test on: Noise + Signals ({total_segments} segments)")
    print(f"  Expected test set: {signal_count} signals, {noise_count} noise")


if __name__ == "__main__":
    main()
