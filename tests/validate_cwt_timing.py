#!/usr/bin/env python3
"""
CWT Timing Validation Script

This script validates the CWT preprocessing timing accuracy using known
gravitational wave events with precise timing information.

Usage:
    python tests/validate_cwt_timing.py --data-dir data/raw --events GW150914,GW151226
"""

import sys
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import CWTPreprocessor, TimingValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_strain_data(data_dir: str, event_name: str) -> np.ndarray:
    """
    Load strain data for a specific event.
    
    Parameters
    ----------
    data_dir : str
        Directory containing strain data files
    event_name : str
        Name of the gravitational wave event
        
    Returns
    -------
    np.ndarray
        Strain data array
        
    Raises
    ------
    FileNotFoundError
        If data file is not found
    """
    # Look for data files in the downloader format
    data_file = None
    
    # Try different naming conventions
    alt_files = [
        Path(data_dir) / f"{event_name}_strain.npy",
        Path(data_dir) / f"{event_name}.npy",
        Path(data_dir) / f"{event_name}_H1.npy",
        Path(data_dir) / f"{event_name}_L1.npy"
    ]
    
    # For GW150914, look for the downloaded files
    if event_name == "GW150914":
        # Look for H1 and L1 detector files
        gw150914_files = [
            Path(data_dir) / "H1_1126259450_32s.npz",
            Path(data_dir) / "L1_1126259450_32s.npz"
        ]
        alt_files.extend(gw150914_files)
    
    for alt_file in alt_files:
        if alt_file.exists():
            data_file = alt_file
            break
    
    if not data_file:
        raise FileNotFoundError(f"No data file found for {event_name} in {data_dir}")
    
    logger.info(f"Loading strain data from {data_file}")
    
    # Handle different file formats
    if data_file.suffix == '.npz':
        # Load from .npz file (downloader format)
        data = np.load(data_file)
        # The strain data is typically stored as 'strain' in the .npz file
        if 'strain' in data:
            strain_data = data['strain']
        else:
            # If no 'strain' key, take the first array
            strain_data = data[list(data.keys())[0]]
    else:
        # Load from .npy file
        strain_data = np.load(data_file)
    
    logger.info(f"Loaded {event_name} data: shape {strain_data.shape}")
    return strain_data


def create_mock_gw_signal(
    sample_rate: int = 4096,
    duration: float = 2.0,
    peak_time: float = 1.0,
    amplitude: float = 1e-21
) -> np.ndarray:
    """
    Create a mock gravitational wave signal for testing.
    
    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz
    duration : float
        Signal duration in seconds
    peak_time : float
        Time of peak amplitude in seconds
    amplitude : float
        Signal amplitude
        
    Returns
    -------
    np.ndarray
        Mock gravitational wave signal
    """
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)
    
    # Create a chirp signal (frequency increases with time)
    f0 = 20.0  # Starting frequency
    f1 = 200.0  # Ending frequency
    
    # Frequency as a function of time
    f = f0 + (f1 - f0) * t / duration
    
    # Create chirp signal
    phase = 2 * np.pi * np.cumsum(f) / sample_rate
    signal = amplitude * np.sin(phase)
    
    # Add Gaussian envelope centered at peak_time
    sigma = 0.1  # Width of the envelope
    envelope = np.exp(-0.5 * ((t - peak_time) / sigma) ** 2)
    
    # Apply envelope
    signal *= envelope
    
    # Add noise
    noise_level = amplitude * 0.1
    noise = np.random.normal(0, noise_level, n_samples)
    signal += noise
    
    return signal


def test_cwt_timing_accuracy(
    data_dir: str,
    events: List[str],
    sample_rate: int = 4096,
    use_mock_data: bool = False
) -> None:
    """
    Test CWT timing accuracy for specified events.
    
    Parameters
    ----------
    data_dir : str
        Directory containing strain data
    events : List[str]
        List of event names to test
    sample_rate : int
        Sampling rate in Hz
    use_mock_data : bool
        Whether to use mock data instead of real data
    """
    logger.info("Starting CWT timing accuracy validation")
    logger.info(f"Events to test: {events}")
    logger.info(f"Sample rate: {sample_rate} Hz")
    
    # Initialize timing validator
    validator = TimingValidator(sample_rate=sample_rate)
    
    # Known event timing information
    event_timing = {
        'GW150914': {'peak_time': 12.4, 'gps_start': 1126259450, 'actual_gps': 1126259462.4},
        'GW151226': {'peak_time': 0.5, 'gps_start': 1135136350.6, 'actual_gps': 1135136350.6},
        'GW170817': {'peak_time': 1.0, 'gps_start': 1187008882.4, 'actual_gps': 1187008882.4}
    }
    
    validation_results = {}
    
    for event_name in events:
        logger.info(f"\nTesting {event_name}...")
        
        try:
            if use_mock_data:
                # Create mock data for testing
                peak_time = event_timing.get(event_name, {}).get('peak_time', 1.0)
                # Use appropriate duration based on peak time
                duration = max(peak_time * 2, 2.0)  # At least 2 seconds, or 2x peak time
                strain_data = create_mock_gw_signal(
                    sample_rate=sample_rate,
                    duration=duration,
                    peak_time=peak_time
                )
                gps_start = event_timing.get(event_name, {}).get('gps_start', 0.0)
            else:
                # Load real data
                strain_data = load_strain_data(data_dir, event_name)
                gps_start = event_timing.get(event_name, {}).get('gps_start', 0.0)
            
            # Validate timing
            results = validator.validate_timing(
                strain_data=strain_data,
                event_name=event_name,
                segment_start_gps=gps_start,
                expected_peak_time=event_timing.get(event_name, {}).get('peak_time')
            )
            
            validation_results[event_name] = results
            
        except Exception as e:
            logger.error(f"Failed to test {event_name}: {e}")
            validation_results[event_name] = {'error': str(e)}
    
    # Generate and print report
    report = validator.generate_timing_report(validation_results)
    print("\n" + report)
    
    # Save report to file
    report_file = Path("results") / "cwt_timing_validation_report.txt"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Timing validation report saved to {report_file}")


def main() -> None:
    """
    Main function for CWT timing validation.
    
    Parses command line arguments and runs timing validation tests.
    """
    parser = argparse.ArgumentParser(
        description="Test CWT timing accuracy using known gravitational wave events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with real data
  python tests/validate_cwt_timing.py --data-dir data/raw --events GW150914,GW151226
  
  # Validate with mock data
  python tests/validate_cwt_timing.py --events GW150914,GW151226,GW170817 --mock-data
  
  # Validate single event
  python tests/validate_cwt_timing.py --data-dir data/raw --events GW150914
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing strain data files"
    )
    parser.add_argument(
        "--events",
        required=True,
        help="Comma-separated list of events to test (e.g., GW150914,GW151226)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=4096,
        help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="Use mock data instead of real data"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse events
    events = [event.strip() for event in args.events.split(',')]
    
    # Check data directory
    if not args.mock_data and not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("Use --mock-data to test with synthetic data")
        sys.exit(1)
    
    try:
        test_cwt_timing_accuracy(
            data_dir=args.data_dir,
            events=events,
            sample_rate=args.sample_rate,
            use_mock_data=args.mock_data
        )
    except KeyboardInterrupt:
        logger.info("\nTiming validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Timing validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
