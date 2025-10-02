"""
Standalone Data Downloader for Gravitational Wave Data

This module provides a clean, standalone data downloader that:
- Never runs as part of other pipelines
- Uses YAML configuration files with schema validation
- Creates JSON manifest of all download attempts
- Prevents duplicate downloads
- Classifies segments as noise/signal/unlabeled
- Tracks data quality (NaN/Inf presence)
- Requires config confirmation to prevent accidental overwrites
"""

import os
import json
import yaml
import logging
import requests
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from gwosc.datasets import event_gps
from gwosc.timeline import get_segments

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class GWOSCDownloader:
    """
    Standalone downloader for GWOSC (Gravitational Wave Open Science Center) data.
    
    This class handles downloading gravitational wave strain data from GWOSC
    with proper error handling, state tracking, and data quality validation.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
        
    Attributes
    ----------
    config_path : str
        Path to configuration file
    config : Dict[str, Any]
        Loaded configuration dictionary
    manifest_path : str
        Path to download manifest JSON file
    raw_data_dir : str
        Directory for raw downloaded data
    manifest : Dict[str, Any]
        Download manifest dictionary
        
    Examples
    --------
    >>> downloader = GWOSCDownloader('config/download_config.yaml')
    >>> results = downloader.download_all_segments()
    """
    
    def __init__(self, config_path: str) -> None:
        """
        Initialize the downloader with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
            
        Raises
        ------
        ValueError
            If configuration file is invalid or missing required fields
        FileNotFoundError
            If configuration file does not exist
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.manifest_path = self.config['downloader']['data_directories']['manifest_file']
        self.raw_data_dir = self.config['downloader']['data_directories']['raw_data']
        self.manifest = self._load_manifest()
        
        # Create directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        
        logger.info(f"Initialized GWOSC downloader")
        logger.info(f"Raw data directory: {self.raw_data_dir}")
        logger.info(f"Manifest file: {self.manifest_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load and validate configuration file.
        
        Returns
        -------
        Dict[str, Any]
            Loaded configuration dictionary
            
        Raises
        ------
        ValueError
            If configuration is missing required keys
        FileNotFoundError
            If configuration file does not exist
        yaml.YAMLError
            If configuration file is malformed
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Basic validation
            required_keys = ['downloader']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load existing download manifest or create new one.
        
        Returns
        -------
        Dict[str, Any]
            Download manifest dictionary with 'created', 'last_updated', and 'downloads' keys
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    manifest = json.load(f)
                logger.info(f"Loaded existing manifest with {len(manifest.get('downloads', []))} entries")
                return manifest
            except Exception as e:
                logger.warning(f"Failed to load manifest, creating new one: {e}")
        
        return {
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "downloads": []
        }
    
    def _save_manifest(self) -> None:
        """
        Save the current manifest to disk.
        
        Raises
        ------
        IOError
            If manifest file cannot be written
        """
        self.manifest["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2, default=str)
            logger.debug(f"Saved manifest to {self.manifest_path}")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            raise
    
    def _get_segment_id(self, detector: str, start_gps: int, duration: int) -> str:
        """
        Generate unique segment ID.
        
        Parameters
        ----------
        detector : str
            Detector name (e.g., 'H1', 'L1', 'V1')
        start_gps : int
            Start GPS time
        duration : int
            Duration in seconds
            
        Returns
        -------
        str
            Unique segment identifier in format '{detector}_{start_gps}_{duration}s'
            
        Examples
        --------
        >>> downloader._get_segment_id('H1', 1126259462, 32)
        'H1_1126259462_32s'
        """
        return f"{detector}_{start_gps}_{duration}s"
    
    def _is_already_downloaded(self, segment_id: str) -> bool:
        """Check if segment has already been downloaded."""
        for download in self.manifest.get('downloads', []):
            if download.get('segment_id') == segment_id:
                return True
        return False
    
    def _download_strain_data(self, detector: str, start_gps: int, duration: int, 
                             sample_rate: int = 4096) -> Optional[np.ndarray]:
        """
        Download strain data from GWOSC for a specific time segment.
        
        Args:
            detector (str): Detector name (H1, L1, V1, etc.)
            start_gps (int): Start GPS time
            duration (int): Duration in seconds
            sample_rate (int): Sample rate in Hz
            
        Returns:
            np.ndarray: Strain data or None if download failed
        """
        try:
            # Check if this is a signal segment (known event)
            event_name = None
            for segment in self.config.get('downloader', {}).get('signal_segments', []):
                if (segment.get('start_gps') <= start_gps <= segment.get('end_gps', start_gps + duration) and
                    segment.get('detector') == detector):
                    event_name = segment.get('known_event')
                    break
            
            if event_name:
                try:
                    # Get the actual GPS time for the event
                    actual_gps = event_gps(event_name)
                    logger.info(f"Found event {event_name} at GPS time {actual_gps}")
                    
                    # Download real data using GWOSC API
                    return self._download_real_gwosc_data(detector, start_gps, duration, sample_rate)
                    
                except Exception as e:
                    logger.warning(f"Failed to get GPS time for {event_name}: {e}")
                    return None
            
            # Check if this is a noise segment
            is_noise_segment = False
            for segment in self.config.get('downloader', {}).get('noise_segments', []):
                if (segment.get('start_gps') <= start_gps <= segment.get('end_gps', start_gps + duration) and
                    segment.get('detector') == detector):
                    is_noise_segment = True
                    break
            
            if is_noise_segment:
                # Download real noise data using GWOSC API
                return self._download_real_gwosc_data(detector, start_gps, duration, sample_rate)
            
            # No matching segment found
            logger.error(f"No matching segment found for {detector} at {start_gps}")
            return None
            
        except Exception as e:
            logger.error(f"Unexpected error downloading {detector} data: {e}")
            return None
    
    def _download_real_gwosc_data(self, detector: str, start_gps: int, duration: int, 
                                 sample_rate: int = 4096) -> Optional[np.ndarray]:
        """
        Download real strain data from GWOSC using the official gwosc client.
        
        Args:
            detector (str): Detector name (H1, L1, V1, etc.)
            start_gps (int): Start GPS time
            duration (int): Duration in seconds
            sample_rate (int): Sample rate in Hz
            
        Returns:
            np.ndarray: Strain data or None if download failed
        """
        try:
            # Import required libraries
            try:
                from gwosc.datasets import event_gps, find_datasets
                from gwosc import locate
                from gwosc.timeline import get_segments
                from gwosc.datasets import run_segment
                from gwpy.timeseries import TimeSeries
            except ImportError as e:
                logger.error(f"Required libraries not available: {e}")
                logger.error("Install with: pip install gwosc gwpy")
                return None
            
            # Check if this is a signal segment or noise segment
            event_name = None
            segment_type = None
            
            # Check signal segments first
            for segment in self.config.get('downloader', {}).get('signal_segments', []):
                if (segment.get('start_gps') <= start_gps <= segment.get('end_gps', start_gps + duration) and
                    segment.get('detector') == detector):
                    event_name = segment.get('known_event')
                    segment_type = 'signal'
                    break
            
            # If not a signal, check if it's a noise segment
            if not event_name:
                for segment in self.config.get('downloader', {}).get('noise_segments', []):
                    if (segment.get('start_gps') <= start_gps <= segment.get('end_gps', start_gps + duration) and
                        segment.get('detector') == detector):
                        segment_type = 'noise'
                        break
            
            if not segment_type:
                logger.error(f"No matching segment found for {detector} at GPS {start_gps}")
                return None
            
            if segment_type == 'signal' and not event_name:
                logger.error("Signal segment found but no event name - cannot download real data")
                return None
            
            logger.info(f"Downloading real GWOSC data for {event_name or 'noise'} ({detector})")
            
            # Handle signal segments vs noise segments differently
            if segment_type == 'signal':
                # For signal segments, use the existing event-based download method
                try:
                    logger.info(f"Attempting to locate direct file URLs for {event_name}")
                    
                    # Get direct file URLs for the event
                    file_urls = locate.get_event_urls(event_name, sample_rate=sample_rate)
                    
                    if not file_urls:
                        logger.error(f"No file URLs found for {event_name}")
                        return None
                    
                    # Filter URLs for the specific detector
                    detector_urls = [url for url in file_urls if detector in url]
                    
                    if not detector_urls:
                        logger.error(f"No file URLs found for {event_name} {detector}")
                        return None
                    
                    # Use the first available URL
                    file_url = detector_urls[0]
                    logger.info(f"Downloading from direct URL: {file_url}")
                    
                    # Download and parse the file
                    import requests
                    import h5py
                    import io
                    
                    response = requests.get(file_url, timeout=60)
                    response.raise_for_status()
                    
                    # Parse HDF5 data
                    with h5py.File(io.BytesIO(response.content), 'r') as f:
                        # Look for strain data in the HDF5 file
                        strain_data = None
                        
                        # Try common strain data paths
                        possible_paths = [
                            f'strain/{detector}',
                            f'strain/Strain',
                            f'{detector}/strain',
                            f'strain'
                        ]
                        
                        for path in possible_paths:
                            if path in f:
                                strain_data = f[path][:]
                                logger.info(f"Found strain data at path: {path}")
                                break
                        
                        if strain_data is None:
                            # List available keys for debugging
                            logger.error("Available keys in HDF5 file:")
                            def print_keys(name, obj):
                                logger.error(f"  {name}: {type(obj)}")
                            f.visititems(print_keys)
                            raise ValueError("Could not find strain data in HDF5 file")
                        
                        # Convert to float32 and ensure correct length
                        strain_array = np.array(strain_data, dtype=np.float32)
                        
                        # Resample if necessary
                        if len(strain_array) != duration * sample_rate:
                            logger.warning(f"Data length {len(strain_array)} != expected {duration * sample_rate}, resampling")
                            from scipy import signal
                            strain_array = signal.resample(strain_array, duration * sample_rate)
                        
                        logger.info(f"Successfully downloaded {len(strain_array)} samples via direct URL")
                        return strain_array
                        
                except Exception as e:
                    logger.error(f"Direct URL download failed: {e}")
                    return None
            
            else:
                # For noise segments, use GWpy's fetch_open_data with science-mode segment validation
                try:
                    logger.info(f"Downloading real noise data for {detector} at GPS {start_gps}")
                    
                    # Check if this GPS time is in a valid science-mode segment
                    if not self._is_valid_science_segment(detector, start_gps, duration):
                        logger.error(f"GPS {start_gps} is not in a valid science-mode segment for {detector}")
                        return None
                    
                    # Use GWpy's fetch_open_data which handles proper GWOSC URLs automatically
                    end_gps = start_gps + duration
                    
                    logger.info(f"Fetching open data for {detector} from GPS {start_gps} to {end_gps}")
                    
                    # Add retry logic for transient failures
                    max_tries = 3
                    delay = 1.0
                    
                    for attempt in range(max_tries):
                        try:
                            # Fetch open data using GWpy - this uses the correct GWOSC URLs
                            ts = TimeSeries.fetch_open_data(
                                detector, 
                                start_gps, 
                                end_gps, 
                                cache=True,
                                host='https://gwosc.org'
                            )
                            
                            # Check for non-finite values
                            if not np.isfinite(ts.value).all():
                                logger.warning(f"Non-finite samples detected in {detector} noise data")
                                # Try to clean the data
                                ts = ts.fillna(0)  # Fill NaN with zeros
                                ts = ts.replace([np.inf, -np.inf], 0)  # Replace inf with zeros
                            
                            # Convert to numpy array and ensure correct length
                            strain_array = np.array(ts.value, dtype=np.float32)
                            
                            # Resample if necessary
                            if len(strain_array) != duration * sample_rate:
                                logger.warning(f"Data length {len(strain_array)} != expected {duration * sample_rate}, resampling")
                                from scipy import signal
                                strain_array = signal.resample(strain_array, duration * sample_rate)
                            
                            logger.info(f"Successfully downloaded {len(strain_array)} real noise samples via GWpy")
                            return strain_array
                            
                        except Exception as e:
                            if attempt < max_tries - 1:
                                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                                time.sleep(delay)
                                delay *= 2
                            else:
                                raise e
                    
                except Exception as e:
                    logger.error(f"Error downloading real noise data via GWpy: {e}")
                    return None
            
        except Exception as e:
            logger.error(f"Unexpected error downloading real GWOSC data: {e}")
            return None
    
    def _is_valid_science_segment(self, detector: str, start_gps: int, duration: int) -> bool:
        """
        Check if the given GPS time is in a valid science-mode segment.
        
        Args:
            detector (str): Detector name (H1, L1, etc.)
            start_gps (int): Start GPS time
            duration (int): Duration in seconds
            
        Returns:
            bool: True if valid science-mode segment, False otherwise
        """
        try:
            end_gps = start_gps + duration
            
            # Get science-mode segments for the detector
            # Use the expression: {detector}_NO_CW_HW_INJ (this excludes hardware injections)
            expr = f"{detector}_NO_CW_HW_INJ"
            
            # Query for science-mode segments that overlap with our time range
            segments = get_segments(expr, start_gps - 100, end_gps + 100)  # Add buffer
            
            if not segments:
                logger.warning(f"No science-mode segments found for {detector} near GPS {start_gps}")
                return False
            
            # Check if our time range overlaps with any science-mode segment
            for seg_start, seg_end in segments:
                if seg_start <= start_gps and end_gps <= seg_end:
                    logger.info(f"GPS {start_gps}-{end_gps} is in valid science-mode segment {seg_start}-{seg_end}")
                    return True
            
            logger.warning(f"GPS {start_gps}-{end_gps} is not in any science-mode segment for {detector}")
            return False
            
        except Exception as e:
            logger.error(f"Error checking science-mode segments: {e}")
            return False
    
    
    def _parse_gwf_file(self, content: bytes, detector: str) -> Optional[np.ndarray]:
        """
        Parse a .gwf file and extract strain data.
        
        Args:
            content (bytes): Raw .gwf file content
            detector (str): Detector name for channel selection
            
        Returns:
            np.ndarray: Strain data or None if parsing failed
        """
        try:
            # This is a placeholder - real .gwf parsing requires specialized libraries
            # For now, we'll need to implement this or use an existing library
            logger.error("GWF file parsing not yet implemented")
            logger.error("Need to implement .gwf file parsing or use a library like gwpy")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing GWF file: {e}")
            return None
    
    
    
    
    def _validate_data_quality(self, strain_data: np.ndarray) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            strain_data (np.ndarray): Strain data to validate
            
        Returns:
            Dict[str, Any]: Quality metrics and statistics
        """
        quality = {
            "has_nan": np.any(np.isnan(strain_data)),
            "has_inf": np.any(np.isinf(strain_data)),
            "has_zero_variance": np.std(strain_data) < 1e-25,
            "has_reasonable_range": np.all(np.abs(strain_data) < 1e-18),
            "nan_count": np.sum(np.isnan(strain_data)),
            "inf_count": np.sum(np.isinf(strain_data)),
            "total_samples": len(strain_data),
            "nan_percentage": (np.sum(np.isnan(strain_data)) / len(strain_data)) * 100,
            "inf_percentage": (np.sum(np.isinf(strain_data)) / len(strain_data)) * 100
        }
        
        return quality
        
    def _clean_data(self, strain_data: np.ndarray, quality: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Clean data by handling NaN/Inf values.
        
        Args:
            strain_data (np.ndarray): Raw strain data
            quality (Dict[str, Any]): Data quality metrics
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Cleaned data and cleaning report
        """
        cleaned_data = strain_data.copy()
        cleaning_report = {
            "original_quality": quality,
            "cleaning_applied": [],
            "final_quality": None
        }
        
        # If more than 50% of data is NaN/Inf, reject the segment
        if quality["nan_percentage"] > 50 or quality["inf_percentage"] > 50:
            logger.warning(f"Segment rejected: {quality['nan_percentage']:.1f}% NaN, {quality['inf_percentage']:.1f}% Inf")
            return None, {"rejected": True, "reason": "Too many NaN/Inf values"}
        
        # Handle NaN values
        if quality["has_nan"]:
            if quality["nan_percentage"] < 5:
                # Few NaNs: interpolate
                nan_mask = np.isnan(cleaned_data)
                if np.any(nan_mask) and not np.all(nan_mask):
                    # Linear interpolation for NaN values
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) > 1:
                        cleaned_data[nan_mask] = np.interp(
                            np.where(nan_mask)[0], 
                            valid_indices, 
                            cleaned_data[valid_indices]
                        )
                        cleaning_report["cleaning_applied"].append("NaN interpolation")
                    else:
                        # All data is NaN, fill with zeros
                        cleaned_data[nan_mask] = 0.0
                        cleaning_report["cleaning_applied"].append("NaN zero-fill")
            else:
                # Many NaNs: zero-fill
                cleaned_data[np.isnan(cleaned_data)] = 0.0
                cleaning_report["cleaning_applied"].append("NaN zero-fill")
        
        # Handle Inf values
        if quality["has_inf"]:
            if quality["inf_percentage"] < 5:
                # Few Infs: replace with max/min values
                inf_mask = np.isinf(cleaned_data)
                finite_data = cleaned_data[~inf_mask]
                if len(finite_data) > 0:
                    max_val = np.max(finite_data)
                    min_val = np.min(finite_data)
                    cleaned_data[cleaned_data == np.inf] = max_val
                    cleaned_data[cleaned_data == -np.inf] = min_val
                    cleaning_report["cleaning_applied"].append("Inf replacement")
                else:
                    # All data is Inf, fill with zeros
                    cleaned_data[inf_mask] = 0.0
                    cleaning_report["cleaning_applied"].append("Inf zero-fill")
            else:
                # Many Infs: zero-fill
                cleaned_data[np.isinf(cleaned_data)] = 0.0
                cleaning_report["cleaning_applied"].append("Inf zero-fill")
        
        # Re-validate cleaned data
        final_quality = self._validate_data_quality(cleaned_data)
        cleaning_report["final_quality"] = final_quality
        
        logger.info(f"Data cleaning applied: {cleaning_report['cleaning_applied']}")
        
        return cleaned_data, cleaning_report
    
    def _download_single_segment(self, segment_config: Dict) -> Dict:
        """
        Download a single data segment.
        
        Args:
            segment_config (Dict): Segment configuration
            
        Returns:
            Dict: Download result with metadata
        """
        detector = segment_config['detector']
        start_gps = segment_config['start_gps']
        duration = segment_config.get('duration', 32)
        sample_rate = segment_config.get('sample_rate', 4096)
        label = segment_config['label']
        segment_type = segment_config['type']
        
        segment_id = self._get_segment_id(detector, start_gps, duration)
        
        # Check if already downloaded
        if self._is_already_downloaded(segment_id):
            logger.info(f"Segment {segment_id} already downloaded, skipping")
            return {
                "segment_id": segment_id,
                "status": "skipped",
                "reason": "already_downloaded"
            }
        
        logger.info(f"Downloading {segment_id}: {label} ({segment_type})")
        
        # Download data
        logger.debug(f"Calling _download_strain_data for {segment_id}")
        strain_data = self._download_strain_data(detector, start_gps, duration, sample_rate)
        logger.debug(f"_download_strain_data completed for {segment_id}, got data: {strain_data is not None}")
        
        if strain_data is None:
            return {
                "segment_id": segment_id,
                "status": "failed",
                "reason": "download_failed"
            }
        
        # Validate data quality
        quality = self._validate_data_quality(strain_data)
        
        # Clean data if needed
        if quality["has_nan"] or quality["has_inf"]:
            logger.info(f"Data quality issues detected: {quality['nan_percentage']:.1f}% NaN, {quality['inf_percentage']:.1f}% Inf")
            cleaned_data, cleaning_report = self._clean_data(strain_data, quality)
            
            if cleaned_data is None:
                return {
                    "segment_id": segment_id,
                    "status": "failed",
                    "reason": "data_quality_rejected",
                    "quality": quality,
                    "cleaning_report": cleaning_report
                }
            
            # Use cleaned data
            strain_data = cleaned_data
            quality = cleaning_report["final_quality"]
        else:
            cleaning_report = None
        
        # Save data
        filename = f"{segment_id}.npz"
        filepath = os.path.join(self.raw_data_dir, filename)
        
        try:
            save_data = {
                "strain": strain_data,
                "detector": detector,
                "start_gps": start_gps,
                "duration": duration,
                "sample_rate": sample_rate,
                "label": label,
                "segment_type": segment_type,
                "download_timestamp": datetime.now().isoformat()
            }
            
            # Add quality and cleaning info if available
            if quality:
                save_data["quality"] = quality
            if cleaning_report:
                save_data["cleaning_report"] = cleaning_report
            
            np.savez_compressed(filepath, **save_data)
            
            # Create manifest entry
            manifest_entry = {
                "segment_id": segment_id,
                "detector": detector,
                "start_gps": start_gps,
                "duration": duration,
                "sample_rate": sample_rate,
                "successful": True,
                "file_path": filepath,
                "segment_type": segment_type,
                "label": label,
                "known_event": segment_config.get('known_event'),
                "has_nan_inf": quality["has_nan"] or quality["has_inf"],
                "quality_metrics": quality,
                "download_timestamp": datetime.now().isoformat()
            }
            
            # Add cleaning information if available
            if cleaning_report:
                manifest_entry["cleaning_applied"] = cleaning_report["cleaning_applied"]
                manifest_entry["original_quality"] = cleaning_report["original_quality"]
            
            self.manifest['downloads'].append(manifest_entry)
            
            logger.info(f"Successfully downloaded {segment_id}")
            return {
                "segment_id": segment_id,
                "status": "success",
                "file_path": filepath,
                "quality": quality
            }
            
        except Exception as e:
            logger.error(f"Failed to save {segment_id}: {e}")
            return {
                "segment_id": segment_id,
                "status": "failed",
                "reason": f"save_failed: {e}"
            }
    
    def download_all_segments(self, require_confirmation: bool = True) -> Dict:
        """
        Download all segments defined in the configuration.
        
        Args:
            require_confirmation (bool): Whether to require user confirmation
            
        Returns:
            Dict: Summary of download results
        """
        logger.info("Starting download of all configured segments")
        
        # Get all segments
        all_segments = []
        all_segments.extend(self.config['downloader'].get('noise_segments', []))
        all_segments.extend(self.config['downloader'].get('signal_segments', []))
        
        if not all_segments:
            logger.warning("No segments configured for download")
            return {"status": "no_segments", "total": 0}
        
        logger.info(f"Found {len(all_segments)} segments to download")
        
        # Show summary
        noise_count = len(self.config['downloader'].get('noise_segments', []))
        signal_count = len(self.config['downloader'].get('signal_segments', []))
        logger.info(f"  - Noise segments: {noise_count}")
        logger.info(f"  - Signal segments: {signal_count}")
        
        # Require confirmation if requested
        if require_confirmation:
            print(f"\nAbout to download {len(all_segments)} segments:")
            print(f"   - Noise: {noise_count}")
            print(f"   - Signals: {signal_count}")
            print(f"   - Raw data directory: {self.raw_data_dir}")
            print(f"   - Manifest file: {self.manifest_path}")
            
            response = input("\nContinue with download? (y/N): ").strip().lower()
            if response != 'y':
                logger.info("Download cancelled by user")
                return {"status": "cancelled", "total": len(all_segments)}
        
        # Download segments
        results = {
            "status": "completed",
            "total": len(all_segments),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "results": []
        }
        
        max_concurrent = self.config['downloader'].get('download_parameters', {}).get('max_concurrent', 4)
        
        logger.info(f"Starting download with {max_concurrent} concurrent workers")
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all download tasks
            logger.info(f"Submitting {len(all_segments)} download tasks to thread pool")
            future_to_segment = {}
            for i, segment in enumerate(all_segments):
                segment_id = self._get_segment_id(segment['detector'], segment['start_gps'], segment.get('duration', 32))
                if i < 10 or i % 100 == 0:  # Log first 10 and every 100th
                    logger.info(f"  Submitting task {i+1}/{len(all_segments)}: {segment_id}")
                future = executor.submit(self._download_single_segment, segment)
                future_to_segment[future] = segment
            
            logger.info(f"All {len(future_to_segment)} tasks submitted, waiting for completion...")
            
            # Process completed downloads
            completed_count = 0
            for future in as_completed(future_to_segment):
                completed_count += 1
                segment = future_to_segment[future]
                segment_id = self._get_segment_id(segment['detector'], segment['start_gps'], segment.get('duration', 32))
                
                if completed_count <= 10 or completed_count % 50 == 0:  # Log first 10 and every 50th
                    logger.info(f"Processing completed task {completed_count}/{len(future_to_segment)}: {segment_id}")
                
                try:
                    result = future.result()
                    if completed_count <= 10 or completed_count % 50 == 0:
                        logger.info(f"Task {segment_id} completed with status: {result['status']}")
                    results["results"].append(result)
                    
                    if result["status"] == "success":
                        results["successful"] += 1
                    elif result["status"] == "failed":
                        results["failed"] += 1
                    elif result["status"] == "skipped":
                        results["skipped"] += 1
                        
                except Exception as e:
                    logger.error(f"Download task failed for {segment['label']}: {e}")
                    results["failed"] += 1
                    results["results"].append({
                        "segment_id": segment_id,
                        "status": "failed",
                        "reason": f"task_exception: {e}"
                    })
            
            logger.info(f"All {completed_count} tasks completed")
        
        # Save updated manifest
        self._save_manifest()
        
        # Log summary
        logger.info(f"Download completed:")
        logger.info(f"  Successful: {results['successful']}")
        logger.info(f"  Failed: {results['failed']}")
        logger.info(f"  Skipped: {results['skipped']}")
        
        return results


def main():
    """Main function for standalone downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GWOSC Data Downloader")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    
    args = parser.parse_args()
    
    try:
        downloader = GWOSCDownloader(args.config)
        results = downloader.download_all_segments(require_confirmation=not args.no_confirm)
        
        if results["status"] == "cancelled":
            print("Download cancelled.")
        else:
            print(f"\nDownload Summary:")
            print(f"   Total segments: {results['total']}")
            print(f"   Successful: {results['successful']}")
            print(f"   Failed: {results['failed']}")
            print(f"   Skipped: {results['skipped']}")
            
    except Exception as e:
        logger.error(f"Downloader failed: {e}")
        raise


if __name__ == "__main__":
    main()
