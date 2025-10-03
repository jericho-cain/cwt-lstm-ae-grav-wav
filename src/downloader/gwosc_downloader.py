#!/usr/bin/env python3
"""
Clean GWOSC Data Downloader

This module provides a clean, documentation-guided approach to downloading
gravitational wave data from GWOSC using the official APIs.

Based on: https://gwosc.readthedocs.io/en/stable/datasets.html
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# GWOSC imports
from gwosc.datasets import find_datasets, run_segment, query_events, event_gps
from gwosc.locate import get_urls
from gwosc.timeline import get_segments

# GWpy for noise data
try:
    from gwpy.timeseries import TimeSeries
    GWPY_AVAILABLE = True
except ImportError:
    GWPY_AVAILABLE = False
    logging.warning("gwpy not available - noise downloads will be limited")

logger = logging.getLogger(__name__)


class CleanGWOSCDownloader:
    """
    Clean GWOSC data downloader using official APIs.
    
    This downloader separates signal and noise downloads:
    - Signals: Uses datasets.find_datasets() and get_urls() for specific events
    - Noise: Uses TimeSeries.fetch_open_data() for science-mode segments
    """
    
    def __init__(self, config_path: str):
        """Initialize downloader with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.manifest = self._load_manifest()
        
        # Setup directories
        self.raw_dir = Path(self.config['downloader']['data_directories']['raw_data'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Available runs from GWOSC
        self.available_runs = ['O1', 'O2', 'O3a', 'O3b', 'O4a']
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_manifest(self) -> Dict[str, Any]:
        """Load or create download manifest."""
        manifest_path = Path(self.config['downloader']['data_directories']['manifest_file'])
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'downloads': [],
                'metadata': {
                    'created': time.time(),
                    'config_file': str(self.config_path)
                }
            }
    
    def _save_manifest(self):
        """Save download manifest."""
        manifest_path = Path(self.config['downloader']['data_directories']['manifest_file'])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def _get_available_events(self, runs: Optional[List[str]] = None, 
                            confident_only: bool = True) -> List[str]:
        """Get list of available events from GWOSC."""
        if runs is None:
            runs = self.available_runs
        
        if confident_only:
            # Use catalog-specific confident events
            events = []
            catalogs = ['GWTC-1-confident', 'GWTC-2.1-confident', 'GWTC-3-confident', 'GWTC-4.0']
            for catalog in catalogs:
                try:
                    catalog_events = find_datasets(catalog=catalog)
                    # Filter out non-event entries
                    catalog_events = [e for e in catalog_events if not e.startswith('GWTC') and not e.startswith('O') and not e.startswith('S')]
                    events.extend(catalog_events)
                    logger.info(f"Found {len(catalog_events)} confident events in {catalog}")
                except Exception as e:
                    logger.error(f"Failed to get events from {catalog}: {e}")
        else:
            # Original method - all events
            events = []
            for run in runs:
                try:
                    start, end = run_segment(run)
                    run_events = query_events([
                        f"gps-time >= {start}",
                        f"gps-time <= {end}"
                    ])
                    events.extend(run_events)
                    logger.info(f"Found {len(run_events)} events in {run}")
                except Exception as e:
                    logger.error(f"Failed to get events for {run}: {e}")
        
        return list(set(events))  # Remove duplicates
    
    def _download_signal_segment(self, detector: str, event: str, 
                                start_gps: int, duration: int) -> Optional[np.ndarray]:
        """Download signal segment using event-specific HDF5 tiles."""
        try:
            # Get HDF5 tile URLs for this event
            urls = get_urls(
                detector=detector,
                start=start_gps,
                end=start_gps + duration,
                sample_rate=self.config['sample_rate'],
                format='hdf5'
            )
            
            if not urls:
                logger.error(f"No HDF5 tiles found for {event} {detector}")
                return None
            
            # Download and parse the first available tile
            response = requests.get(urls[0], timeout=60)
            response.raise_for_status()
            
            # Parse HDF5 data
            import h5py
            import io
            
            with h5py.File(io.BytesIO(response.content), 'r') as f:
                strain_data = f["/strain/Strain"][()]
            
            # Convert to float32 and ensure correct length
            strain_data = strain_data.astype(np.float32)
            expected_length = duration * self.config['downloader']['sample_rate']
            
            if len(strain_data) != expected_length:
                logger.warning(f"Length mismatch: {len(strain_data)} vs {expected_length}")
                # Resample if needed
                from scipy import signal
                strain_data = signal.resample(strain_data, expected_length).astype(np.float32)
            
            logger.info(f"Downloaded {event} {detector}: {len(strain_data)} samples")
            return strain_data
            
        except Exception as e:
            logger.error(f"Failed to download {event} {detector}: {e}")
            return None
    
    def _download_noise_segment(self, detector: str, start_gps: int, 
                               duration: int) -> Optional[np.ndarray]:
        """Download noise segment using TimeSeries.fetch_open_data."""
        if not GWPY_AVAILABLE:
            logger.error("gwpy not available - cannot download noise data")
            return None
        
        try:
            end_gps = start_gps + duration
            
            # Check if this time is in science mode
            segments = get_segments(f"{detector}_DATA", start_gps - 64, end_gps + 64)
            if not segments:
                logger.error(f"No science-mode segments for {detector} near {start_gps}")
                return None
            
            # Check if window is covered by science segments
            coverage = 0
            for seg_start, seg_end in segments:
                overlap_start = max(start_gps, seg_start)
                overlap_end = min(end_gps, seg_end)
                if overlap_start < overlap_end:
                    coverage += overlap_end - overlap_start
            
            total_duration = duration
            if coverage / total_duration < self.config.get('science_min_coverage', 0.8):
                logger.error(f"Insufficient science coverage: {coverage/total_duration:.2f}")
                return None
            
            # Download using gwpy
            ts = TimeSeries.fetch_open_data(
                detector, start_gps, end_gps, 
                cache=True, host='https://gwosc.org'
            )
            
            strain_data = np.asarray(ts.value, dtype=np.float32)
            
            # Ensure correct length
            expected_length = duration * self.config['downloader']['sample_rate']
            if len(strain_data) != expected_length:
                from scipy import signal
                strain_data = signal.resample(strain_data, expected_length).astype(np.float32)
            
            # Check for non-finite values
            if not np.isfinite(strain_data).all():
                logger.warning("Non-finite samples detected, zero-filling")
                strain_data = np.nan_to_num(strain_data, copy=False)
            
            logger.info(f"Downloaded noise {detector}: {len(strain_data)} samples")
            return strain_data
            
        except Exception as e:
            logger.error(f"Failed to download noise {detector}: {e}")
            return None
    
    def _save_segment(self, detector: str, start_gps: int, duration: int,
                     strain_data: np.ndarray, segment_type: str, 
                     label: str, event: Optional[str] = None) -> bool:
        """Save downloaded segment to disk."""
        try:
            filename = f"{detector}_{start_gps}_{duration}s.npz"
            filepath = self.raw_dir / filename
            
            # Prepare data for saving
            save_data = {
                'strain': strain_data,
                'detector': detector,
                'start_gps': start_gps,
                'duration': duration,
                'sample_rate': self.config['downloader']['sample_rate'],
                'segment_type': segment_type,
                'label': label
            }
            
            if event:
                save_data['event'] = event
            
            np.savez_compressed(filepath, **save_data)
            
            # Update manifest
            self.manifest['downloads'].append({
                'segment_id': filename.replace('.npz', ''),
                'detector': detector,
                'start_gps': start_gps,
                'duration': duration,
                'segment_type': segment_type,
                'label': label,
                'event': event,
                'successful': True,
                'timestamp': time.time()
            })
            
            logger.info(f"Saved {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save segment: {e}")
            return False
    
    def download_signals(self, events: Optional[List[str]] = None, 
                        detectors: Optional[List[str]] = None) -> Dict[str, int]:
        """Download signal segments for specified events."""
        if events is None:
            events = self._get_available_events()
        
        if detectors is None:
            detectors = self.config['downloader']['signals'].get('detectors', ['H1'])
        
        duration = self.config['downloader']['duration']
        results = {'successful': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Downloading {len(events)} events for {len(detectors)} detectors")
        
        for event in events:
            try:
                event_gps_time = event_gps(event)
                start_gps = int(event_gps_time)
                
                for detector in detectors:
                    segment_id = f"{detector}_{start_gps}_{duration}s"
                    
                    # Check if already downloaded
                    if any(d.get('segment_id') == segment_id for d in self.manifest['downloads']):
                        logger.info(f"Skipping {segment_id} - already downloaded")
                        results['skipped'] += 1
                        continue
                    
                    # Download segment
                    strain_data = self._download_signal_segment(
                        detector, event, start_gps, duration
                    )
                    
                    if strain_data is not None:
                        success = self._save_segment(
                            detector, start_gps, duration, strain_data,
                            'signal', f"{event}_{detector}", event
                        )
                        if success:
                            results['successful'] += 1
                        else:
                            results['failed'] += 1
                    else:
                        results['failed'] += 1
                        
            except Exception as e:
                logger.error(f"Failed to process event {event}: {e}")
                results['failed'] += 1
        
        return results
    
    def download_noise(self, runs: Optional[List[str]] = None,
                      detectors: Optional[List[str]] = None,
                      segments_per_run: int = 10) -> Dict[str, int]:
        """Download noise segments from science-mode data."""
        if not GWPY_AVAILABLE:
            logger.error("gwpy not available - cannot download noise")
            return {'successful': 0, 'failed': 0, 'skipped': 0}
        
        if runs is None:
            runs = self.config['downloader']['noise'].get('runs', ['O1', 'O2', 'O3a'])
        
        if detectors is None:
            detectors = self.config['downloader']['noise'].get('detectors', ['H1'])
        
        duration = self.config['downloader']['duration']
        results = {'successful': 0, 'failed': 0, 'skipped': 0}
        
        logger.info(f"Downloading noise from {len(runs)} runs for {len(detectors)} detectors")
        
        for run in runs:
            try:
                start, end = run_segment(run)
                logger.info(f"Processing {run}: {start} - {end}")
                
                for detector in detectors:
                    # Get science-mode segments
                    segments = get_segments(f"{detector}_DATA", start, end)
                    if not segments:
                        logger.warning(f"No science segments for {detector} in {run}")
                        continue
                    
                    # Filter segments that are long enough for our duration
                    valid_segments = [(s, e) for s, e in segments if (e - s) >= duration]
                    logger.info(f"{detector} in {run}: {len(segments)} total segments, {len(valid_segments)} long enough for {duration}s")
                    
                    if not valid_segments:
                        logger.warning(f"No segments long enough for {detector} in {run}")
                        continue
                    
                    # Sample random times from valid segments
                    import random
                    sampled_times = []
                    
                    segments_per_run = self.config['downloader']['noise'].get('segments_per_run', segments_per_run)
                    for _ in range(segments_per_run):
                        # Pick a random valid segment
                        seg_start, seg_end = random.choice(valid_segments)
                        
                        # Pick a random time within the segment
                        max_start = seg_end - duration
                        sample_time = random.randint(seg_start, max_start)
                        sampled_times.append(sample_time)
                    
                    # Download sampled segments
                    for start_gps in sampled_times:
                        segment_id = f"{detector}_{start_gps}_{duration}s"
                        
                        # Check if already downloaded
                        if any(d.get('segment_id') == segment_id for d in self.manifest['downloads']):
                            results['skipped'] += 1
                            continue
                        
                        # Download segment
                        strain_data = self._download_noise_segment(
                            detector, start_gps, duration
                        )
                        
                        if strain_data is not None:
                            success = self._save_segment(
                                detector, start_gps, duration, strain_data,
                                'noise', f"{run}_{detector}_{start_gps}"
                            )
                            if success:
                                results['successful'] += 1
                            else:
                                results['failed'] += 1
                        else:
                            results['failed'] += 1
                            
            except Exception as e:
                logger.error(f"Failed to process run {run}: {e}")
                results['failed'] += 1
        
        return results
    
    def download_all(self, **kwargs) -> Dict[str, Any]:
        """Download both signals and noise."""
        logger.info("Starting full download process")
        
        # Download signals
        signal_results = self.download_signals(**kwargs)
        
        # Download noise
        noise_results = self.download_noise(**kwargs)
        
        # Save manifest
        self._save_manifest()
        
        # Return combined results
        return {
            'signals': signal_results,
            'noise': noise_results,
            'total': {
                'successful': signal_results['successful'] + noise_results['successful'],
                'failed': signal_results['failed'] + noise_results['failed'],
                'skipped': signal_results['skipped'] + noise_results['skipped']
            }
        }


def main():
    """Main entry point for the downloader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean GWOSC Data Downloader")
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--signals-only', action='store_true', help='Download only signals')
    parser.add_argument('--noise-only', action='store_true', help='Download only noise')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize downloader
    downloader = CleanGWOSCDownloader(args.config)
    
    # Confirm before starting
    if not args.no_confirm:
        response = input("Start download? (y/N): ")
        if response.lower() != 'y':
            print("Download cancelled")
            return
    
    # Download based on options
    if args.signals_only:
        results = downloader.download_signals()
    elif args.noise_only:
        results = downloader.download_noise()
    else:
        results = downloader.download_all()
    
    # Print results
    print("\nDownload Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
