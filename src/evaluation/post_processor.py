"""
Post-Processing Module for Gravitational Wave Hunter v2.0

This module provides post-processing capabilities for anomaly detection results,
including timing analysis, peak detection, and result enhancement.

Author: Jericho Cain
Date: October 2, 2025
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import yaml

from preprocessing import CWTPreprocessor

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processor for anomaly detection results.
    
    This class enhances anomaly detection results with timing information,
    peak detection, and additional analysis. It maps sample indices back
    to time domain and provides comprehensive detection results.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Attributes
    ----------
    config : Dict[str, Any]
        Loaded configuration dictionary
    preprocessor : CWTPreprocessor
        CWT preprocessor for timing analysis
    sample_rate : float
        Sampling rate in Hz
    segment_duration : float
        Duration of each segment in seconds
        
    Examples
    --------
    >>> postprocessor = PostProcessor('config/download_config.yaml')
    >>> enhanced_results = postprocessor.add_timing(detection_results, cwt_data)
    >>> print(f"Detection times: {enhanced_results['detection_times']}")
    """
    
    def __init__(self, config_path: str) -> None:
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Get preprocessing configuration
        preprocessing_config = self.config['preprocessing']['cwt']
        
        # Initialize CWT preprocessor for timing analysis
        self.preprocessor = CWTPreprocessor(
            sample_rate=preprocessing_config['sample_rate'],
            target_height=preprocessing_config['target_height'],
            use_analytic=preprocessing_config['use_analytic'],
            fmin=preprocessing_config['fmin'],
            fmax=preprocessing_config['fmax']
        )
        
        # Get timing parameters
        self.sample_rate = preprocessing_config['sample_rate']
        self.segment_duration = self.config['downloader']['duration']
        
        logger.info(f"Initialized post-processor with sample rate: {self.sample_rate} Hz")
        
    def add_timing(self, detection_results: Dict[str, Any], cwt_data: np.ndarray) -> Dict[str, Any]:
        """
        Add timing information to anomaly detection results.
        
        Parameters
        ----------
        detection_results : Dict[str, Any]
            Results from anomaly detection
        cwt_data : np.ndarray
            CWT data used for detection, shape (n_samples, height, width)
            
        Returns
        -------
        Dict[str, Any]
            Enhanced results with timing information
        """
        logger.info("Adding timing information to detection results...")
        
        # Extract detection information
        predictions = detection_results['predictions']
        reconstruction_errors = detection_results['reconstruction_errors']
        
        # Find anomaly samples
        anomaly_indices = np.where(predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            logger.info("No anomalies detected, returning results without timing")
            return self._add_empty_timing(detection_results)
            
        logger.info(f"Processing timing for {len(anomaly_indices)} detected anomalies")
        
        # Compute timing information
        detection_times = []
        peak_times = []
        confidence_scores = []
        
        for idx in anomaly_indices:
            # Get CWT data for this sample
            sample_cwt = cwt_data[idx]
            
            # Find peak time within the segment
            peak_time = self._find_peak_time(sample_cwt)
            peak_times.append(peak_time)
            
            # Convert sample index to absolute time
            detection_time = self._sample_index_to_time(idx)
            detection_times.append(detection_time)
            
            # Use reconstruction error as confidence score
            confidence = reconstruction_errors[idx]
            confidence_scores.append(confidence)
            
        # Create enhanced results
        enhanced_results = detection_results.copy()
        enhanced_results.update({
            'detection_times': np.array(detection_times),
            'peak_times': np.array(peak_times),
            'confidence_scores': np.array(confidence_scores),
            'anomaly_indices': anomaly_indices,
            'timing_analysis': {
                'sample_rate': self.sample_rate,
                'segment_duration': self.segment_duration,
                'total_anomalies': len(anomaly_indices),
                'time_resolution': 1.0 / self.sample_rate
            }
        })
        
        logger.info(f"Timing analysis complete:")
        logger.info(f"  Detection times: {detection_times[:5]}{'...' if len(detection_times) > 5 else ''}")
        logger.info(f"  Peak times: {peak_times[:5]}{'...' if len(peak_times) > 5 else ''}")
        
        return enhanced_results
        
    def _find_peak_time(self, cwt_scalogram: np.ndarray) -> float:
        """
        Find peak time within a CWT scalogram.
        
        Parameters
        ----------
        cwt_scalogram : np.ndarray
            CWT scalogram of shape (height, width)
            
        Returns
        -------
        float
            Peak time in seconds within the segment
        """
        try:
            # Use the preprocessor's peak detection method
            peak_idx, peak_time = self.preprocessor.find_peak_time(cwt_scalogram)
            return peak_time
        except Exception as e:
            logger.warning(f"Peak detection failed: {e}, using center time")
            # Fallback to center of segment
            return self.segment_duration / 2.0
            
    def _sample_index_to_time(self, sample_idx: int) -> float:
        """
        Convert sample index to absolute time.
        
        Parameters
        ----------
        sample_idx : int
            Sample index
            
        Returns
        -------
        float
            Absolute time in seconds
        """
        # For now, assume samples are sequential in time
        # This could be enhanced to use actual timestamps if available
        return sample_idx * self.segment_duration
        
    def _add_empty_timing(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add empty timing information when no anomalies are detected.
        
        Parameters
        ----------
        detection_results : Dict[str, Any]
            Original detection results
            
        Returns
        -------
        Dict[str, Any]
            Results with empty timing information
        """
        enhanced_results = detection_results.copy()
        enhanced_results.update({
            'detection_times': np.array([]),
            'peak_times': np.array([]),
            'confidence_scores': np.array([]),
            'anomaly_indices': np.array([]),
            'timing_analysis': {
                'sample_rate': self.sample_rate,
                'segment_duration': self.segment_duration,
                'total_anomalies': 0,
                'time_resolution': 1.0 / self.sample_rate
            }
        })
        return enhanced_results
        
    def analyze_detection_patterns(self, enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patterns in detection results.
        
        Parameters
        ----------
        enhanced_results : Dict[str, Any]
            Enhanced results with timing information
            
        Returns
        -------
        Dict[str, Any]
            Pattern analysis results
        """
        if len(enhanced_results['detection_times']) == 0:
            return {'pattern_analysis': 'No anomalies detected'}
            
        detection_times = enhanced_results['detection_times']
        peak_times = enhanced_results['peak_times']
        confidence_scores = enhanced_results['confidence_scores']
        
        # Analyze timing patterns
        time_intervals = np.diff(detection_times)
        peak_time_stats = {
            'mean': np.mean(peak_times),
            'std': np.std(peak_times),
            'min': np.min(peak_times),
            'max': np.max(peak_times)
        }
        
        # Analyze confidence patterns
        confidence_stats = {
            'mean': np.mean(confidence_scores),
            'std': np.std(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores)
        }
        
        # Detect clustering
        clustering_analysis = self._analyze_clustering(detection_times)
        
        pattern_analysis = {
            'total_detections': len(detection_times),
            'time_span': np.max(detection_times) - np.min(detection_times),
            'mean_interval': np.mean(time_intervals) if len(time_intervals) > 0 else 0,
            'peak_time_stats': peak_time_stats,
            'confidence_stats': confidence_stats,
            'clustering_analysis': clustering_analysis
        }
        
        logger.info(f"Pattern analysis complete:")
        logger.info(f"  Total detections: {pattern_analysis['total_detections']}")
        logger.info(f"  Time span: {pattern_analysis['time_span']:.1f} seconds")
        logger.info(f"  Mean peak time: {peak_time_stats['mean']:.3f} seconds")
        
        return {'pattern_analysis': pattern_analysis}
        
    def _analyze_clustering(self, detection_times: np.ndarray) -> Dict[str, Any]:
        """
        Analyze clustering of detection times.
        
        Parameters
        ----------
        detection_times : np.ndarray
            Array of detection times
            
        Returns
        -------
        Dict[str, Any]
            Clustering analysis results
        """
        if len(detection_times) < 2:
            return {'clusters': 0, 'cluster_info': []}
            
        # Simple clustering based on time intervals
        time_intervals = np.diff(detection_times)
        cluster_threshold = self.segment_duration * 2  # 2 segments apart
        
        clusters = []
        current_cluster = [detection_times[0]]
        
        for i, interval in enumerate(time_intervals):
            if interval <= cluster_threshold:
                current_cluster.append(detection_times[i + 1])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [detection_times[i + 1]]
                
        # Add final cluster
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
            
        # Analyze clusters
        cluster_info = []
        for i, cluster in enumerate(clusters):
            cluster_info.append({
                'cluster_id': i,
                'size': len(cluster),
                'start_time': np.min(cluster),
                'end_time': np.max(cluster),
                'duration': np.max(cluster) - np.min(cluster),
                'mean_time': np.mean(cluster)
            })
            
        return {
            'clusters': len(clusters),
            'cluster_info': cluster_info,
            'clustering_threshold': cluster_threshold
        }
        
    def generate_detection_report(self, enhanced_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable detection report.
        
        Parameters
        ----------
        enhanced_results : Dict[str, Any]
            Enhanced results with timing information
            
        Returns
        -------
        str
            Formatted detection report
        """
        if len(enhanced_results['detection_times']) == 0:
            return "No gravitational wave signals detected in the analyzed data."
            
        detection_times = enhanced_results['detection_times']
        peak_times = enhanced_results['peak_times']
        confidence_scores = enhanced_results['confidence_scores']
        timing_analysis = enhanced_results['timing_analysis']
        
        report = []
        report.append("GRAVITATIONAL WAVE DETECTION REPORT")
        report.append("=" * 50)
        report.append(f"Total detections: {len(detection_times)}")
        report.append(f"Sample rate: {timing_analysis['sample_rate']} Hz")
        report.append(f"Segment duration: {timing_analysis['segment_duration']} seconds")
        report.append(f"Time resolution: {timing_analysis['time_resolution']:.6f} seconds")
        report.append("")
        
        # Detection summary
        report.append("DETECTION SUMMARY:")
        report.append(f"  First detection: {detection_times[0]:.3f} seconds")
        report.append(f"  Last detection: {detection_times[-1]:.3f} seconds")
        report.append(f"  Time span: {detection_times[-1] - detection_times[0]:.3f} seconds")
        report.append("")
        
        # Peak time analysis
        report.append("PEAK TIME ANALYSIS:")
        report.append(f"  Mean peak time: {np.mean(peak_times):.3f} seconds")
        report.append(f"  Peak time std: {np.std(peak_times):.3f} seconds")
        report.append(f"  Peak time range: {np.min(peak_times):.3f} - {np.max(peak_times):.3f} seconds")
        report.append("")
        
        # Confidence analysis
        report.append("CONFIDENCE ANALYSIS:")
        report.append(f"  Mean confidence: {np.mean(confidence_scores):.6f}")
        report.append(f"  Confidence std: {np.std(confidence_scores):.6f}")
        report.append(f"  Confidence range: {np.min(confidence_scores):.6f} - {np.max(confidence_scores):.6f}")
        report.append("")
        
        # Individual detections
        report.append("INDIVIDUAL DETECTIONS:")
        for i, (det_time, peak_time, confidence) in enumerate(zip(detection_times, peak_times, confidence_scores)):
            report.append(f"  Detection {i+1}:")
            report.append(f"    Time: {det_time:.3f} seconds")
            report.append(f"    Peak: {peak_time:.3f} seconds")
            report.append(f"    Confidence: {confidence:.6f}")
            
        return "\n".join(report)
