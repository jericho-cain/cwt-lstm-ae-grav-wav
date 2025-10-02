"""
Configuration Validator for Gravitational Wave Hunter v2.0

This module validates YAML configuration files against the defined schema
to ensure proper configuration before running any operations.
"""

import yaml
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates configuration files against the defined schema.
    
    This class provides validation for different types of configuration files
    used in the gravitational wave detection system, including downloader,
    training, and preprocessing configurations.
    
    Parameters
    ----------
    schema_path : str, optional
        Path to schema definition file, by default "config/schema.yaml"
        
    Attributes
    ----------
    schema_path : str
        Path to schema definition file
    schema : Dict[str, Any]
        Loaded schema dictionary
        
    Examples
    --------
    >>> validator = ConfigValidator()
    >>> result = validator.validate_downloader_config('config/download_config.yaml')
    >>> if result['valid']:
    ...     print("Configuration is valid")
    """
    
    def __init__(self, schema_path: str = "config/schema.yaml") -> None:
        """
        Initialize validator with schema.
        
        Parameters
        ----------
        schema_path : str, optional
            Path to schema definition file, by default "config/schema.yaml"
            
        Raises
        ------
        FileNotFoundError
            If schema file does not exist
        yaml.YAMLError
            If schema file is malformed
        """
        self.schema_path = schema_path
        self.schema = self._load_schema()
    
    def _load_schema(self) -> Dict:
        """Load the schema definition."""
        try:
            with open(self.schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            logger.info(f"Loaded schema from {self.schema_path}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def validate_downloader_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a downloader configuration file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Validation results with 'valid' boolean and 'errors' list
        """
        logger.info(f"Validating downloader config: {config_path}")
        
        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Failed to load config file: {e}"],
                "warnings": []
            }
        
        # Get downloader schema
        downloader_schema = self.schema.get('downloader_schema', {})
        if not downloader_schema:
            return {
                "valid": False,
                "errors": ["No downloader schema found"],
                "warnings": []
            }
        
        # Validate configuration
        errors = []
        warnings = []
        
        # Check required top-level keys
        if 'downloader' not in config:
            errors.append("Missing required 'downloader' section")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        downloader_config = config['downloader']
        
        # Validate required fields
        required_fields = downloader_schema.get('required', [])
        for field in required_fields:
            if field not in downloader_config:
                errors.append(f"Missing required field: {field}")
        
        # Validate data directories
        if 'data_directories' in downloader_config:
            data_dirs = downloader_config['data_directories']
            if 'raw_data' not in data_dirs:
                errors.append("Missing 'raw_data' in data_directories")
            if 'manifest_file' not in data_dirs:
                errors.append("Missing 'manifest_file' in data_directories")
        
        # Validate detectors
        if 'detectors' in downloader_config:
            valid_detectors = ['H1', 'L1', 'V1', 'G1', 'K1']
            for detector in downloader_config['detectors']:
                if detector not in valid_detectors:
                    errors.append(f"Invalid detector: {detector}. Must be one of {valid_detectors}")
        
        # Validate observing runs
        if 'observing_runs' in downloader_config:
            valid_runs = ['O1', 'O2', 'O3a', 'O3b', 'O4a', 'S5', 'S6']
            for run in downloader_config['observing_runs']:
                if run not in valid_runs:
                    errors.append(f"Invalid observing run: {run}. Must be one of {valid_runs}")
        
        # Validate segments
        self._validate_segments(downloader_config.get('noise_segments', []), 'noise', errors, warnings)
        self._validate_segments(downloader_config.get('signal_segments', []), 'signal', errors, warnings)
        
        # Validate download parameters
        if 'download_params' in downloader_config:
            params = downloader_config['download_params']
            if 'segment_duration' in params and params['segment_duration'] <= 0:
                errors.append("segment_duration must be positive")
            if 'sample_rate' in params and params['sample_rate'] <= 0:
                errors.append("sample_rate must be positive")
            if 'retry_attempts' in params:
                retry = params['retry_attempts']
                if retry < 1 or retry > 10:
                    errors.append("retry_attempts must be between 1 and 10")
        
        # Check for potential issues
        if not downloader_config.get('noise_segments') and not downloader_config.get('signal_segments'):
            warnings.append("No segments configured for download")
        
        if downloader_config.get('safety', {}).get('require_confirmation', True) is False:
            warnings.append("Confirmation disabled - downloads will proceed automatically")
        
        valid = len(errors) == 0
        
        logger.info(f"Config validation {'PASSED' if valid else 'FAILED'}")
        if errors:
            logger.error(f"Validation errors: {errors}")
        if warnings:
            logger.warning(f"Validation warnings: {warnings}")
        
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "config": config
        }
    
    def _validate_segments(self, segments: List[Dict], segment_type: str, errors: List[str], warnings: List[str]):
        """
        Validate a list of segments.
        
        Args:
            segments (List[Dict]): List of segment configurations
            segment_type (str): Type of segments ('noise' or 'signal')
            errors (List[str]): List to append errors to
            warnings (List[str]): List to append warnings to
        """
        required_fields = ['start_gps', 'end_gps', 'label', 'type', 'detector']
        
        for i, segment in enumerate(segments):
            # Check required fields
            for field in required_fields:
                if field not in segment:
                    errors.append(f"{segment_type}_segments[{i}]: Missing required field '{field}'")
            
            # Validate GPS times
            if 'start_gps' in segment and 'end_gps' in segment:
                start = segment['start_gps']
                end = segment['end_gps']
                if start >= end:
                    errors.append(f"{segment_type}_segments[{i}]: start_gps must be less than end_gps")
                if end - start > 3600:  # 1 hour
                    warnings.append(f"{segment_type}_segments[{i}]: Very long segment ({end-start}s)")
            
            # Validate detector
            if 'detector' in segment:
                valid_detectors = ['H1', 'L1', 'V1', 'G1', 'K1']
                if segment['detector'] not in valid_detectors:
                    errors.append(f"{segment_type}_segments[{i}]: Invalid detector '{segment['detector']}'")
            
            # Validate segment type
            if 'type' in segment:
                if segment['type'] != segment_type:
                    errors.append(f"{segment_type}_segments[{i}]: Type mismatch - expected '{segment_type}', got '{segment['type']}'")
            
            # Signal segments must have known_event
            if segment_type == 'signal' and 'known_event' not in segment:
                errors.append(f"{segment_type}_segments[{i}]: Signal segments must specify 'known_event'")
    
    def validate_training_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a training configuration file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # TODO: Implement training config validation
        logger.info(f"Training config validation not yet implemented: {config_path}")
        return {
            "valid": True,
            "errors": [],
            "warnings": ["Training config validation not implemented"],
            "config": {}
        }
    
    def validate_cwt_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate a CWT preprocessing configuration file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Validation results
        """
        # TODO: Implement CWT config validation
        logger.info(f"CWT config validation not yet implemented: {config_path}")
        return {
            "valid": True,
            "errors": [],
            "warnings": ["CWT config validation not implemented"],
            "config": {}
        }


def main():
    """Main function for config validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Validator")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--type", choices=['downloader', 'training', 'cwt'], 
                       default='downloader', help="Type of configuration to validate")
    parser.add_argument("--schema", default="config/schema.yaml", help="Path to schema file")
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.schema)
    
    if args.type == 'downloader':
        result = validator.validate_downloader_config(args.config)
    elif args.type == 'training':
        result = validator.validate_training_config(args.config)
    elif args.type == 'cwt':
        result = validator.validate_cwt_config(args.config)
    
    if result['valid']:
        print("✅ Configuration is valid")
        if result['warnings']:
            print("⚠️ Warnings:")
            for warning in result['warnings']:
                print(f"   - {warning}")
    else:
        print("❌ Configuration is invalid")
        print("Errors:")
        for error in result['errors']:
            print(f"   - {error}")
        exit(1)


if __name__ == "__main__":
    main()
